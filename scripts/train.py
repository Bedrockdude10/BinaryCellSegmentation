#!/usr/bin/env python3
# scripts/train.py
"""
Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --override configs/ablations/dice.yaml
    python scripts/train.py --config configs/default.yaml --resume outputs/unet_swin_pretrained/best.pt
"""
import argparse
import logging
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.data.dataset import PanNukeDataset
from src.data.monuseg import MoNuSegDataset
from src.data.features import binary_mask
from src.data.preprocessing import PreprocessedDataset
from src.models.encoders import SwinEncoder, VGGEncoder, ResNetEncoder
from src.models.unet import UNet
from src.losses.loss import DiceLoss
from src.evaluation.metrics import dice, iou
from src.evaluation.evaluator import evaluate
from src.train import train_one_epoch


# ── Builders (plain functions, no registries) ────────────────────────────────

def resolve_device(s):
    if s == "auto":
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return s


def build_model(cfg):
    arch = cfg["model"]["architecture"]
    params = cfg["model"].get("params", {})
    in_ch = params.get("in_channels", 3)
    out_ch = params.get("out_channels", 1)

    encoders = {
    "vgg": VGGEncoder,
    "resnet": ResNetEncoder,
    "swin": SwinEncoder,
    "swin_pretrained": lambda in_channels: SwinEncoder(in_channels=in_channels, pretrained=True),
    }
    if arch in encoders:
        encoder = encoders[arch](in_channels=in_ch)
        return UNet(encoder, out_channels=out_ch)
    raise ValueError(f"Unknown architecture: {arch}")


def build_loss(cfg):
    name = cfg["loss"] if isinstance(cfg["loss"], str) else cfg["loss"]["function"]
    if name == "bce_logits":
        return torch.nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {name}")


def build_optimizer(cfg, model):
    o = cfg["optimizer"]
    params = {k: v for k, v in o.items() if k != "type"}
    if o["type"] == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    if o["type"] == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    raise ValueError(f"Unknown optimizer: {o['type']}")


def build_scheduler(cfg, optimizer):
    s = cfg.get("scheduler")
    if not s or not s.get("type"):
        return None
    if s["type"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, s["step_size"], s["gamma"])
    if s["type"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["training"]["epochs"])
    return None


class BinaryMaskFeature:
    def __call__(self, sample):
        return {"image": sample["image"], "mask": binary_mask(sample["mask"])}


def make_feature_fn(cfg):
    dataset = cfg["data"].get("dataset", "pannuke")
    if dataset == "monuseg":
        return None  # MoNuSegDataset already returns a binary mask
    return BinaryMaskFeature()


def _build_raw_dataset(cfg, split_key):
    d = cfg["data"]
    dataset = d.get("dataset", "pannuke")
    split = d[split_key]
    if dataset == "monuseg":
        return MoNuSegDataset(split=split, data_root=d["data_root"])
    return PanNukeDataset(split)


def build_loaders(cfg):
    from torch.utils.data import random_split
    d = cfg["data"]
    p = cfg["preprocessing"]
    feature_fn = make_feature_fn(cfg)
    dataset = d.get("dataset", "pannuke")

    if dataset == "monuseg":
        full_train = _build_raw_dataset(cfg, "train_split")
        val_fraction = d.get("val_fraction", 0.2)
        raw_train, raw_val = full_train.image_level_split(val_fraction=val_fraction)
        train_ds = PreprocessedDataset(raw_train, p, is_train=True, feature_fn=feature_fn)
        val_ds = PreprocessedDataset(raw_val, p, is_train=False, feature_fn=feature_fn)
    else:
        train_ds = PreprocessedDataset(_build_raw_dataset(cfg, "train_split"), p, is_train=True, feature_fn=feature_fn)
        val_ds = PreprocessedDataset(_build_raw_dataset(cfg, "val_split"), p, is_train=False, feature_fn=feature_fn)

    kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], pin_memory=False)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. outputs/exp/best.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    exp = cfg["experiment"]
    out_dir = Path(exp["output_dir"]) / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "train.log")],
    )
    log = logging.getLogger(__name__)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    torch.manual_seed(exp.get("seed", 42))
    device = resolve_device(exp.get("device", "auto"))
    log.info(f"Experiment: {exp['name']} | Device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    metric_fns = {"dice": dice, "iou": iou}

    log.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    best = -float("inf")
    stale = 0
    start_epoch = 1
    es = cfg["training"].get("early_stopping", {})
    patience = es.get("patience", float("inf"))
    es_metric = es.get("metric", "dice")

    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best", -float("inf"))
        # Fast-forward scheduler to match resumed epoch
        if scheduler:
            for _ in range(ckpt["epoch"]):
                scheduler.step()
        log.info(f"Resumed from epoch {ckpt['epoch']} (best {es_metric}: {best:.4f})")

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        log.info(f"Epoch {epoch}")
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                               cfg["training"].get("log_interval", 10))

        val = evaluate(model, val_loader, metric_fns, device=device)
        log.info(f"  Loss: {loss:.4f} | Val: {val}")

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Dice/val", val["dice"], epoch)
        writer.add_scalar("IoU/val", val["iou"], epoch)

        if val[es_metric] > best:
            best = val[es_metric]
            stale = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "best": best}, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

        if scheduler:
            scheduler.step()

        if epoch % cfg["training"].get("checkpoint_interval", 10) == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()}, out_dir / f"epoch_{epoch}.pt")

    writer.close()
    log.info(f"Done. Best {es_metric}: {best:.4f}")


if __name__ == "__main__":
    main()