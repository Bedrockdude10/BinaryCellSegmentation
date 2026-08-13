# src/builders.py
"""Config -> object builders, shared by training, evaluation and benchmarking.

Extracted from ``scripts/train.py`` so the sweep runner, the single-run CLI and
``scripts/evaluate.py`` all construct models and loaders through one code path.
The model / loss / optimizer / scheduler builders are unchanged from the
reviewed experiment; only the data-loading side gained seeding and caching.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.cache import CachedPatchDataset, cache_is_complete, read_meta
from src.data.features import binary_mask
from src.data.preprocessing import PreprocessedDataset
from src.losses.loss import DiceLoss
from src.models.encoders import ResNetEncoder, SwinEncoder, VGGEncoder
from src.models.unet import UNet
from src.seeding import make_generator, seed_worker
from src.splits import (
    monuseg_fingerprint, monuseg_image_level_split, pannuke_fingerprint,
)

DEFAULT_SPLIT_SEED = 42


# ── Device / model / loss / optim / sched (unchanged behaviour) ───────────────

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
    """Raw PanNuke yields a 5-channel mask that must be collapsed to binary.

    MoNuSeg and every flat cache already store a binary mask, so they need no
    feature fn.
    """
    if cfg["data"].get("dataset", "pannuke") == "monuseg":
        return None
    if _pannuke_cache_dir(cfg, cfg["data"]["train_split"]) is not None:
        return None
    return BinaryMaskFeature()


# ── Cache location ───────────────────────────────────────────────────────────

def cache_root(cfg) -> Path:
    """Where flat caches live. Slurm tasks point this at node-local NVMe."""
    return Path(cfg["data"].get("cache_root", cfg["data"].get("data_root", "data")))


def pannuke_cache_path(root, fold) -> Path:
    return Path(root) / f"pannuke_flat_{fold}"


def monuseg_cache_path(root, split, patch_size=256) -> Path:
    return Path(root) / f"monuseg_flat_{split}_{patch_size}"


def _pannuke_cache_dir(cfg, fold):
    d = pannuke_cache_path(cache_root(cfg), fold)
    return d if cache_is_complete(d) else None


def _monuseg_cache_dir(cfg, split):
    d = monuseg_cache_path(cache_root(cfg), split)
    return d if cache_is_complete(d) else None


# ── Datasets ─────────────────────────────────────────────────────────────────

def _raw_pannuke(split):
    from src.data.dataset import PanNukeDataset
    return PanNukeDataset(split)


def _raw_monuseg(cfg, split, patch_indices=None):
    from src.data.monuseg import MoNuSegDataset
    return MoNuSegDataset(split=split, data_root=cfg["data"]["data_root"],
                          patch_indices=patch_indices)


def _monuseg_counts(cfg, split):
    """Patches-per-source-image for a MoNuSeg split, from cache or raw dir."""
    cached = _monuseg_cache_dir(cfg, split)
    if cached is not None:
        return read_meta(cached)["patch_counts"]
    return _raw_monuseg(cfg, split).patch_counts


def _pannuke_len(cfg, fold) -> int:
    cached = _pannuke_cache_dir(cfg, fold)
    if cached is not None:
        return int(read_meta(cached)["n"])
    return len(_raw_pannuke(fold))


def split_fingerprint(cfg, split_seed: int = DEFAULT_SPLIT_SEED) -> dict:
    """The split, as a hashable description. One code path for runs and manifest."""
    d = cfg["data"]
    if d.get("dataset", "pannuke") == "monuseg":
        return monuseg_fingerprint(
            _monuseg_stems(cfg, "train"), _monuseg_counts(cfg, "train"),
            d.get("val_fraction", 0.2), split_seed,
            _monuseg_counts(cfg, "test"), _monuseg_stems(cfg, "test"),
        )
    folds = (d["train_split"], d["val_split"], d["test_split"])
    return pannuke_fingerprint(*folds, {f: _pannuke_len(cfg, f) for f in folds})


def _monuseg_stems(cfg, split):
    """Source image names in cache order.

    Prefers the flat cache's meta so the source TIFs need not be staged to the
    compute node just to fingerprint the split.
    """
    cached = _monuseg_cache_dir(cfg, split)
    if cached is not None:
        stems = read_meta(cached).get("source_images")
        if stems:
            return stems
    from src.splits import monuseg_source_stems
    return monuseg_source_stems(cfg["data"]["data_root"], split)


# ── Loaders ──────────────────────────────────────────────────────────────────

def _loader_kwargs(cfg, device):
    """Throughput-only DataLoader settings — none of these change arithmetic.

    ``pin_memory`` affects host->device transfer, ``persistent_workers`` and
    ``prefetch_factor`` affect when work happens, not what it computes.
    """
    d = cfg["data"]
    perf = cfg.get("performance", {})
    workers = int(perf.get("num_workers", d.get("num_workers", 4)))
    kw = dict(
        batch_size=d["batch_size"],
        num_workers=workers,
        pin_memory=bool(perf.get("pin_memory", True)) and device == "cuda",
        worker_init_fn=seed_worker,
    )
    if workers > 0:
        kw["persistent_workers"] = bool(perf.get("persistent_workers", True))
        kw["prefetch_factor"] = int(perf.get("prefetch_factor", 4))
    return kw


def build_loaders(cfg, train_seed: int, split_seed: int = DEFAULT_SPLIT_SEED,
                  device: str = "cpu"):
    """Train and val loaders plus the split fingerprint detail.

    The loader generator is seeded from ``train_seed`` so shuffle order is
    reproducible and independent of how many draws model init consumed.
    """
    d = cfg["data"]
    p = cfg["preprocessing"]
    dataset = d.get("dataset", "pannuke")
    feature_fn = make_feature_fn(cfg)

    if dataset == "monuseg":
        val_fraction = d.get("val_fraction", 0.2)
        counts = _monuseg_counts(cfg, "train")
        train_idx, val_idx, _ = monuseg_image_level_split(counts, val_fraction, split_seed)
        cached = _monuseg_cache_dir(cfg, "train")
        if cached is not None:
            raw_train = CachedPatchDataset(cached, train_idx)
            raw_val = CachedPatchDataset(cached, val_idx)
        else:
            raw_train = _raw_monuseg(cfg, "train", train_idx)
            raw_val = _raw_monuseg(cfg, "train", val_idx)
    else:
        train_fold, val_fold = d["train_split"], d["val_split"]
        raw_train = (CachedPatchDataset(_pannuke_cache_dir(cfg, train_fold))
                     if _pannuke_cache_dir(cfg, train_fold) else _raw_pannuke(train_fold))
        raw_val = (CachedPatchDataset(_pannuke_cache_dir(cfg, val_fold))
                   if _pannuke_cache_dir(cfg, val_fold) else _raw_pannuke(val_fold))

    split_detail = split_fingerprint(cfg, split_seed)

    train_ds = PreprocessedDataset(raw_train, p, is_train=True, feature_fn=feature_fn)
    val_ds = PreprocessedDataset(raw_val, p, is_train=False, feature_fn=feature_fn)

    kw = _loader_kwargs(cfg, device)
    generator = make_generator(train_seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    return train_loader, val_loader, split_detail, generator


def build_test_loader(cfg, device: str = "cpu", split: str | None = None):
    """Test loader plus ``patches_per_image`` for MoNuSeg's per-image protocol.

    Order matters here: per-image Dice slices the concatenated predictions by
    patch count, so the loader must not shuffle and the cache must be in build
    order.
    """
    d = cfg["data"]
    dataset = d.get("dataset", "pannuke")
    split = split or d["test_split"]
    feature_fn = make_feature_fn(cfg)

    patches_per_image = None
    if dataset == "monuseg":
        cached = _monuseg_cache_dir(cfg, split)
        raw = CachedPatchDataset(cached) if cached is not None else _raw_monuseg(cfg, split)
        patches_per_image = _monuseg_counts(cfg, split)
    else:
        cached = _pannuke_cache_dir(cfg, split)
        raw = CachedPatchDataset(cached) if cached is not None else _raw_pannuke(split)

    test_ds = PreprocessedDataset(raw, cfg["preprocessing"], is_train=False, feature_fn=feature_fn)
    kw = _loader_kwargs(cfg, device)
    kw.pop("persistent_workers", None)
    return DataLoader(test_ds, shuffle=False, **kw), patches_per_image


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model) -> int:
    """Params as reported in the paper's tables (all parameters, not just trainable)."""
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "BinaryMaskFeature", "build_loaders", "build_loss", "build_model",
    "build_optimizer", "build_scheduler", "build_test_loader", "cache_root",
    "count_params", "make_feature_fn", "monuseg_cache_path", "pannuke_cache_path",
    "resolve_device", "split_fingerprint", "total_params", "DEFAULT_SPLIT_SEED",
]
