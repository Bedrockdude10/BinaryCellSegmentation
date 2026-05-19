#!/usr/bin/env python3
# scripts/evaluate.py
"""
Usage:
    python scripts/evaluate.py --experiment outputs/baseline_mlp
    python scripts/evaluate.py --experiment outputs/baseline_mlp --split fold3
"""
import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import PanNukeDataset
from src.data.monuseg import MoNuSegDataset
from src.data.features import binary_mask
from src.data.preprocessing import PreprocessedDataset
from src.evaluation.metrics import dice, iou
from src.evaluation.evaluator import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Path to experiment output dir")
    parser.add_argument("--split", default=None, help="Override test split (default: from config)")
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    cfg = load_config(str(exp_dir / "config.yaml"))

    # Import the right model builder from train script
    # (keeps evaluate.py from duplicating build logic)
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_script", "scripts/train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    device = train_mod.resolve_device(cfg["experiment"].get("device", "auto"))
    model = train_mod.build_model(cfg).to(device)

    ckpt_path = exp_dir / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    log.info(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    split = args.split or cfg["data"]["test_split"]
    feature_fn = train_mod.make_feature_fn(cfg)
    dataset_name = cfg["data"].get("dataset", "pannuke")

    patches_per_image = None
    if dataset_name == "monuseg":
        raw = MoNuSegDataset(split=split, data_root=cfg["data"]["data_root"])
        # Load per-image patch counts for MedT-compatible per-image averaged Dice
        counts_path = raw._cache_dir / "image_patch_counts.npy"
        patches_per_image = np.load(counts_path).tolist()
    else:
        raw = PanNukeDataset(split)

    test_ds = PreprocessedDataset(raw, cfg["preprocessing"], is_train=False, feature_fn=feature_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], num_workers=0)

    results = evaluate(model, test_loader, {"dice": dice, "iou": iou},
                       device=device, patches_per_image=patches_per_image)
    log.info(f"Test ({split}): {results}")

    out_path = exp_dir / f"eval_{split}.json"
    json.dump({"split": split, "checkpoint": args.checkpoint, **results}, open(out_path, "w"), indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()