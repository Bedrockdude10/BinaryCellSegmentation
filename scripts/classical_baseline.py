#!/usr/bin/env python3
# scripts/classical_baseline.py
"""
Classical baseline: Otsu thresholding for binary nucleus segmentation.
No training — just threshold each image and compute Dice/IoU on the test set.

Usage:
    PYTHONPATH=. python scripts/classical_baseline.py
    PYTHONPATH=. python scripts/classical_baseline.py --split fold3
"""
import argparse
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from src.data.dataset import PanNukeDataset
from src.data.features import binary_mask
from src.evaluation.metrics import dice, iou

import json
from pathlib import Path

def otsu_segment(image_np):
    """Apply Otsu thresholding to an RGB image. Returns (H, W) binary mask."""
    gray = rgb2gray(image_np)  # float64 in [0, 1]
    try:
        thresh = threshold_otsu(gray)
    except ValueError:
        # Uniform image — no threshold found
        return np.zeros(gray.shape, dtype=np.float32)
    return (gray < thresh).astype(np.float32)  # nuclei are darker than background


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="fold3")
    args = parser.parse_args()

    print(f"Loading {args.split}...")
    dataset = PanNukeDataset(args.split)

    dice_scores = []
    iou_scores = []

    for i in range(len(dataset)):
        sample = dataset[i]
        image_np = sample["image"]                          # (256, 256, 3) uint8
        gt = binary_mask(sample["mask"]).squeeze(-1)        # (256, 256) float32

        pred = otsu_segment(image_np)                       # (256, 256) float32

        # Convert to tensors for metric functions
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

        dice_scores.append(dice(pred_t, gt_t))
        iou_scores.append(iou(pred_t, gt_t))

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(dataset)}")

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    print(f"\nOtsu Baseline ({args.split}):")
    print(f"  Dice: {mean_dice:.4f}")
    print(f"  IoU:  {mean_iou:.4f}")
    print(f"  Samples: {len(dataset)}")
    
    out_path = f"outputs/classical_otsu/eval_{args.split}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"split": args.split, "method": "otsu", "dice": mean_dice, "iou": mean_iou},
              open(out_path, "w"), indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()