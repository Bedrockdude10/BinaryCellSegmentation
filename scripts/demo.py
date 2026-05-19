#!/usr/bin/env python3
# scripts/demo.py
"""
Run inference with a trained VGG U-Net checkpoint on one or more images and
save side-by-side visualizations (input | ground truth | prediction).

Pretrained checkpoints (~259 MB each):
  MoNuSeg model : <INSERT GOOGLE DRIVE LINK>
  PanNuke model : <INSERT GOOGLE DRIVE LINK>

Usage:
    # Two MoNuSeg test images (ground-truth XML auto-detected if present):
    python scripts/demo.py \\
        --checkpoint outputs/unet_vgg_monuseg/best.pt \\
        --images data/MoNuSegTestData/TCGA-2Z-A9J9-01A-01-TS1.tif \\
                 data/MoNuSegTestData/TCGA-44-2665-01B-06-BS6.tif \\
        --out demo_output

    # Any JPEG/PNG image (no ground truth):
    python scripts/demo.py \\
        --checkpoint outputs/unet_vgg_monuseg/best.pt \\
        --images my_slide.png \\
        --out demo_output
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ensure project root is on sys.path when running as `python scripts/demo.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.encoders import VGGEncoder
from src.models.unet import UNet

# ImageNet statistics used during training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PATCH_SIZE = 256


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    encoder = VGGEncoder(in_channels=3)
    model = UNet(encoder, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gt_mask(image_path: Path, h: int, w: int):
    xml_path = image_path.with_suffix(".xml")
    if not xml_path.exists():
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in ET.parse(xml_path).iter("Region"):
        verts = region.findall("Vertices/Vertex")
        if len(verts) < 3:
            continue
        pts = np.array([[float(v.attrib["X"]), float(v.attrib["Y"])] for v in verts],
                       dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def pad_to_multiple(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    ph = (size - h % size) % size
    pw = (size - w % size) % size
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return image, h, w


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    x = patch.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


@torch.no_grad()
def predict_full_image(image: np.ndarray, model: torch.nn.Module,
                       device: str, patch_size: int = PATCH_SIZE) -> np.ndarray:
    padded, orig_h, orig_w = pad_to_multiple(image, patch_size)
    ph, pw = padded.shape[:2]
    pred_map = np.zeros((ph, pw), dtype=np.float32)

    for y in range(0, ph, patch_size):
        for x in range(0, pw, patch_size):
            patch = padded[y:y + patch_size, x:x + patch_size]
            tensor = preprocess_patch(patch).to(device)
            logit = model(tensor)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_map[y:y + patch_size, x:x + patch_size] = prob

    return (pred_map[:orig_h, :orig_w] > 0.5).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    out = image.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1 - alpha) + v * alpha,
                                out[:, :, c])
    return out.clip(0, 255).astype(np.uint8)


def save_figure(image: np.ndarray, gt_mask, pred_mask: np.ndarray,
                out_path: Path, title: str):
    has_gt = gt_mask is not None
    ncols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    col = 1
    if has_gt:
        axes[col].imshow(overlay(image, gt_mask, (0, 200, 80)))
        axes[col].set_title("Ground Truth")
        axes[col].axis("off")
        col += 1

    axes[col].imshow(overlay(image, pred_mask, (255, 80, 80)))
    axes[col].set_title("Prediction")
    axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── dice ──────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * inter / denom) if denom > 0 else 1.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demo inference with trained VGG U-Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--images", nargs="+", required=True,
                        help="One or more image paths (.tif, .png, .jpg, …)")
    parser.add_argument("--out", default="demo_output",
                        help="Output directory for visualizations (default: demo_output)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    for img_path_str in args.images:
        img_path = Path(img_path_str)
        print(f"\nProcessing: {img_path.name}")

        image = load_image(img_path)
        h, w = image.shape[:2]
        gt_mask = load_gt_mask(img_path, h, w)
        pred_mask = predict_full_image(image, model, device)

        if gt_mask is not None:
            d = dice_score(pred_mask, gt_mask)
            print(f"  Dice: {d:.4f}")

        out_path = out_dir / (img_path.stem + "_demo.png")
        save_figure(image, gt_mask, pred_mask, out_path, title=img_path.stem)

    print(f"\nDone. Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
