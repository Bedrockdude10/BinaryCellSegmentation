#!/usr/bin/env python3
# scripts/visualize.py
"""
Generate qualitative comparison figures for the final report.

PanNuke mode (default):
  1. Per-sample detailed grids (overlay + error map rows)
  2. Compact summary grid (overlays only, one row per sample)
  3. Error map summary (TP/FP/FN maps, one row per sample)
  4. Otsu vs best-model comparison grid

MoNuSeg mode:
  5. monuseg_comparison.png — N rows x 4 cols (Input | GT | VGG | Swin)
     with per-cell Dice scores, using whole test images tiled into patches.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --num_samples 8 --output_dir figures
    python scripts/visualize.py --indices 0 42 100 200
    python scripts/visualize.py --dataset monuseg --data_root data --num_samples 3 --output_dir figures
"""
import argparse
import logging
import sys
import xml.etree.ElementTree as ET
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
from matplotlib.patches import Patch
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import PanNukeDataset
from src.data.features import binary_mask
from src.evaluation.metrics import dice as dice_fn, iou as iou_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Friendly display names for experiment directories ────────────────────────

DISPLAY_NAMES = {
    "unet_vgg": "VGG+BCE",
    "unet_resnet": "ResNet+BCE",
    "unet_vgg_dice": "VGG+Dice",
    "unet_swin": "Swin (scratch)",
    "unet_swin_pretrained": "Swin (pretrained)",
}


# ── Model loading (mirrors scripts/train.py logic) ──────────────────────────

def resolve_device(s):
    if s == "auto":
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return s


def load_model(experiment_dir, device):
    """Load a trained model from an experiment directory."""
    exp_dir = Path(experiment_dir)
    config_path = exp_dir / "config.yaml"
    ckpt_path = exp_dir / "best.pt"

    if not config_path.exists() or not ckpt_path.exists():
        log.warning(f"Skipping {exp_dir} — missing config or checkpoint")
        return None, None, None

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from src.models.unet import UNet
    from src.models.encoders import VGGEncoder, ResNetEncoder

    arch = cfg["model"]["architecture"]
    params = cfg["model"].get("params", {})
    in_ch = params.get("in_channels", 3)
    out_ch = params.get("out_channels", 1)

    if arch == "vgg":
        encoder = VGGEncoder(in_channels=in_ch)
    elif arch == "resnet":
        encoder = ResNetEncoder(in_channels=in_ch)
    elif arch in ("swin", "swin_pretrained"):
        from src.models.encoders import SwinEncoder
        encoder = SwinEncoder(in_channels=in_ch, pretrained=False)
    else:
        log.warning(f"Unknown architecture: {arch}")
        return None, None, None

    model = UNet(encoder, out_channels=out_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    raw_name = cfg["experiment"]["name"]
    display_name = DISPLAY_NAMES.get(raw_name, raw_name)
    log.info(f"Loaded {display_name} from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model, display_name, raw_name


def predict(model, image_tensor, device):
    """Run inference and return binary mask as numpy (H, W)."""
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
    return pred.squeeze().cpu().numpy()


# ── Classical baseline ───────────────────────────────────────────────────────

def otsu_segment(image_np):
    """Apply Otsu thresholding to an RGB image. Returns (H, W) binary mask."""
    gray = rgb2gray(image_np)
    try:
        thresh = threshold_otsu(gray)
    except ValueError:
        return np.zeros(gray.shape, dtype=np.float32)
    return (gray < thresh).astype(np.float32)


# ── Preprocessing (matches training pipeline) ────────────────────────────────

def preprocess_image(image_np, mean, std):
    """Normalize and convert to tensor, matching training preprocessing."""
    img = image_np.astype(np.float32) / 255.0
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).float()


# ── Metrics helper ───────────────────────────────────────────────────────────

def compute_sample_metrics(pred_np, gt_np):
    """Compute Dice and IoU for a single sample. Returns (dice, iou) floats."""
    pred_t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float()
    gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
    return dice_fn(pred_t, gt_t), iou_fn(pred_t, gt_t)


# ── Visualization helpers ────────────────────────────────────────────────────

def make_overlay(image_np, mask, color=(0, 255, 0), alpha=0.35):
    """Overlay a binary mask on an RGB image."""
    overlay = image_np.copy().astype(np.float32)
    mask_bool = mask > 0.5
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_error_map(gt_mask, pred_mask):
    """Create an RGB error map: green=TP, red=FP, blue=FN, black=TN."""
    gt = gt_mask > 0.5
    pred = pred_mask > 0.5
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    error[gt & pred] = [0, 200, 0]       # true positive — green
    error[~gt & pred] = [220, 40, 40]    # false positive — red
    error[gt & ~pred] = [40, 80, 220]    # false negative — blue
    return error


ERROR_LEGEND = [
    Patch(facecolor=(0, 0.78, 0), label="True Positive"),
    Patch(facecolor=(0.86, 0.16, 0.16), label="False Positive"),
    Patch(facecolor=(0.16, 0.31, 0.86), label="False Negative"),
]


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_comparison_grid(image_np, gt_mask, predictions, sample_idx, output_dir):
    """
    Per-sample detailed comparison.
    Row 1: Original | GT overlay | Model overlays ...
    Row 2: (blank)  | GT mask    | Error maps ...
    Dice score annotated below each model column.
    """
    n_models = len(predictions)
    n_cols = 2 + n_models

    fig = plt.figure(figsize=(3.2 * n_cols, 6.8))
    gs = gridspec.GridSpec(2, n_cols, hspace=0.12, wspace=0.05)

    gt_binary = gt_mask.squeeze()

    # Row 1: overlays
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_np)
    ax.set_title("Input", fontsize=10, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(make_overlay(image_np, gt_binary, color=(0, 255, 0)))
    ax.set_title("Ground Truth", fontsize=10, fontweight="bold")
    ax.axis("off")

    for i, (name, pred, d, iou) in enumerate(predictions):
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(make_overlay(image_np, pred, color=(0, 200, 255)))
        ax.set_title(f"{name}\nDice={d:.3f}", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Row 2: masks and error maps
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
    ax.set_title("GT Mask", fontsize=9)
    ax.axis("off")

    for i, (name, pred, d, iou) in enumerate(predictions):
        ax = fig.add_subplot(gs[1, 2 + i])
        ax.imshow(make_error_map(gt_binary, pred))
        ax.set_title("Error Map", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Sample {sample_idx}", fontsize=12, fontweight="bold", y=0.99)

    # Error map legend
    fig.legend(handles=ERROR_LEGEND, loc="lower center", ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))

    out_path = output_dir / f"comparison_sample_{sample_idx}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def plot_summary_grid(image_list, gt_list, all_predictions, model_names, output_dir):
    """
    Compact overlay summary: one row per sample, columns = Input | GT | models.
    Per-sample Dice annotated in each cell.
    """
    n_samples = len(image_list)
    n_cols = 2 + len(model_names)

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.8 * n_cols, 2.8 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth"] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    for i in range(n_samples):
        image_np = image_list[i]
        gt = gt_list[i].squeeze()

        axes[i, 0].imshow(image_np)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(make_overlay(image_np, gt, color=(0, 255, 0)))
        axes[i, 1].axis("off")

        for j, name in enumerate(model_names):
            pred = all_predictions[name][i]
            d, _ = compute_sample_metrics(pred, gt)
            axes[i, 2 + j].imshow(make_overlay(image_np, pred, color=(0, 200, 255)))
            axes[i, 2 + j].text(
                4, 250, f"D={d:.3f}", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )
            axes[i, 2 + j].axis("off")

    fig.tight_layout(pad=0.5)
    out_path = output_dir / "summary_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def plot_error_summary(image_list, gt_list, all_predictions, model_names, output_dir):
    """
    Error map summary: one row per sample. Green=TP, Red=FP, Blue=FN.
    """
    n_samples = len(image_list)
    n_cols = 2 + len(model_names)

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.8 * n_cols, 2.8 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "GT Mask"] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    for i in range(n_samples):
        image_np = image_list[i]
        gt = gt_list[i].squeeze()

        axes[i, 0].imshow(image_np)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")

        for j, name in enumerate(model_names):
            pred = all_predictions[name][i]
            d, _ = compute_sample_metrics(pred, gt)
            axes[i, 2 + j].imshow(make_error_map(gt, pred))
            axes[i, 2 + j].text(
                4, 250, f"D={d:.3f}", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )
            axes[i, 2 + j].axis("off")

    fig.legend(handles=ERROR_LEGEND, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=0.5)
    out_path = output_dir / "error_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def plot_otsu_comparison(image_list, gt_list, all_predictions, best_model_name, output_dir):
    """
    Side-by-side: Input | GT | Otsu | Best learned model.
    Shows the classical-to-learned gap clearly.
    """
    n_samples = len(image_list)
    n_cols = 4

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(11.2, 2.8 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Otsu Threshold", best_model_name]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    for i in range(n_samples):
        image_np = image_list[i]
        gt = gt_list[i].squeeze()

        # Otsu prediction
        otsu_pred = otsu_segment(image_np)
        d_otsu, _ = compute_sample_metrics(otsu_pred, gt)

        # Best model prediction
        best_pred = all_predictions[best_model_name][i]
        d_best, _ = compute_sample_metrics(best_pred, gt)

        axes[i, 0].imshow(image_np)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(make_overlay(image_np, gt, color=(0, 255, 0)))
        axes[i, 1].axis("off")

        axes[i, 2].imshow(make_overlay(image_np, otsu_pred, color=(255, 140, 0)))
        axes[i, 2].text(
            4, 250, f"D={d_otsu:.3f}", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
        )
        axes[i, 2].axis("off")

        axes[i, 3].imshow(make_overlay(image_np, best_pred, color=(0, 200, 255)))
        axes[i, 3].text(
            4, 250, f"D={d_best:.3f}", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
        )
        axes[i, 3].axis("off")

    fig.tight_layout(pad=0.5)
    out_path = output_dir / "otsu_vs_learned.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── Training curves ──────────────────────────────────────────────────────────

def parse_train_log(log_path):
    """Parse a train.log and return (epochs, train_losses, val_dices) lists."""
    import re
    epochs, train_losses, val_dices = [], [], []
    epoch = 0
    with open(log_path) as f:
        for line in f:
            if re.search(r"Epoch \d+", line):
                m = re.search(r"Epoch (\d+)", line)
                if m:
                    epoch = int(m.group(1))
            m = re.search(r"Loss: ([\d.]+) \| Val: \{.*'dice': ([\d.]+)", line)
            if m:
                epochs.append(epoch)
                train_losses.append(float(m.group(1)))
                val_dices.append(float(m.group(2)))
    return epochs, train_losses, val_dices


def plot_training_curves(experiment_dirs, display_names, output_dir):
    """Plot val Dice and train loss curves for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, exp_dir in enumerate(experiment_dirs):
        log_path = Path(exp_dir) / "train.log"
        if not log_path.exists():
            continue
        name = display_names.get(Path(exp_dir).name, Path(exp_dir).name)
        epochs, train_losses, val_dices = parse_train_log(log_path)
        if not epochs:
            continue
        c = colors[i % len(colors)]
        ax1.plot(epochs, val_dices, label=name, color=c, linewidth=1.8)
        ax2.plot(epochs, train_losses, label=name, color=c, linewidth=1.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Dice")
    ax1.set_title("Validation Dice over Training")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Loss")
    ax2.set_title("Training Loss over Training")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── MoNuSeg helpers ──────────────────────────────────────────────────────────

MONUSEG_DISPLAY = {
    "unet_vgg_monuseg": "VGG",
    "unet_swin_pretrained_monuseg": "Swin (pretrained)",
}

MONUSEG_EXPERIMENTS = [
    "outputs/unet_vgg_monuseg",
    "outputs/unet_swin_pretrained_monuseg",
]


def _load_gt_mask(image_path: Path, h: int, w: int) -> np.ndarray:
    xml_path = image_path.with_suffix(".xml")
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in ET.parse(xml_path).iter("Region"):
        verts = region.findall("Vertices/Vertex")
        if len(verts) < 3:
            continue
        pts = np.array([[float(v.attrib["X"]), float(v.attrib["Y"])] for v in verts],
                       dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _pad_to_multiple(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    ph = (size - h % size) % size
    pw = (size - w % size) % size
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return image, h, w


@torch.no_grad()
def _predict_monuseg(image: np.ndarray, model, device: str,
                     mean: np.ndarray, std: np.ndarray,
                     patch_size: int = 256) -> np.ndarray:
    padded, orig_h, orig_w = _pad_to_multiple(image, patch_size)
    ph, pw = padded.shape[:2]
    pred_map = np.zeros((ph, pw), dtype=np.float32)
    for y in range(0, ph, patch_size):
        for x in range(0, pw, patch_size):
            patch = padded[y:y + patch_size, x:x + patch_size]
            tensor = preprocess_image(patch, mean, std).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
            pred_map[y:y + patch_size, x:x + patch_size] = prob
    return (pred_map[:orig_h, :orig_w] > 0.5).astype(np.uint8)


def plot_monuseg_comparison(image_paths, models_ordered, device, mean, std, output_dir):
    """N rows × (2 + n_models) cols: Input | GT | model predictions.

    image_paths: list of Path to .tif test images (XML must be alongside)
    models_ordered: list of (display_name, model) in desired column order
    """
    n_rows = len(image_paths)
    model_names = [name for name, _ in models_ordered]
    n_cols = 2 + len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth"] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")

    for i, img_path in enumerate(image_paths):
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        gt = _load_gt_mask(img_path, h, w)

        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(make_overlay(image, gt, color=(0, 220, 80)))
        axes[i, 1].axis("off")

        for j, (name, model) in enumerate(models_ordered):
            pred = _predict_monuseg(image, model, device, mean, std)
            d, _ = compute_sample_metrics(pred.astype(np.float32),
                                          gt.astype(np.float32))
            axes[i, 2 + j].imshow(make_overlay(image, pred, color=(0, 180, 255)))
            axes[i, 2 + j].text(
                0.04, 0.05, f"Dice={d:.3f}", fontsize=8, color="white",
                transform=axes[i, 2 + j].transAxes,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )
            axes[i, 2 + j].axis("off")

    fig.tight_layout(pad=0.4)
    out_path = output_dir / "monuseg_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pannuke", choices=["pannuke", "monuseg"])
    parser.add_argument("--data_root", default="data",
                        help="MoNuSeg data root (only used with --dataset monuseg)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Override experiment dirs (default depends on --dataset)")
    parser.add_argument("--num_samples", type=int, default=6,
                        help="Number of images/samples to visualize")
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Specific test set indices to visualize (PanNuke only)")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    log.info(f"Device: {device}")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── MoNuSeg mode ─────────────────────────────────────────────────────────
    if args.dataset == "monuseg":
        exp_dirs = args.experiments or MONUSEG_EXPERIMENTS
        display_map = MONUSEG_DISPLAY

        models_ordered = []
        for exp_dir in exp_dirs:
            model, display_name, raw_name = load_model(exp_dir, device)
            if model is not None:
                label = display_map.get(raw_name, display_name)
                models_ordered.append((label, model))

        if not models_ordered:
            log.error("No MoNuSeg models loaded!")
            return

        test_dir = Path(args.data_root) / "MoNuSegTestData"
        all_tifs = sorted(test_dir.glob("*.tif"))
        rng = np.random.RandomState(args.seed)
        chosen = rng.choice(len(all_tifs), size=min(args.num_samples, len(all_tifs)),
                            replace=False)
        image_paths = [all_tifs[i] for i in sorted(chosen)]
        log.info(f"MoNuSeg images: {[p.name for p in image_paths]}")

        plot_monuseg_comparison(image_paths, models_ordered, device, mean, std, output_dir)
        log.info(f"Done! Figure saved to {output_dir}/")
        return

    # ── PanNuke mode ──────────────────────────────────────────────────────────
    exp_dirs = args.experiments or [
        "outputs/unet_vgg",
        "outputs/unet_resnet",
        "outputs/unet_vgg_dice",
        "outputs/unet_swin",
        "outputs/unet_swin_pretrained",
    ]

    plot_training_curves(exp_dirs, DISPLAY_NAMES, output_dir)

    models = {}
    for exp_dir in exp_dirs:
        model, display_name, raw_name = load_model(exp_dir, device)
        if model is not None:
            models[display_name] = model

    if not models:
        log.error("No models loaded!")
        return

    log.info(f"Loaded {len(models)} models: {list(models.keys())}")

    log.info("Loading test set (fold3)...")
    dataset = PanNukeDataset("fold3")

    if args.indices is not None:
        indices = args.indices
    else:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(dataset), size=args.num_samples, replace=False)
        indices.sort()

    log.info(f"Visualizing indices: {indices}")

    image_list = []
    gt_list = []
    all_predictions = {name: [] for name in models}

    for idx in indices:
        sample = dataset[idx]
        image_np = sample["image"]
        gt = binary_mask(sample["mask"])
        image_tensor = preprocess_image(image_np, mean, std)
        image_list.append(image_np)
        gt_list.append(gt)
        for name, model in models.items():
            pred = predict(model, image_tensor, device)
            all_predictions[name].append(pred)

    model_names = list(models.keys())

    for i, idx in enumerate(indices):
        gt_sq = gt_list[i].squeeze()
        predictions = []
        for name in model_names:
            pred = all_predictions[name][i]
            d, iou_val = compute_sample_metrics(pred, gt_sq)
            predictions.append((name, pred, d, iou_val))
        plot_comparison_grid(image_list[i], gt_list[i], predictions, idx, output_dir)

    plot_summary_grid(image_list, gt_list, all_predictions, model_names, output_dir)
    plot_error_summary(image_list, gt_list, all_predictions, model_names, output_dir)

    best_name = model_names[0]
    plot_otsu_comparison(image_list, gt_list, all_predictions, best_name, output_dir)

    log.info(f"Done! All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()