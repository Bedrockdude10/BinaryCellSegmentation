#!/usr/bin/env python3
"""Figure 1, row 1, rendered for legibility — Reviewer K9Em's complaint.

    python scripts/render_figure1_highcontrast.py <stage_dir> <out_dir>

The published figure used ``make_overlay``, which blends the mask colour at
alpha=0.35 over the H&E image, so tissue texture shows through and the masks are
hard to read. That is what the reviewer objected to. Two replacements, both
dropping the background entirely:

  ``..._solid.png``     masks as solid fills on white. Same semantics as the
                        published figure (ground truth, then each prediction),
                        just legible. The literal answer to the review.
  ``..._errormap.png``  per-pixel agreement instead: green true positive,
                        red false positive, blue false negative, white correct
                        background. Carries strictly more information and is the
                        variant that actually shows the boundary behaviour the
                        caption claims, at the cost of changing what the columns
                        mean.

Both keep the input H&E column — the reviewer wanted the mask panels readable,
not the tissue removed — and both keep the per-image Dice annotation.

Image selection matches the published figure: RandomState(42).choice(14, size=3)
picks the same three test slides; row 1 is the first (TCGA-2Z-A9J9-01A-01-TS1).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("viz", REPO / "scripts/visualize.py")
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Deliberately saturated and far apart in both hue and luminance, so the panels
# survive greyscale printing and projection.
NUCLEUS = (11, 61, 145)      # deep blue on white
TP = (0, 140, 60)            # green
FP = (200, 30, 30)           # red
FN = (30, 90, 200)           # blue


def solid_mask(mask: np.ndarray, color=NUCLEUS) -> np.ndarray:
    """Binary mask as a solid fill on white. No image underneath."""
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[mask > 0.5] = color
    return out


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP / FP / FN on white. True negatives stay white so the eye ignores them."""
    g, p = gt > 0.5, pred > 0.5
    out = np.full((*gt.shape, 3), 255, dtype=np.uint8)
    out[g & p] = TP
    out[~g & p] = FP
    out[g & ~p] = FN
    return out


def render(stage: pathlib.Path, out_dir: pathlib.Path, mode: str) -> pathlib.Path:
    device = viz.resolve_device("auto")
    models = []
    for d in ["unet_vgg_monuseg", "unet_swin_pretrained_monuseg"]:
        m, disp, raw = viz.load_model(stage / d, device)
        if m is None:
            raise SystemExit(f"could not load {d} from {stage}")
        models.append((viz.MONUSEG_DISPLAY.get(raw, disp), m))

    tifs = sorted((REPO / "data/MoNuSegTestData").glob("*.tif"))
    chosen = sorted(np.random.RandomState(42).choice(len(tifs), size=3, replace=False))
    img_path = tifs[chosen[0]]

    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    gt = viz._load_gt_mask(img_path, h, w)

    n_cols = 2 + len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.35))
    titles = ["Input", "Ground Truth"] + [n for n, _ in models]

    axes[0].imshow(image)
    axes[1].imshow(solid_mask(gt))
    for j, (name, model) in enumerate(models):
        pred = viz._predict_monuseg(image, model, device, MEAN, STD)
        d, _ = viz.compute_sample_metrics(pred.astype(np.float32), gt.astype(np.float32))
        panel = solid_mask(pred) if mode == "solid" else error_map(gt, pred)
        axes[2 + j].imshow(panel)
        axes[2 + j].text(0.04, 0.05, f"Dice={d:.3f}", fontsize=9, color="black",
                         transform=axes[2 + j].transAxes, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                   ec="black", alpha=0.9))
        print(f"  {name}: Dice={d:.3f}")
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.axis("off")

    if mode == "errormap":
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(fc=np.array(TP) / 255, ec="black", label="True positive"),
                            Patch(fc=np.array(FP) / 255, ec="black", label="False positive"),
                            Patch(fc=np.array(FN) / 255, ec="black", label="False negative")],
                   loc="lower center", ncol=3, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(bottom=0.10)

    fig.tight_layout(pad=0.4, rect=(0, 0.06 if mode == "errormap" else 0, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"monuseg_comparison_row1_seed42_{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")
    return out


if __name__ == "__main__":
    stage, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for mode in ("solid", "errormap"):
        print(f"{mode}:")
        render(stage, out_dir, mode)
