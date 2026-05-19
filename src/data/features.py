# src/data/features.py
"""Derives features from raw data. Stateless, numpy only."""
import numpy as np


def binary_mask(mask: np.ndarray) -> np.ndarray:
    """(H, W, 5) multi-class → (H, W, 1) binary."""
    return (mask.max(axis=-1, keepdims=True) > 0).astype(np.float32)


def extract_patches(image, mask, patch_size=32, stride=32):
    """Grid-aligned patches from image and mask."""
    H, W = image.shape[:2]
    patches = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patches.append({
                "image": image[r:r + patch_size, c:c + patch_size],
                "mask": mask[r:r + patch_size, c:c + patch_size],
            })
    return patches


def patch_has_cells(mask_patch: np.ndarray, threshold: float = 0.01) -> float:
    """1.0 if enough cell pixels in patch, else 0.0."""
    return 1.0 if mask_patch.max(axis=-1).mean() > threshold else 0.0