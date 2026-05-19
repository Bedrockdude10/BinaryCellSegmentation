# src/data/monuseg.py
"""MoNuSeg dataset: parses XML polygon annotations into binary masks and
tiles 1000x1000 images into 256x256 patches on first use, caching to disk."""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


# ── XML → binary mask ────────────────────────────────────────────────────────

def _xml_to_mask(xml_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    tree = ET.parse(xml_path)
    for region in tree.iter("Region"):
        vertices = region.findall("Vertices/Vertex")
        if len(vertices) < 3:
            continue
        pts = np.array(
            [[float(v.attrib["X"]), float(v.attrib["Y"])] for v in vertices],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 1)
    return mask


# ── Patch extraction ─────────────────────────────────────────────────────────

def _extract_patches(image: np.ndarray, mask: np.ndarray, patch_size: int = 256):
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            img_patch = image[y : y + patch_size, x : x + patch_size]
            msk_patch = mask[y : y + patch_size, x : x + patch_size]
            patches.append((img_patch, msk_patch))
    return patches


def _pad_to_multiple(image: np.ndarray, mask: np.ndarray, patch_size: int):
    """Pad image and mask up to the next multiple of patch_size.
    Image is reflection-padded; mask is zero-padded (padded pixels = background).
    """
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image, mask


def _build_cache(image_dir: Path, ann_dir: Path, cache_dir: Path,
                 patch_size: int = 256, pad: bool = False):
    """Cache patches grouped by image.

    pad: if True, pad each image to the next patch_size multiple before
         tiling so no pixels are dropped. Used for test split.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(image_dir.glob("*.tif"))
    image_patch_counts = []
    idx = 0
    for tif_path in tif_files:
        xml_path = ann_dir / (tif_path.stem + ".xml")
        if not xml_path.exists():
            continue
        image = cv2.imread(str(tif_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask = _xml_to_mask(xml_path, h, w)
        if pad:
            image, mask = _pad_to_multiple(image, mask, patch_size)
        patches = _extract_patches(image, mask, patch_size)
        for img_patch, msk_patch in patches:
            np.save(cache_dir / f"{idx:05d}_img.npy", img_patch)
            np.save(cache_dir / f"{idx:05d}_msk.npy", msk_patch)
            idx += 1
        image_patch_counts.append(len(patches))
    np.save(cache_dir / "image_patch_counts.npy", np.array(image_patch_counts))
    return idx


# ── Dataset ──────────────────────────────────────────────────────────────────

class MoNuSegDataset(Dataset):
    """Returns {"image": (256,256,3) uint8, "mask": (256,256,1) float32}.

    patch_indices: if provided, only expose this subset of patch indices.
    Used for image-level train/val splitting to prevent leakage.
    """

    def __init__(self, split: str, data_root: str, patch_size: int = 256,
                 patch_indices: list = None):
        root = Path(data_root)
        if split == "train":
            image_dir = root / "MoNuSeg 2018 Training Data" / "Tissue Images"
            ann_dir = root / "MoNuSeg 2018 Training Data" / "Annotations"
        elif split == "test":
            image_dir = root / "MoNuSegTestData"
            ann_dir = root / "MoNuSegTestData"
        else:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        cache_dir = root / f"monuseg_cache_{split}_{patch_size}"
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            _build_cache(image_dir, ann_dir, cache_dir, patch_size, pad=True)

        all_img_paths = sorted(cache_dir.glob("*_img.npy"))
        all_msk_paths = sorted(cache_dir.glob("*_msk.npy"))
        assert len(all_img_paths) == len(all_msk_paths), "Cache mismatch"

        if patch_indices is not None:
            self.img_paths = [all_img_paths[i] for i in patch_indices]
            self.msk_paths = [all_msk_paths[i] for i in patch_indices]
        else:
            self.img_paths = all_img_paths
            self.msk_paths = all_msk_paths

        self._cache_dir = cache_dir

    def image_level_split(self, val_fraction: float = 0.2, seed: int = 42):
        """Split patches by image to avoid leakage. Returns (train_ds, val_ds)."""
        counts_path = self._cache_dir / "image_patch_counts.npy"
        counts = np.load(counts_path).tolist()
        n_images = len(counts)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_images).tolist()
        n_val = max(1, int(n_images * val_fraction))
        val_image_indices = set(indices[:n_val])
        train_image_indices = set(indices[n_val:])

        patch_start = 0
        train_patches, val_patches = [], []
        for img_idx, count in enumerate(counts):
            patch_range = list(range(patch_start, patch_start + count))
            if img_idx in val_image_indices:
                val_patches.extend(patch_range)
            else:
                train_patches.extend(patch_range)
            patch_start += count

        root = self._cache_dir.parent
        split = "train"
        train_ds = MoNuSegDataset(split, str(root), patch_indices=train_patches)
        val_ds = MoNuSegDataset(split, str(root), patch_indices=val_patches)
        return train_ds, val_ds

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.load(self.img_paths[idx])                       # (256,256,3) uint8
        mask = np.load(self.msk_paths[idx]).astype(np.float32)     # (256,256)
        mask = mask[:, :, np.newaxis]                              # (256,256,1)
        return {"image": image, "mask": mask}
