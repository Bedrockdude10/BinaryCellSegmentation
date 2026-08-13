# src/data/cache.py
"""Flat memmap patch caches for PanNuke and MoNuSeg.

Both datasets are materialised once into two contiguous ``.npy`` files per
split:

    images.npy  (N, 256, 256, 3) uint8
    masks.npy   (N, 256, 256, 1) uint8   # 0/1 binary nucleus mask
    meta.json

This is purely a storage change and is **numerics-neutral**:

* PanNuke — the cached mask is ``binary_mask()`` applied to exactly the
  5-channel mask the on-the-fly loader builds, so the value handed to the model
  is bit-identical. It just stops being recomputed every epoch.
  ``scripts/prepare_cache.py --verify`` asserts this elementwise.
* MoNuSeg — a byte copy of the per-patch ``.npy`` files that
  ``src/data/monuseg.py`` already wrote, concatenated in the same sorted order.

The point of the flat layout is IO shape, not arithmetic: one memmap-able file
instead of 1,184 small reads per epoch, which matters on shared network home
and lets a whole split be staged to node-local NVMe with two copies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

CACHE_VERSION = 1
_META = "meta.json"


class CachedPatchDataset(Dataset):
    """Reads a flat memmap cache. Returns the same dict shape as the raw loaders.

    ``{"image": (H,W,3) uint8, "mask": (H,W,1) float32}`` — already binary, so
    the caller passes ``feature_fn=None``.
    """

    def __init__(self, cache_dir: str | Path, indices: list[int] | None = None):
        self.cache_dir = Path(cache_dir)
        self.meta = read_meta(self.cache_dir)
        # mmap_mode="r" keeps this cheap to fork into DataLoader workers: the
        # pages are shared through the OS page cache rather than copied.
        self._images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
        self._masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")
        if len(self._images) != len(self._masks):
            raise RuntimeError(f"{self.cache_dir}: image/mask count mismatch")
        self.indices = list(range(len(self._images))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        j = self.indices[idx]
        return {
            "image": np.asarray(self._images[j]),
            "mask": np.asarray(self._masks[j]).astype(np.float32),
        }


def read_meta(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / _META
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the cache with `python scripts/prepare_cache.py`."
        )
    meta = json.loads(path.read_text())
    if meta.get("cache_version") != CACHE_VERSION:
        raise RuntimeError(
            f"{path}: cache_version {meta.get('cache_version')} != {CACHE_VERSION}; rebuild."
        )
    return meta


def cache_is_complete(cache_dir: str | Path) -> bool:
    d = Path(cache_dir)
    if not all((d / f).exists() for f in ("images.npy", "masks.npy", _META)):
        return False
    try:
        meta = read_meta(d)
    except (RuntimeError, FileNotFoundError):
        return False
    return meta.get("complete", False)


def _writer(path: Path, n: int, channels: int, patch_size: int):
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(n, patch_size, patch_size, channels)
    )


def write_cache(cache_dir: str | Path, n: int, patch_size: int, meta_extra: dict,
                fill) -> dict:
    """Allocate the two memmaps, hand them to ``fill``, then commit meta.json.

    ``meta.json`` is only written with ``complete: true`` after ``fill``
    returns, so an interrupted build is detected rather than silently used.
    """
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    images = _writer(d / "images.npy", n, 3, patch_size)
    masks = _writer(d / "masks.npy", n, 1, patch_size)
    try:
        fill(images, masks)
        images.flush()
        masks.flush()
    finally:
        del images, masks
    meta = {"cache_version": CACHE_VERSION, "n": int(n), "patch_size": int(patch_size),
            "complete": True, **meta_extra}
    (d / _META).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta
