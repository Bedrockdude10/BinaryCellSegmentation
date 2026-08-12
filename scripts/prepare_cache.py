#!/usr/bin/env python3
# scripts/prepare_cache.py
"""Materialise PanNuke and MoNuSeg into flat memmap caches.

    python scripts/prepare_cache.py                    # build both, then verify
    python scripts/prepare_cache.py --datasets pannuke
    python scripts/prepare_cache.py --verify-only

Why: the sweep runs 35 trainings over the same patches. Decoding PanNuke out of
the HuggingFace arrow store and re-deriving the binary mask from per-instance
masks on every epoch of every run is wasted work, and MoNuSeg's per-patch layout
means 1,184 small file opens per epoch — painful on shared network home. Two
contiguous uint8 arrays per split instead: stage-able to node-local NVMe with
two copies, memmap-able, and shareable across DataLoader workers.

This is a storage change only. ``--verify`` asserts elementwise equality between
the cache and the original pipeline, which is what makes "numerics-neutral" a
checked claim rather than an assertion.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.builders import monuseg_cache_path, pannuke_cache_path  # noqa: E402
from src.data.cache import CachedPatchDataset, cache_is_complete, write_cache  # noqa: E402
from src.data.features import binary_mask  # noqa: E402
from src.splits import monuseg_source_stems  # noqa: E402

PANNUKE_FOLDS = ("fold1", "fold2", "fold3")
MONUSEG_SPLITS = ("train", "test")
PATCH = 256


# ── PanNuke ──────────────────────────────────────────────────────────────────

def build_pannuke(fold: str, out_root: Path, force: bool = False) -> Path:
    from src.data.dataset import PanNukeDataset

    dest = pannuke_cache_path(out_root, fold)
    if cache_is_complete(dest) and not force:
        print(f"  {fold}: already cached at {dest}")
        return dest

    raw = PanNukeDataset(fold)
    n = len(raw)
    print(f"  {fold}: {n} patches -> {dest}")
    t0 = time.time()

    def fill(images, masks):
        for i in range(n):
            sample = raw[i]
            images[i] = sample["image"]
            # Exactly the transform the training pipeline applied on the fly:
            # 5-channel instance-category mask collapsed to binary.
            masks[i] = binary_mask(sample["mask"]).astype(np.uint8)
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    write_cache(dest, n, PATCH, {"source": f"RationAI/PanNuke:{fold}", "fold": fold,
                                 "dataset": "pannuke"}, fill)
    print(f"  {fold}: done in {time.time() - t0:.0f}s")
    return dest


def verify_pannuke(fold: str, out_root: Path, n_check: int, rng: np.random.Generator) -> None:
    from src.data.dataset import PanNukeDataset

    raw = PanNukeDataset(fold)
    cached = CachedPatchDataset(pannuke_cache_path(out_root, fold))
    if len(raw) != len(cached):
        raise AssertionError(f"{fold}: {len(raw)} raw vs {len(cached)} cached")
    idx = rng.choice(len(raw), size=min(n_check, len(raw)), replace=False)
    for i in idx:
        want = raw[int(i)]
        got = cached[int(i)]
        if not np.array_equal(want["image"], got["image"]):
            raise AssertionError(f"{fold}[{i}]: image mismatch")
        want_mask = binary_mask(want["mask"])
        if not np.array_equal(want_mask, got["mask"]):
            raise AssertionError(f"{fold}[{i}]: mask mismatch")
        if want_mask.dtype != got["mask"].dtype:
            raise AssertionError(f"{fold}[{i}]: mask dtype {want_mask.dtype} vs {got['mask'].dtype}")
    print(f"  {fold}: {len(idx)} samples bit-identical to the on-the-fly pipeline")


# ── MoNuSeg ──────────────────────────────────────────────────────────────────

def build_monuseg(split: str, data_root: Path, out_root: Path, force: bool = False) -> Path:
    from src.data.monuseg import MoNuSegDataset

    dest = monuseg_cache_path(out_root, split, PATCH)
    if cache_is_complete(dest) and not force:
        print(f"  {split}: already cached at {dest}")
        return dest

    # Builds the per-patch cache first if it is not there yet.
    raw = MoNuSegDataset(split=split, data_root=str(data_root))
    n = len(raw)
    counts = raw.patch_counts
    stems = monuseg_source_stems(data_root, split)
    if sum(counts) != n:
        raise AssertionError(f"{split}: patch counts sum to {sum(counts)} but cache holds {n}")
    if len(stems) != len(counts):
        raise AssertionError(f"{split}: {len(stems)} source images vs {len(counts)} count entries")
    print(f"  {split}: {n} patches from {len(stems)} images -> {dest}")

    def fill(images, masks):
        for i in range(n):
            sample = raw[i]
            images[i] = sample["image"]
            masks[i] = sample["mask"].astype(np.uint8)

    write_cache(dest, n, PATCH, {
        "source": f"MoNuSeg:{split}", "dataset": "monuseg", "split": split,
        "patch_counts": [int(c) for c in counts], "source_images": stems,
        "patch_files": [p.name for p in raw.img_paths],
    }, fill)
    print(f"  {split}: done")
    return dest


def verify_monuseg(split: str, data_root: Path, out_root: Path, n_check: int,
                   rng: np.random.Generator) -> None:
    from src.data.monuseg import MoNuSegDataset

    raw = MoNuSegDataset(split=split, data_root=str(data_root))
    cached = CachedPatchDataset(monuseg_cache_path(out_root, split, PATCH))
    if len(raw) != len(cached):
        raise AssertionError(f"{split}: {len(raw)} raw vs {len(cached)} cached")
    idx = rng.choice(len(raw), size=min(n_check, len(raw)), replace=False)
    for i in idx:
        want, got = raw[int(i)], cached[int(i)]
        if not np.array_equal(want["image"], got["image"]):
            raise AssertionError(f"{split}[{i}]: image mismatch")
        if not np.array_equal(want["mask"], got["mask"]):
            raise AssertionError(f"{split}[{i}]: mask mismatch")
    print(f"  {split}: {len(idx)} samples bit-identical to the per-patch cache")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pannuke", "monuseg"],
                    choices=["pannuke", "monuseg"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data", help="Where the flat caches go")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", type=int, default=64,
                    help="Spot-check this many samples per split against the original pipeline")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    data_root, out_root = Path(args.data_root), Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    if not args.verify_only:
        if "pannuke" in args.datasets:
            print("PanNuke:")
            for fold in PANNUKE_FOLDS:
                build_pannuke(fold, out_root, args.force)
        if "monuseg" in args.datasets:
            print("MoNuSeg:")
            for split in MONUSEG_SPLITS:
                build_monuseg(split, data_root, out_root, args.force)

    if args.verify:
        print(f"Verifying ({args.verify} samples per split):")
        if "pannuke" in args.datasets:
            for fold in PANNUKE_FOLDS:
                verify_pannuke(fold, out_root, args.verify, rng)
        if "monuseg" in args.datasets:
            for split in MONUSEG_SPLITS:
                verify_monuseg(split, data_root, out_root, args.verify, rng)

    total = sum(p.stat().st_size for p in out_root.rglob("*flat*/*.npy"))
    print(f"\nFlat caches: {total / 1e9:.2f} GB under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
