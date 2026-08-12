# src/splits.py
"""Train/val/test split derivation and fingerprinting.

Two seeds, deliberately separate:

* ``split_seed`` — FIXED at 42 for the entire study. It selects which MoNuSeg
  images are held out for validation. It is not varied because MoNuSeg has 37
  training images: resampling the split would conflate split variance with
  training-seed variance, and the multi-seed study is about the latter.
* ``train_seed`` — varies across the 5 repeats (see ``src.seeding``).

Every run recomputes the split fingerprint and asserts it against
``configs/split_manifest.json``. If the on-disk data or the split logic ever
drifts, the run dies on line one instead of quietly reporting numbers from a
different split than the rest of the table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST_PATH = Path("configs/split_manifest.json")


class SplitDriftError(RuntimeError):
    """Raised when the realised split does not match the recorded manifest."""


# ── MoNuSeg image-level split ────────────────────────────────────────────────

def monuseg_image_level_split(counts: list[int], val_fraction: float, split_seed: int):
    """Partition patch indices by source image so no slide spans both splits.

    Reproduces the original ``MoNuSegDataset.image_level_split`` exactly
    (``default_rng(seed).permutation``, ``n_val = max(1, int(n * frac))``, val
    taken from the head of the permutation) so the published split is preserved
    bit for bit.

    Returns ``(train_patch_indices, val_patch_indices, val_image_indices)``.
    """
    n_images = len(counts)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n_images).tolist()
    n_val = max(1, int(n_images * val_fraction))
    val_images = sorted(order[:n_val])
    val_set = set(val_images)

    train_patches: list[int] = []
    val_patches: list[int] = []
    start = 0
    for img_idx, count in enumerate(counts):
        rng_patches = range(start, start + count)
        (val_patches if img_idx in val_set else train_patches).extend(rng_patches)
        start += count
    return train_patches, val_patches, val_images


# ── Fingerprinting ───────────────────────────────────────────────────────────

def _hash(detail: dict) -> str:
    blob = json.dumps(detail, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def monuseg_source_stems(data_root: str | Path, split: str) -> list[str]:
    """Source image stems in the order ``_build_cache`` consumed them.

    Mirrors the cache builder: sorted ``*.tif`` glob, skipping any image whose
    XML annotation is missing.
    """
    root = Path(data_root)
    if split == "train":
        image_dir = root / "MoNuSeg 2018 Training Data" / "Tissue Images"
        ann_dir = root / "MoNuSeg 2018 Training Data" / "Annotations"
    elif split == "test":
        image_dir = ann_dir = root / "MoNuSegTestData"
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return [p.stem for p in sorted(image_dir.glob("*.tif")) if (ann_dir / (p.stem + ".xml")).exists()]


def monuseg_fingerprint(stems, counts, val_fraction, split_seed, test_counts,
                        test_stems) -> dict:
    """Fingerprint the MoNuSeg split.

    ``stems``/``test_stems`` are the source image names in cache order. They
    come from the flat cache's meta.json when present so the fingerprint does
    not require the multi-hundred-megabyte source TIFs to be staged alongside.
    """
    train_idx, val_idx, val_images = monuseg_image_level_split(counts, val_fraction, split_seed)
    stems = list(stems)
    if len(stems) != len(counts):
        raise SplitDriftError(
            f"MoNuSeg cache holds {len(counts)} images but {len(stems)} source image "
            f"names were resolved — the cache is stale, rebuild it."
        )
    return {
        "dataset": "monuseg",
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "n_source_images": len(stems),
        "source_images": stems,
        "patch_counts": [int(c) for c in counts],
        "val_images": [stems[i] for i in val_images],
        "train_images": [stems[i] for i in range(len(stems)) if i not in set(val_images)],
        "n_train_patches": len(train_idx),
        "n_val_patches": len(val_idx),
        "train_patch_files": [f"{i:05d}_img.npy" for i in train_idx],
        "val_patch_files": [f"{i:05d}_img.npy" for i in val_idx],
        "test_images": list(test_stems),
        "test_patch_counts": [int(c) for c in test_counts],
    }


def pannuke_fingerprint(train_split, val_split, test_split, counts: dict) -> dict:
    """PanNuke uses the dataset's own predefined folds — no RNG is involved.

    The fold assignment plus the example count per fold is the whole split, so
    that is what gets hashed.
    """
    return {
        "dataset": "pannuke",
        "split_seed": None,  # folds are fixed by the dataset, not sampled
        "folds": {"train": train_split, "val": val_split, "test": test_split},
        "num_examples": {k: int(v) for k, v in sorted(counts.items())},
    }


def fingerprint_hash(detail: dict) -> str:
    return _hash(detail)


# ── Manifest read / assert ───────────────────────────────────────────────────

def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise SplitDriftError(
            f"{p} is missing. Generate it once with "
            f"`python scripts/make_split_manifest.py` and commit it."
        )
    return json.loads(p.read_text())


def assert_split(dataset: str, detail: dict, manifest_path: str | Path = MANIFEST_PATH) -> str:
    """Assert the realised split matches the recorded one. Returns the hash."""
    got = fingerprint_hash(detail)
    manifest = load_manifest(manifest_path)
    if dataset not in manifest:
        raise SplitDriftError(
            f"No recorded split for dataset {dataset!r} in {manifest_path}. "
            f"Regenerate the manifest."
        )
    expected = manifest[dataset]["hash"]
    if got != expected:
        diffs = _describe_drift(manifest[dataset].get("detail", {}), detail)
        raise SplitDriftError(
            f"Split drift for {dataset!r}: expected {expected[:12]}, got {got[:12]}.\n"
            f"{diffs}\n"
            f"Every run in a table must share one split. Fix the data or, if the "
            f"change is intended, regenerate the manifest AND re-run every seed."
        )
    return got


def _describe_drift(expected: dict, got: dict) -> str:
    lines = []
    for key in sorted(set(expected) | set(got)):
        e, g = expected.get(key, "<missing>"), got.get(key, "<missing>")
        if e != g:
            if isinstance(e, list) and isinstance(g, list):
                lines.append(f"  {key}: {len(e)} entries -> {len(g)} entries")
            else:
                lines.append(f"  {key}: {e!r} -> {g!r}")
    return "\n".join(lines) if lines else "  (hash differs but no field-level diff found)"
