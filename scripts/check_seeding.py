#!/usr/bin/env python3
# scripts/check_seeding.py
"""Verify the seeding guarantees, and show what the reviewed code actually did.

    python scripts/check_seeding.py

Findings this script demonstrates (all measured, not assumed):

* PyTorch DOES reseed ``numpy`` inside DataLoader workers, from a ``base_seed``
  it draws when the iterator is created. So with ``num_workers > 0`` — which
  every config in this study uses — the reviewed code's numpy augmentation was
  already deterministic given ``torch.manual_seed(seed)``. The often-cited
  "numpy in workers is unseeded" bug did NOT bite the published runs.
* It was nonetheless fragile in two concrete ways, both fixed here:
    - ``base_seed`` came from the GLOBAL torch RNG because no ``generator=`` was
      passed, so the augmentation and shuffle streams depended on how many
      global draws unrelated code had already consumed. Constructing the model
      before rather than after the loaders changes every batch (check 5).
    - with ``num_workers = 0`` nothing seeds numpy at all, so augmentation was
      genuinely non-reproducible on that path (check 4).
* ``random`` was never seeded, and ``numpy`` was never seeded in the main
  process. ``set_seed`` now covers both.

Checks:
  1. same ``train_seed`` -> bit-identical augmented batches (num_workers > 0)
  2. different ``train_seed`` -> different batches
  3. ``split_seed`` is independent of ``train_seed``
  4. num_workers=0 is reproducible now, and was not before
  5. the stream does not depend on unrelated global-RNG consumption

Exits non-zero if any guarantee fails.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.builders import _monuseg_counts, build_loaders  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.monuseg import MoNuSegDataset  # noqa: E402
from src.data.preprocessing import PreprocessedDataset  # noqa: E402
from src.models.encoders import VGGEncoder  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.splits import monuseg_image_level_split  # noqa: E402

CFG = ("configs/default.yaml", ["configs/ablations/monuseg.yaml"])
N_BATCHES = 5


def _digest(loader, n: int = N_BATCHES) -> str:
    h = hashlib.sha256()
    for i, (images, masks) in enumerate(loader):
        if i >= n:
            break
        h.update(images.numpy().tobytes())
        h.update(masks.numpy().tobytes())
    return h.hexdigest()[:16]


def _cfg(workers: int) -> dict:
    cfg = load_config(*CFG)
    cfg["performance"] = {"num_workers": workers, "persistent_workers": workers > 0,
                          "pin_memory": False, "prefetch_factor": 2}
    return cfg


def fixed_pass(train_seed: int, workers: int = 4, model_first: bool = False) -> str:
    """The current code path."""
    cfg = _cfg(workers)
    set_seed(train_seed)
    if model_first:
        UNet(VGGEncoder())  # consume global-RNG draws before the loaders exist
    loader, _, _, _ = build_loaders(cfg, train_seed=train_seed, split_seed=42, device="cpu")
    return _digest(loader)


def original_pass(train_seed: int, workers: int = 4, model_first: bool = False) -> str:
    """The reviewed code path: torch.manual_seed only, no generator, no worker_init_fn."""
    cfg = load_config(*CFG)
    torch.manual_seed(train_seed)
    full = MoNuSegDataset(split="train", data_root=cfg["data"]["data_root"])
    raw_train, _ = full.image_level_split(val_fraction=cfg["data"]["val_fraction"])
    ds = PreprocessedDataset(raw_train, cfg["preprocessing"], is_train=True, feature_fn=None)
    if model_first:
        UNet(VGGEncoder())
    loader = DataLoader(ds, shuffle=True, batch_size=cfg["data"]["batch_size"],
                        num_workers=workers, pin_memory=False)
    return _digest(loader)


def main() -> int:
    failures: list[str] = []

    print("1. same train_seed -> identical augmented batches (num_workers=4)")
    a, b = fixed_pass(42), fixed_pass(42)
    print(f"   {a} vs {b}")
    if a != b:
        failures.append("two identically-seeded passes differ — seeding is not reproducible")
    else:
        print("   OK")

    print("\n2. different train_seed -> different batches")
    c = fixed_pass(43)
    print(f"   seed 43: {c}")
    if c == a:
        failures.append("seeds 42 and 43 give identical batches — train_seed has no effect")
    else:
        print("   OK")

    print("\n3. split_seed independent of train_seed")
    counts = _monuseg_counts(load_config(*CFG), "train")
    splits = set()
    for train_seed in (42, 43, 44):
        set_seed(train_seed)  # deliberately perturb the global RNGs first
        tr, va, val_imgs = monuseg_image_level_split(counts, 0.2, 42)
        splits.add((tuple(val_imgs), len(tr), len(va)))
    only = next(iter(splits))
    print(f"   {len(splits)} distinct split(s) across 3 train seeds; "
          f"val images {list(only[0])}, {only[1]}/{only[2]} train/val patches")
    if len(splits) != 1:
        failures.append("the split changes with train_seed — split and training RNG are coupled")
    else:
        print("   OK split determined by split_seed alone")

    print("\n4. num_workers=0 path")
    o0a, o0b = original_pass(42, workers=0), original_pass(42, workers=0)
    f0a, f0b = fixed_pass(42, workers=0), fixed_pass(42, workers=0)
    print(f"   reviewed code: {o0a} vs {o0b} -> "
          f"{'reproducible' if o0a == o0b else 'NOT reproducible (numpy never seeded)'}")
    print(f"   current  code: {f0a} vs {f0b} -> "
          f"{'reproducible' if f0a == f0b else 'NOT reproducible'}")
    if f0a != f0b:
        failures.append("num_workers=0 is still not reproducible")

    print("\n5. stream must not depend on unrelated global-RNG consumption")
    print("   (building the model before vs after the loaders)")
    o_after, o_before = original_pass(42, 4), original_pass(42, 4, model_first=True)
    f_after, f_before = fixed_pass(42, 4), fixed_pass(42, 4, model_first=True)
    print(f"   reviewed code: {o_after} vs {o_before} -> "
          f"{'independent' if o_after == o_before else 'COUPLED to global RNG state'}")
    print(f"   current  code: {f_after} vs {f_before} -> "
          f"{'independent' if f_after == f_before else 'COUPLED to global RNG state'}")
    if f_after != f_before:
        failures.append("the augmentation/shuffle stream still depends on global-RNG "
                        "consumption order — the explicit generator= is not taking effect")

    print("\n6. workers do get numpy seeded (reviewed code, num_workers=4)")
    oa, ob = original_pass(42, 4), original_pass(42, 4)
    print(f"   {oa} vs {ob} -> {'reproducible' if oa == ob else 'NOT reproducible'}")
    print("   PyTorch reseeds numpy per worker from base_seed, so the published "
          "runs' augmentation was seeded — via the global torch RNG, not explicitly.")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("all seeding guarantees hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
