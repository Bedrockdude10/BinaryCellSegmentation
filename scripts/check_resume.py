#!/usr/bin/env python3
# scripts/check_resume.py
"""Prove that a resumed run reproduces an uninterrupted one exactly.

    python scripts/check_resume.py                       # default: MoNuSeg VGG, 4 epochs
    python scripts/check_resume.py --epochs 6 --stop-after 3

Trains one configuration twice:

  A. straight through for N epochs;
  B. interrupted after K epochs (checkpointed the way a wall-clock kill would
     leave it), then resumed to N.

If RNG state, optimizer, scheduler and counters are all restored correctly, B's
per-epoch train loss and validation Dice match A's **exactly**, epoch for epoch.
Anything less means a preempted run is not the same experiment as an unpreempted
one, and whether a run got preempted is pure scheduling luck — exactly the kind
of hidden confound this sweep exists to remove.

This is why the sweep sets ``persistent_workers: false``. Persistent workers
keep their numpy augmentation state inside worker processes across epochs, where
it cannot be checkpointed; respawning them each epoch makes an epoch's
augmentation a function of the (checkpointed) loader generator state alone.
``--persistent-workers`` runs the same test the other way to show the
difference.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import RunInterrupted, run  # noqa: E402
from src.sweep import build_specs  # noqa: E402


def _spec(sweep_id: str, seed: int, epochs: int, persistent: bool, cache_root: str | None):
    for spec, _ in build_specs(cache_root=cache_root):
        if spec.cfg["experiment"]["sweep_id"] == sweep_id and spec.train_seed == seed:
            spec.cfg["training"] = {**spec.cfg["training"], "epochs": epochs}
            spec.cfg["performance"] = {**spec.cfg["performance"],
                                       "persistent_workers": persistent,
                                       "checkpoint_every_epochs": 10**9}
            return spec
    raise SystemExit(f"no sweep entry {sweep_id!r} at seed {seed}")


def _history(res: dict) -> list[tuple[float, float]]:
    return list(zip(res["train_loss_per_epoch"], res["val_dice_per_epoch"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="monuseg_vgg_bce")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--stop-after", type=int, default=2)
    ap.add_argument("--persistent-workers", action="store_true",
                    help="Run the test with persistent workers, which is expected to FAIL")
    ap.add_argument("--root", default="runs_smoke/_resume_check")
    ap.add_argument("--cache-root", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    persistent = args.persistent_workers

    print(f"config={args.config} seed={args.seed} epochs={args.epochs} "
          f"stop_after={args.stop_after} persistent_workers={persistent}\n")

    print(f"A. uninterrupted, {args.epochs} epochs")
    res_a = run(_spec(args.config, args.seed, args.epochs, persistent, args.cache_root),
                root / "full")

    print(f"\nB. interrupted after {args.stop_after}, then resumed")
    spec_b = _spec(args.config, args.seed, args.epochs, persistent, args.cache_root)
    try:
        run(spec_b, root / "resumed", stop_after_epoch=args.stop_after)
    except RunInterrupted as e:
        print(f"   interrupted as designed: {e}")
    else:
        raise SystemExit("the interruption hook did not fire")
    ckpt = root / "resumed" / "last.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} — nothing to resume from")
    res_b = run(_spec(args.config, args.seed, args.epochs, persistent, args.cache_root),
                root / "resumed")

    hist_a, hist_b = _history(res_a), _history(res_b)
    print(f"\n{'epoch':>5s}  {'A train loss':>14s} {'B train loss':>14s}   "
          f"{'A val dice':>12s} {'B val dice':>12s}  match")
    ok = True
    for i, ((la, da), (lb, db)) in enumerate(zip(hist_a, hist_b), start=1):
        same = (la == lb) and (da == db)
        ok &= same
        marker = "=" if same else "DIFFER"
        print(f"{i:5d}  {la:14.10f} {lb:14.10f}   {da:12.9f} {db:12.9f}  {marker}"
              + ("   <- resumed here" if i == args.stop_after + 1 else ""))

    print(f"\nresumed_from_epoch: {res_b['resumed_from_epoch']}  "
          f"epochs_run A/B: {res_a['epochs_run']}/{res_b['epochs_run']}")
    print(f"test metrics  A: {res_a['test']['dice']:.10f}  B: {res_b['test']['dice']:.10f}")
    if len(hist_a) != len(hist_b):
        ok = False
        print(f"FAIL: A ran {len(hist_a)} epochs, B ran {len(hist_b)}", file=sys.stderr)

    print()
    if ok and res_a["test"]["dice"] == res_b["test"]["dice"]:
        print("PASS: resumed run is bit-identical to the uninterrupted run")
        return 0
    print("FAIL: resumed run diverges from the uninterrupted run", file=sys.stderr)
    if persistent:
        print("(expected with --persistent-workers: worker numpy state lives in "
              "other processes and cannot be checkpointed)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
