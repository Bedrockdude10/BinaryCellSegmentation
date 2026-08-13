#!/usr/bin/env python3
# scripts/check_repro.py
"""Gate before releasing the remaining seeds: does seed 42 reproduce the paper?

    python scripts/check_repro.py                # after the 7 seed-42 runs finish
    python scripts/check_repro.py --tolerance 0.015

The published Tables 1 and 2 were produced at seed 42 on an Apple M3 via the
PyTorch MPS backend. The re-run is on a Tesla V100 via CUDA. Exact reproduction
is neither expected nor possible, for two specific reasons:

1. **Backend and AMP regime.** The reviewed code enables
   ``torch.autocast(dtype=float16)`` for any non-CPU device. On MPS that was at
   best partially effective; on CUDA fp16 autocast is fully active. Different
   kernels, different reduction orders, genuinely different arithmetic.
2. **The augmentation stream necessarily changed.** Passing the DataLoader an
   explicit ``generator=`` — a required correctness fix, since the stream
   previously depended on unrelated global-RNG consumption — changes which
   augmentations each sample receives. Seed 42 is therefore a fresh draw from
   the seed distribution, not a replay of the published trajectory.

So this script checks two things:

* **Tolerance** (default 0.015 Dice). The paper states that differences below
  ~0.005 Dice are not meaningful run-to-run noise for this setup. 0.015 is three
  times that: loose enough to absorb the backend change and a different seed
  draw, but still 3x smaller than the 0.049 Dice VGG-over-Swin gap the paper's
  central claim rests on. A config outside this band means something structural
  changed, not just noise.
* **Structural claims.** The tolerance can pass while the paper's actual
  argument breaks (or vice versa), so the orderings and gap magnitudes the
  conclusions depend on are asserted directly.

Exits non-zero if either check fails — that is the signal to stop and
investigate rather than burning 28 more runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sweep import build_specs, sweep_entries  # noqa: E402

# Claims the paper's conclusions rest on, as (a, b, relation, threshold) on Dice.
STRUCTURAL_CLAIMS = [
    ("pannuke_vgg_bce", "pannuke_swin_pretrained", "exceeds_by", 0.030),
    ("pannuke_resnet_bce", "pannuke_swin_pretrained", "exceeds_by", 0.020),
    ("pannuke_vgg_bce", "pannuke_resnet_bce", "within", 0.010),
    ("pannuke_vgg_bce", "pannuke_vgg_dice", "within", 0.010),
    ("pannuke_swin_pretrained", "pannuke_swin_scratch", "exceeds_by", 0.150),
    ("monuseg_vgg_bce", "monuseg_swin_pretrained", "exceeds_by", 0.030),
]
OTSU_DICE = 0.4818559766684619  # outputs/classical_otsu/eval_fold3.json, deterministic


def load_seed(run_root: Path, seed: int) -> dict[str, dict]:
    out = {}
    for spec, run_dir in build_specs(run_root=run_root):
        if spec.train_seed != seed:
            continue
        path = run_dir / "results.json"
        if path.exists():
            res = json.loads(path.read_text())
            if res.get("status") == "complete":
                out[spec.cfg["experiment"]["sweep_id"]] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance", type=float, default=0.015,
                    help="Max |Dice delta| vs the published single-seed value")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # Supplementary configs have no published counterpart to reproduce.
    entries = {k: v for k, v in sweep_entries().items() if v.get("published")}
    got = load_seed(Path(args.run_root), args.seed)
    missing = [k for k in entries if k not in got]
    if missing:
        print(f"Not ready: seed {args.seed} incomplete for {missing}", file=sys.stderr)
        print(f"Finish those runs first (python scripts/run_sweep.py --seeds {args.seed} --list)",
              file=sys.stderr)
        return 2

    gpus = {r["gpu_model"] for r in got.values()}
    splits = {(r["dataset"], r["split_hash"]) for r in got.values()}
    print(f"seed {args.seed} | gpu {gpus} | {len(got)} configs")
    for d, h in sorted(splits):
        print(f"  split {d}: {h[:12]}")
    if len(gpus) > 1:
        print(f"FAIL: seed-{args.seed} runs span multiple GPU models: {gpus}", file=sys.stderr)
        return 1

    print(f"\ntolerance: +-{args.tolerance:.3f} Dice\n")
    print(f"{'config':26s} {'pub Dice':>9s} {'new Dice':>9s} {'delta':>8s}  "
          f"{'pub IoU':>8s} {'new IoU':>8s} {'delta':>8s}  {'pub ep':>6s} {'new ep':>6s}  ok")
    failures = []
    rows = {}
    for sid, entry in entries.items():
        pub, res = entry["published"], got[sid]
        d_new, i_new = res["test"]["dice"], res["test"]["iou"]
        dd, di = d_new - pub["dice"], i_new - pub["iou"]
        ok = abs(dd) <= args.tolerance
        rows[sid] = {"published": pub, "dice": d_new, "iou": i_new,
                     "delta_dice": dd, "delta_iou": di,
                     "epochs": res["epochs_run"], "within_tolerance": ok}
        if not ok:
            failures.append(f"{sid}: Dice delta {dd:+.4f} exceeds +-{args.tolerance:.3f} "
                            f"(published {pub['dice']:.4f}, new {d_new:.4f})")
        print(f"{sid:26s} {pub['dice']:9.4f} {d_new:9.4f} {dd:+8.4f}  "
              f"{pub['iou']:8.4f} {i_new:8.4f} {di:+8.4f}  "
              f"{pub['epochs']:6d} {res['epochs_run']:6d}  {'yes' if ok else 'NO'}")

    max_abs = max(abs(r["delta_dice"]) for r in rows.values())
    mean_abs = sum(abs(r["delta_dice"]) for r in rows.values()) / len(rows)
    print(f"\nDice delta: max |{max_abs:.4f}|, mean |{mean_abs:.4f}|")

    print("\nstructural claims the paper's conclusions rest on:")
    for a, b, rel, thr in STRUCTURAL_CLAIMS:
        da, db = rows[a]["dice"], rows[b]["dice"]
        if rel == "exceeds_by":
            ok = (da - db) >= thr
            desc = f"{a} - {b} = {da - db:+.4f} (need >= {thr:+.3f})"
        else:
            ok = abs(da - db) <= thr
            desc = f"|{a} - {b}| = {abs(da - db):.4f} (need <= {thr:.3f})"
        print(f"  [{'ok ' if ok else 'FAIL'}] {desc}")
        if not ok:
            failures.append(f"structural claim broken: {desc}")

    scratch = rows["pannuke_swin_scratch"]["dice"]
    print(f"  [{'ok ' if scratch > OTSU_DICE else 'FAIL'}] Swin-scratch {scratch:.4f} "
          f"> Otsu {OTSU_DICE:.4f}")
    if scratch <= OTSU_DICE:
        failures.append("from-scratch Swin no longer clears the Otsu floor")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"seed": args.seed, "tolerance": args.tolerance, "rows": rows,
             "failures": failures, "gpu_model": gpus.pop()}, indent=2, sort_keys=True))
        print(f"\nWrote {args.json}")

    print()
    if failures:
        print("REPRODUCTION CHECK FAILED — do not release the remaining seeds:",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("REPRODUCTION CHECK PASSED — safe to release the remaining seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
