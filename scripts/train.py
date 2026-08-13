#!/usr/bin/env python3
# scripts/train.py
"""Train one configuration at one seed.

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --override configs/ablations/dice_loss.yaml
    python scripts/train.py --config configs/default.yaml --train-seed 43

For the multi-seed sweep use ``scripts/run_sweep.py``, which drives the same
runner over the whole (dataset x encoder x seed) matrix with claiming and
resume. This script is the single-run entry point: useful for debugging one
config, and it is what the sweep's per-run behaviour reduces to.

Resume is automatic — a ``last.pt`` in the output directory is picked up,
including RNG state. Pass ``--no-resume`` to start over.

The builders are re-exported here because ``scripts/evaluate.py`` loads this
module by path and calls ``build_model`` / ``make_feature_fn`` / ``resolve_device``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.builders import (  # noqa: E402,F401 - re-exported for scripts/evaluate.py
    DEFAULT_SPLIT_SEED, BinaryMaskFeature, build_loaders, build_loss, build_model,
    build_optimizer, build_scheduler, build_test_loader, make_feature_fn, resolve_device,
)
from src.config import load_config  # noqa: E402
from src.runner import RunSpec, install_signal_handlers, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--train-seed", type=int, default=None,
                    help="Training-randomness seed (default: experiment.seed in the config)")
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED,
                    help="Fixed across the whole study; do not vary it per run")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <experiment.output_dir>/<experiment.name>")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None, help="Smoke tests only")
    ap.add_argument("--amp", dest="amp", action="store_true", default=None,
                    help="Force fp16 autocast on. NOT what the sweep uses — on CUDA this "
                         "stops the ResNet encoder training (scripts/diagnose_amp.py)")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="Train in fp32, the regime the published numbers came from")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    exp = cfg["experiment"]
    train_seed = args.train_seed if args.train_seed is not None else exp.get("seed", 42)
    if args.max_epochs is not None:
        cfg["training"] = {**cfg["training"], "epochs": int(args.max_epochs)}
    cfg["experiment"] = {**exp, "train_seed": train_seed, "split_seed": args.split_seed}
    if args.amp is not None:
        cfg["performance"] = {**cfg.get("performance", {}), "amp": args.amp}

    out_dir = Path(args.out_dir) if args.out_dir else Path(exp["output_dir"]) / exp["name"]
    dataset = cfg["data"].get("dataset", "pannuke")
    loss = cfg["loss"] if isinstance(cfg["loss"], str) else cfg["loss"]["function"]

    spec = RunSpec(
        run_id=f"{exp['name']}_seed{train_seed}", cfg=cfg, train_seed=train_seed,
        split_seed=args.split_seed, dataset=dataset,
        encoder=cfg["model"]["architecture"], loss=loss,
        table="2" if dataset == "monuseg" else "1",
    )

    install_signal_handlers()
    results = run(spec, out_dir, resume=not args.no_resume)
    print(f"\n{results['run_id']}: test {results['test']} "
          f"({results['epochs_run']} epochs, {results['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
