# src/sweep.py
"""The (dataset x encoder x train_seed) matrix, resolved from configs/sweep.yaml.

Run order is config-major: all 5 seeds of one configuration are adjacent. A
Slurm array task claims work dynamically (see ``scripts/run_sweep.py``), so
ordering only affects which runs finish first, and finishing whole
configurations first makes partial aggregation useful.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.config import load_config
from src.runspec import RunSpec

SWEEP_PATH = Path("configs/sweep.yaml")


def load_sweep(path: str | Path = SWEEP_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_specs(path: str | Path = SWEEP_PATH, run_root: str | Path = "runs",
                cache_root: str | None = None) -> list[tuple[RunSpec, Path]]:
    """Every (config, seed) pair as a ``(RunSpec, run_dir)`` list.

    ``cache_root`` overrides ``data.cache_root`` so a Slurm task can point the
    data path at node-local NVMe without touching the committed configs.
    """
    sweep = load_sweep(path)
    seeds = sweep["train_seeds"]
    split_seed = sweep["split_seed"]
    perf = sweep.get("performance", {})
    run_root = Path(run_root)

    specs: list[tuple[RunSpec, Path]] = []
    for entry in sweep["runs"]:
        base_cfg = load_config(entry["base"], entry.get("overrides") or [])
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["performance"] = {**perf, **cfg.get("performance", {})}
            # Recorded in the config so results.json is self-describing.
            cfg["experiment"] = {
                **cfg["experiment"],
                "name": f"{entry['id']}_seed{seed}",
                "train_seed": seed,
                "split_seed": split_seed,
                "sweep_id": entry["id"],
            }
            if cache_root is not None:
                cfg["data"] = {**cfg["data"], "cache_root": cache_root}
            run_id = f"{entry['id']}_seed{seed}"
            specs.append((
                RunSpec(run_id=run_id, cfg=cfg, train_seed=seed, split_seed=split_seed,
                        dataset=entry["dataset"], encoder=entry["encoder"],
                        loss=entry["loss"], table=str(entry["table"])),
                run_root / entry["id"] / f"seed{seed}",
            ))
    return specs


def sweep_entries(path: str | Path = SWEEP_PATH) -> dict[str, dict]:
    """Configuration metadata keyed by sweep id (row labels, published numbers)."""
    return {e["id"]: e for e in load_sweep(path)["runs"]}
