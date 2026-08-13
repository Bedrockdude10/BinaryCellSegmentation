# src/runspec.py
"""The identity of one run, as plain data.

Deliberately torch-free and dependency-light. ``src.sweep`` resolves the whole
matrix into these, so the aggregation, progress-listing and reproduction-check
scripts can run on the login node — where importing torch gets OOM-killed —
while only the training path pulls in the heavy imports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSpec:
    run_id: str
    cfg: dict
    train_seed: int
    split_seed: int
    dataset: str
    encoder: str
    loss: str
    table: str


def results_path(run_dir) -> Path:
    return Path(run_dir) / "results.json"


def is_complete(run_dir) -> bool:
    """True if this run already banked a finished results.json."""
    p = results_path(run_dir)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False
