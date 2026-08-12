# src/runner.py
"""One training run, end to end, with resumable state and a results artifact.

The training recipe itself is unchanged from the reviewed experiment: same
model construction, loss, optimizer, LR schedule, augmentation policy, AMP
setting and early-stopping rule. What is new around it:

* ``train_seed`` seeds every RNG through ``src.seeding``; ``split_seed`` is
  separate and fixed.
* ``last.pt`` carries optimizer, scheduler, epoch counter, early-stopping
  counter and **RNG state**, so a run killed by the wall clock resumes the same
  random stream rather than starting a fresh one mid-experiment.
* Every run drops one machine-readable ``results.json`` with the provenance the
  camera-ready needs (git SHA, split hash, GPU model, per-epoch val Dice,
  wall clock, peak memory, library versions).

Checkpoint layout per run directory:
    best.pt      model weights at the best validation epoch (for test eval)
    last.pt      full resumable state, deleted once the run completes
    results.json the artifact ``scripts/aggregate.py`` reads
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.builders import (
    build_loaders, build_loss, build_model, build_optimizer, build_scheduler,
    build_test_loader, count_params, resolve_device, total_params,
)
from src.evaluation.evaluator import evaluate
from src.evaluation.metrics import dice, iou
from src.runspec import RunSpec, is_complete, results_path  # noqa: F401
from src.seeding import capture_rng_state, restore_rng_state, set_seed
from src.splits import assert_split
from src.train import train_one_epoch

log = logging.getLogger(__name__)

# Set by SIGUSR1/SIGTERM (Slurm sends these before the wall clock kills us).
_STOP_REQUESTED = False


class RunInterrupted(RuntimeError):
    """Raised when a run stopped early for time/preemption, not for convergence.

    State is checkpointed before this propagates, so resubmitting continues.
    """


def install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        log.warning("Signal %s received — will checkpoint and stop after this epoch.", signum)

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # not in main thread / unsupported
            pass


# ── Provenance ───────────────────────────────────────────────────────────────

def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=15, check=True).stdout.strip()
    except Exception:
        return "unknown"


_CODE_GLOBS = ("src/**/*.py", "scripts/*.py", "configs/*.yaml", "configs/ablations/*.yaml")


def code_hash(repo_root: Path | None = None) -> str:
    """Content hash of the training code and configs.

    Complements the git SHA: runs are launched from a working tree that may have
    uncommitted changes, and ``git_sha`` alone would not distinguish them. This
    pins the actual bytes that produced a result, so the aggregator's "one
    regime" claim can be checked even mid-development.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for pattern in _CODE_GLOBS:
        for path in sorted(root.glob(pattern)):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def provenance(device: str) -> dict:
    """Everything needed to prove all runs in a table came from one regime."""
    gpu_model, gpu_capability = None, None
    if device == "cuda" and torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        gpu_capability = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    elif device == "mps":
        gpu_model = f"AppleSilicon-MPS ({platform.processor() or platform.machine()})"
    else:
        gpu_model = f"CPU ({platform.machine()})"

    try:
        import timm
        timm_version = timm.__version__
    except Exception:
        timm_version = None

    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "code_hash": code_hash(),
        "gpu_model": gpu_model,
        "gpu_capability": gpu_capability,
        "device": device,
        "node": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "timm": timm_version,
        },
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


# ── Run ──────────────────────────────────────────────────────────────────────

def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _setup_logging(run_dir: Path) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
    )


def run(spec: RunSpec, run_dir: str | Path, deadline_epoch: float | None = None,
        resume: bool = True, stop_after_epoch: int | None = None) -> dict:
    """Execute (or continue) one run. Returns the results dict it wrote.

    ``stop_after_epoch`` forces an interruption at a known point. It exists so
    ``scripts/check_resume.py`` can prove a resumed run reproduces an
    uninterrupted one exactly; it is not used by the sweep.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)

    cfg = spec.cfg
    started = time.time()
    device = resolve_device(cfg["experiment"].get("device", "auto"))
    prov = provenance(device)

    (run_dir / "config.resolved.yaml").write_text(yaml.dump(cfg, sort_keys=True))
    log.info("Run %s | seed %s | device %s | gpu %s | node %s",
             spec.run_id, spec.train_seed, device, prov["gpu_model"], prov["node"])

    # Seed BEFORE anything stochastic: weight init and the loader generator.
    set_seed(spec.train_seed)

    train_loader, val_loader, split_detail, loader_gen = build_loaders(
        cfg, train_seed=spec.train_seed, split_seed=spec.split_seed, device=device)

    # Fail on line one if the split drifted, rather than contaminating a table.
    split_hash = assert_split(spec.dataset, split_detail)
    log.info("Split hash %s (%s train / %s val patches)",
             split_hash[:12], len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    metric_fns = {"dice": dice, "iou": iou}

    es = cfg["training"].get("early_stopping", {})
    patience = es.get("patience", float("inf"))
    es_metric = es.get("metric", "dice")
    max_epochs = cfg["training"]["epochs"]
    ckpt_every = int(cfg.get("performance", {}).get("checkpoint_every_epochs", 5))
    # None => src.train's original default (autocast fp16 on any non-CPU device).
    use_amp = cfg.get("performance", {}).get("amp", None)
    if use_amp is None:
        use_amp = device != "cpu"
    log.info("AMP: %s (fp16 autocast)", use_amp)

    best = -float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1
    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    prior_wall = 0.0
    prior_peak = 0
    resumed_from = None

    last_ckpt = run_dir / "last.pt"
    if resume and last_ckpt.exists():
        # weights_only=False: our own checkpoint, and RNG state is not a tensor.
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        if ck.get("train_seed") != spec.train_seed:
            raise RuntimeError(
                f"{last_ckpt} was written with train_seed {ck.get('train_seed')} "
                f"but this run is seed {spec.train_seed}. Refusing to mix runs."
            )
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        restore_rng_state(ck["rng"], loader_gen)
        start_epoch = ck["epoch"] + 1
        best, best_epoch, stale = ck["best"], ck["best_epoch"], ck["stale"]
        history = ck["history"]
        prior_wall = ck.get("wall_clock_seconds", 0.0)
        prior_peak = ck.get("peak_gpu_memory_bytes", 0)
        resumed_from = ck["epoch"]
        log.info("Resumed from epoch %s (best %s %.4f, stale %s)",
                 ck["epoch"], es_metric, best, stale)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def snapshot(epoch: int) -> None:
        _atomic_save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": capture_rng_state(loader_gen),
            "best": best, "best_epoch": best_epoch, "stale": stale,
            "history": history, "train_seed": spec.train_seed,
            "wall_clock_seconds": prior_wall + (time.time() - started),
            "peak_gpu_memory_bytes": max(prior_peak, _peak_mem(device)),
        }, last_ckpt)

    early_stopped = False
    epoch = start_epoch - 1
    epoch_seconds: list[float] = []
    interrupted_reason = None

    for epoch in range(start_epoch, max_epochs + 1):
        if _STOP_REQUESTED:
            interrupted_reason = "signal"
            break
        if deadline_epoch is not None and _would_overrun(deadline_epoch, epoch_seconds):
            interrupted_reason = "time-budget"
            break

        t_epoch = time.time()
        log.info("Epoch %s", epoch)
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                               cfg["training"].get("log_interval", 10), use_amp=use_amp)
        val = evaluate(model, val_loader, metric_fns, device=device)
        log.info("  Loss: %.4f | Val: %s", loss, val)

        history["train_loss"].append(float(loss))
        history["val_dice"].append(float(val["dice"]))
        history["val_iou"].append(float(val["iou"]))

        if val[es_metric] > best:
            best = float(val[es_metric])
            best_epoch = epoch
            stale = 0
            _atomic_save({"epoch": epoch, "model": model.state_dict(),
                          "best": best, "train_seed": spec.train_seed}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                log.info("Early stopping at epoch %s", epoch)
                early_stopped = True

        if scheduler:
            scheduler.step()

        epoch_seconds.append(time.time() - t_epoch)
        if early_stopped or epoch % ckpt_every == 0 or epoch == max_epochs:
            snapshot(epoch)
        if early_stopped:
            break
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            snapshot(epoch)
            interrupted_reason = "test-hook"
            epoch += 1  # so the shared bookkeeping below records `epoch` as done
            break

    if interrupted_reason is not None:
        snapshot(epoch - 1 if epoch >= start_epoch else start_epoch - 1)
        _write_json({
            "run_id": spec.run_id, "status": "interrupted",
            "reason": interrupted_reason, "epochs_done": epoch - 1,
            "train_seed": spec.train_seed, **prov,
        }, run_dir / "progress.json")
        raise RunInterrupted(f"{spec.run_id}: stopped at epoch {epoch - 1} ({interrupted_reason})")

    # ── Test evaluation from the best checkpoint ─────────────────────────────
    best_ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_loader, patches_per_image = build_test_loader(cfg, device=device)
    test_metrics = evaluate(model, test_loader, metric_fns, device=device,
                            patches_per_image=patches_per_image)
    log.info("Test (%s): %s", cfg["data"]["test_split"], test_metrics)

    wall = prior_wall + (time.time() - started)
    results = {
        "run_id": spec.run_id,
        "status": "complete",
        "table": spec.table,
        "dataset": spec.dataset,
        "encoder": spec.encoder,
        "loss": spec.loss,
        "train_seed": spec.train_seed,
        "split_seed": spec.split_seed,
        "split_hash": split_hash,
        "split_summary": _split_summary(split_detail),
        "config": cfg,
        "epochs_run": epoch,
        "epochs_max": max_epochs,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        f"best_val_{es_metric}": best,
        "early_stopping_metric": es_metric,
        "early_stopping_patience": patience if patience != float("inf") else None,
        "val_dice_per_epoch": history["val_dice"],
        "val_iou_per_epoch": history["val_iou"],
        "train_loss_per_epoch": history["train_loss"],
        "test": {"split": cfg["data"]["test_split"], **{k: float(v) for k, v in test_metrics.items()},
                 "protocol": "per-image mean" if patches_per_image else "pooled patches"},
        "params_total": total_params(model),
        "params_trainable": count_params(model),
        "wall_clock_seconds": wall,
        "wall_clock_seconds_this_segment": time.time() - started,
        "resumed_from_epoch": resumed_from,
        "peak_gpu_memory_bytes": max(prior_peak, _peak_mem(device)),
        "amp_enabled": bool(use_amp),
        "amp_dtype": "float16" if use_amp else "float32",
        "grad_scaler": False,
        "n_train_patches": len(train_loader.dataset),
        "n_val_patches": len(val_loader.dataset),
        "n_test_patches": len(test_loader.dataset),
        **prov,
    }
    _write_json(results, results_path(run_dir))

    # last.pt only exists to survive preemption; the run is banked now.
    last_ckpt.unlink(missing_ok=True)
    (run_dir / "progress.json").unlink(missing_ok=True)
    log.info("Done %s in %.1f min | test %s", spec.run_id, wall / 60, test_metrics)
    return results


def _peak_mem(device: str) -> int:
    if device == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch, "mps"):
        try:
            return int(torch.mps.driver_allocated_memory())
        except Exception:
            return 0
    return 0


def _would_overrun(deadline_epoch: float, epoch_seconds: list[float]) -> bool:
    """Stop before an epoch we probably cannot finish.

    Uses the slowest epoch seen so far with 25% headroom; with no history yet,
    require 10 minutes so a fresh run does not start against a nearly-expired
    allocation.
    """
    budget = max(epoch_seconds[-5:], default=0.0) * 1.25 if epoch_seconds else 600.0
    return time.time() + budget >= deadline_epoch


def _split_summary(detail: dict) -> dict:
    """Compact, human-checkable view; the full detail lives in the manifest."""
    keys = ("dataset", "split_seed", "val_fraction", "folds", "num_examples",
            "n_source_images", "n_train_patches", "n_val_patches", "val_images")
    return {k: detail[k] for k in keys if k in detail}
