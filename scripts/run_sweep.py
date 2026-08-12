#!/usr/bin/env python3
# scripts/run_sweep.py
"""Driver for the multi-seed sweep. One entry point for Slurm and for local runs.

    # what would run, and what is already banked
    python scripts/run_sweep.py --list

    # one run (debugging / single-config repro)
    python scripts/run_sweep.py --index 3

    # Slurm array task: claim work until the matrix is done or time runs out
    python scripts/run_sweep.py --worker --time-budget 28800

    # local sequential fallback (no Slurm)
    python scripts/run_sweep.py --local

Resumability works at two levels:

* a run whose ``results.json`` says ``complete`` is skipped without loading a
  model at all;
* a run killed mid-training leaves ``last.pt`` and is picked up from that epoch,
  RNG state included.

Work is claimed dynamically through atomic ``mkdir`` rather than statically
striped across array tasks, so a task that draws several slow configurations
does not become the tail of the whole sweep, and resubmitting the identical
array is always safe.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runspec import is_complete  # noqa: E402
from src.sweep import build_specs  # noqa: E402

CLAIM_DIRNAME = ".claims"
HEARTBEAT = "heartbeat"
STALE_SECONDS = 1800


# ── Claiming ─────────────────────────────────────────────────────────────────

def _claim_dir(run_root: Path, run_id: str) -> Path:
    return run_root / CLAIM_DIRNAME / run_id


def try_claim(run_root: Path, run_id: str, stale_seconds: int = STALE_SECONDS) -> bool:
    """Atomically claim a run. Returns False if someone else holds it."""
    d = _claim_dir(run_root, run_id)
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        d.mkdir()
    except FileExistsError:
        hb = d / HEARTBEAT
        try:
            age = time.time() - hb.stat().st_mtime
        except OSError:
            age = float("inf")
        if age < stale_seconds:
            return False
        # Owner died (node failure, hard kill): steal the claim.
        print(f"  reclaiming stale claim for {run_id} (heartbeat {age / 60:.0f} min old)")
    token = f"{socket.gethostname()}:{os.getpid()}:{os.environ.get('SLURM_JOB_ID', '-')}"
    (d / HEARTBEAT).touch()
    (d / "owner.json").write_text(json.dumps({
        "token": token, "host": socket.gethostname(), "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "claimed_at": time.time(),
    }, indent=2))
    # Read back: if two tasks raced (or mkdir was not atomic on this shared
    # filesystem), only the one whose token survived proceeds. Cheap insurance
    # against two tasks training into the same run directory.
    time.sleep(0.5)
    try:
        if json.loads((d / "owner.json").read_text()).get("token") != token:
            print(f"  lost claim race for {run_id}")
            return False
    except (OSError, json.JSONDecodeError):
        return False
    return True


def release_claim(run_root: Path, run_id: str) -> None:
    d = _claim_dir(run_root, run_id)
    for f in (HEARTBEAT, "owner.json"):
        (d / f).unlink(missing_ok=True)
    try:
        d.rmdir()
    except OSError:
        pass


class Heartbeat:
    """Touches the claim file while a run is in flight so it is not seen as stale."""

    def __init__(self, run_root: Path, run_id: str, interval: int = 60):
        self.path = _claim_dir(run_root, run_id) / HEARTBEAT
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def beat():
            while not self._stop.wait(self.interval):
                try:
                    self.path.touch()
                except OSError:
                    pass
        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False


# ── Selection ────────────────────────────────────────────────────────────────

def select(specs, only, seeds, index):
    picked = list(enumerate(specs))
    if index is not None:
        picked = [(i, s) for i, s in picked if i == index]
        if not picked:
            raise SystemExit(f"--index {index} out of range (0..{len(specs) - 1})")
    if only:
        picked = [(i, s) for i, s in picked
                  if any(pat in s[0].run_id for pat in only)]
    if seeds:
        picked = [(i, s) for i, s in picked if s[0].train_seed in seeds]
    return picked


def apply_max_epochs(spec, max_epochs: int | None):
    if max_epochs is None:
        return spec
    spec.cfg["training"] = {**spec.cfg["training"], "epochs": int(max_epochs)}
    return spec


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--cache-root", default=None,
                    help="Override data.cache_root (Slurm points this at node-local NVMe)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int, default=None, help="Run exactly this matrix index")
    ap.add_argument("--worker", action="store_true", help="Claim-and-run loop (Slurm array task)")
    ap.add_argument("--local", action="store_true", help="Run everything sequentially, no claiming")
    ap.add_argument("--only", action="append", default=[], help="Substring filter on run_id")
    ap.add_argument("--seeds", type=int, action="append", default=[],
                    help="Restrict to these train seeds (repeatable)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Cap epochs — smoke tests only, NOT for reportable runs")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="Seconds from now after which no new epoch is started")
    ap.add_argument("--reserve", type=float, default=900,
                    help="Seconds of the budget held back for test eval and checkpointing")
    ap.add_argument("--claim-stale-seconds", type=int, default=STALE_SECONDS)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    specs = build_specs(args.sweep, run_root=run_root, cache_root=args.cache_root)
    picked = select(specs, args.only, set(args.seeds), args.index)

    if args.list:
        done = 0
        for i, (spec, run_dir) in picked:
            complete = is_complete(run_dir)
            done += complete
            partial = (run_dir / "last.pt").exists()
            state = "complete" if complete else ("partial" if partial else "-")
            print(f"{i:3d}  {spec.run_id:38s} {state:9s} {run_dir}")
        print(f"\n{done}/{len(picked)} complete")
        return 0

    # src.runner pulls in torch, which the login node OOM-kills. Everything above
    # this point (--list, selection) stays torch-free so progress can be checked
    # from the login node; the heavy import happens only when we intend to train.
    from src.runner import RunInterrupted, install_signal_handlers, run  # noqa: PLC0415

    if args.max_epochs is not None:
        print(f"!! --max-epochs {args.max_epochs}: SMOKE MODE, results are not reportable",
              file=sys.stderr)

    deadline = time.time() + args.time_budget - args.reserve if args.time_budget else None
    install_signal_handlers()
    print(f"host={platform.node()} runs={len(picked)} "
          f"deadline={'none' if deadline is None else time.strftime('%H:%M:%S', time.localtime(deadline))}")

    completed, skipped, interrupted, failed = 0, 0, 0, 0
    for i, (spec, run_dir) in picked:
        if is_complete(run_dir):
            skipped += 1
            continue
        if deadline is not None and time.time() >= deadline:
            print("Time budget exhausted; stopping cleanly.")
            break

        claimed = True
        if args.worker:
            claimed = try_claim(run_root, spec.run_id, args.claim_stale_seconds)
            if not claimed:
                continue

        print(f"[{i}] {spec.run_id} -> {run_dir}", flush=True)
        apply_max_epochs(spec, args.max_epochs)
        try:
            with Heartbeat(run_root, spec.run_id) if args.worker else _NullCtx():
                run(spec, run_dir, deadline_epoch=deadline)
            completed += 1
        except RunInterrupted as e:
            print(f"  interrupted: {e}", flush=True)
            interrupted += 1
            if args.worker:
                release_claim(run_root, spec.run_id)
            break
        except Exception as e:  # noqa: BLE001 - one bad config must not sink the array task
            failed += 1
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            (run_dir).mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILED.txt").write_text(f"{type(e).__name__}: {e}\n")
        finally:
            if args.worker and claimed:
                release_claim(run_root, spec.run_id)

    print(f"\ncompleted={completed} skipped={skipped} interrupted={interrupted} failed={failed}")
    # Interruption is expected (wall clock) and must not mark the array task
    # failed; a genuine exception should.
    return 1 if failed else 0


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


if __name__ == "__main__":
    raise SystemExit(main())
