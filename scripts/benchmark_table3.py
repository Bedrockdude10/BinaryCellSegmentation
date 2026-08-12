#!/usr/bin/env python3
# scripts/benchmark_table3.py
"""Isolated inference latency / memory benchmark for Table 3.

    # on an exclusively allocated GPU node
    python scripts/benchmark_table3.py --out runs/_bench/table3.json

Deliberately separate from training: latency measured while training jobs share
the GPU is not a number worth printing. The script refuses to run if another
compute process is resident on the device (override with ``--allow-shared``,
which also stamps the output as non-exclusive).

Conditions, all recorded in the output JSON:
  * batch size 1, 3x256x256 input, ``eval()`` + ``no_grad``, fp32 (no autocast)
  * 50 warmup iterations discarded, then 300 timed iterations
  * device synchronised before starting the clock and before stopping it
  * per-iteration times kept so median and spread are visible, not just a mean
  * peak memory from ``max_memory_allocated`` over one forward pass, after
    ``reset_peak_memory_stats``, with the weights-only footprint reported too

Replaces the Table 3 numbers, which must come from the same GPU model as
Tables 1 and 2 for the comparison to mean anything.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.encoders import ResNetEncoder, SwinEncoder, VGGEncoder  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.runner import provenance  # noqa: E402
from src.seeding import set_seed  # noqa: E402

# Same three rows, same order, as tab:inference_cost in WhenSimpleWins.tex.
CONFIGS = [
    ("VGG U-Net", lambda: UNet(VGGEncoder())),
    ("ResNet U-Net", lambda: UNet(ResNetEncoder())),
    ("Swin-T U-Net (pretrained)", lambda: UNet(SwinEncoder(pretrained=True))),
]
PUBLISHED_M3 = {  # for reference in the report; measured on Apple M3 / MPS
    "VGG U-Net": (25.2, 130.5),
    "ResNet U-Net": (34.2, 144.0),
    "Swin-T U-Net (pretrained)": (46.6, 333.6),
}


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def other_gpu_processes() -> list[str]:
    """Other compute processes on this GPU, via nvidia-smi. Empty list if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:
        return []
    import os
    mine = str(os.getpid())
    return [ln for ln in out.splitlines() if ln.strip() and ln.split(",")[0].strip() != mine]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _peak_bytes(device: str) -> int:
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps":
        try:
            return int(torch.mps.current_allocated_memory())
        except Exception:
            return 0
    return 0


def bench_one(build, device: str, batch_size: int, warmup: int, trials: int) -> dict:
    set_seed(0)  # only affects the random init of the un-pretrained weights
    model = build().eval().to(device)
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    x = torch.randn(batch_size, 3, 256, 256, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync(device)

        # Peak memory for a single forward, isolated from the warmup allocations.
        _reset_peak(device)
        model(x)
        _sync(device)
        peak = _peak_bytes(device)

        per_iter = []
        _sync(device)
        t_all = time.perf_counter()
        for _ in range(trials):
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            per_iter.append((time.perf_counter() - t0) * 1000.0)
        _sync(device)
        total_ms = (time.perf_counter() - t_all) * 1000.0

    per_iter.sort()
    n = len(per_iter)
    result = {
        "params_total": sum(p.numel() for p in model.parameters()),
        "weights_mb": weights_bytes / 1024 ** 2,
        "peak_memory_mb": peak / 1024 ** 2,
        "latency_ms_mean": total_ms / trials,
        "latency_ms_median": per_iter[n // 2],
        "latency_ms_p05": per_iter[max(0, int(0.05 * n))],
        "latency_ms_p95": per_iter[min(n - 1, int(0.95 * n))],
        "latency_ms_min": per_iter[0],
        "latency_ms_max": per_iter[-1],
    }
    del model, x
    _reset_peak(device)
    return result


def latex_body(rows: list[tuple[str, dict]]) -> str:
    """Body for tab:inference_cost — columns @{}lrrr@{}:
    Model & Params & Latency (ms) & Peak Memory (MB)."""
    out = []
    for name, r in rows:
        out.append(f"{name} & {r['params_total'] / 1e6:.1f}M & "
                   f"{r['latency_ms_median']:.1f} & {r['peak_memory_mb']:.1f} \\\\")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--allow-shared", action="store_true",
                    help="Measure even if another process holds the GPU (marks output non-exclusive)")
    ap.add_argument("--out", default="runs/_bench/table3.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    others = other_gpu_processes() if device == "cuda" else []
    if others and not args.allow_shared:
        print("Refusing to benchmark: another compute process is on this GPU:", file=sys.stderr)
        for line in others:
            print(f"  {line}", file=sys.stderr)
        print("Allocate the GPU exclusively, or pass --allow-shared.", file=sys.stderr)
        return 1

    prov = provenance(device)
    print(f"device={device} gpu={prov['gpu_model']} node={prov['node']} "
          f"batch={args.batch_size} warmup={args.warmup} trials={args.trials}")
    if others:
        print(f"!! non-exclusive: {len(others)} other process(es) on the GPU")

    rows = []
    for name, build in CONFIGS:
        r = bench_one(build, device, args.batch_size, args.warmup, args.trials)
        rows.append((name, r))
        pub = PUBLISHED_M3.get(name)
        ref = f"   (published M3: {pub[0]:.1f} ms / {pub[1]:.1f} MB)" if pub else ""
        print(f"{name:30s} {r['params_total'] / 1e6:5.1f}M  "
              f"{r['latency_ms_median']:6.2f} ms (p05 {r['latency_ms_p05']:.2f} / "
              f"p95 {r['latency_ms_p95']:.2f})  peak {r['peak_memory_mb']:7.1f} MB{ref}")

    payload = {
        "conditions": {
            "batch_size": args.batch_size, "input": "3x256x256", "dtype": "fp32",
            "autocast": False, "mode": "eval/no_grad",
            "warmup_iters": args.warmup, "timed_iters": args.trials,
            "synchronised_per_iteration": True,
            "exclusive_gpu": not others,
            "other_gpu_processes": others,
            "peak_memory_definition": "max_memory_allocated over one forward pass "
                                      "after reset_peak_memory_stats (weights + activations)",
        },
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "results": {name: r for name, r in rows},
        "published_m3_reference": PUBLISHED_M3,
        **prov,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    print("\n" + "=" * 78)
    print("% Table 3 body (tab:inference_cost) — columns: @{}lrrr@{}")
    print("=" * 78)
    print(latex_body(rows))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
