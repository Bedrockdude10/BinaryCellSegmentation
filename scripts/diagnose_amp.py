#!/usr/bin/env python3
# scripts/diagnose_amp.py
"""Why does the ResNet encoder stop learning on CUDA but not on MPS?

    python scripts/diagnose_amp.py --cache-root /tmp/nucseg_cache

The reviewed code trains under ``torch.autocast(dtype=float16)`` for any
non-CPU device, with no ``GradScaler``. On the Apple M3 / MPS backend the
published runs behaved like fp32 (ResNet loss 0.256 -> 0.201 -> 0.168 -> ...
-> 0.103 over 47 epochs). Re-running the identical config on a V100, where fp16
autocast is fully active, ResNet's loss sits at ~0.25 from epoch 1 and never
moves, landing at Dice 0.667 instead of the published 0.849 — while VGG under
the same settings trains normally to 0.850.

This script trains the ResNet encoder for a few epochs under three regimes and
prints the loss trajectories side by side, plus VGG as a control:

  amp_fp16          exactly what the reviewed code does on CUDA
  amp_fp16_scaler   the same, with the GradScaler fp16 training normally needs
  fp32              autocast off

It changes nothing and writes nothing to runs/. It exists to identify the cause
before deciding which regime the whole sweep should sit in, and to measure what
fp32 costs per epoch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.builders import (  # noqa: E402
    build_loaders, build_loss, build_model, build_optimizer, resolve_device,
)
from src.config import load_config  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.evaluation.metrics import dice, iou  # noqa: E402
from src.seeding import set_seed  # noqa: E402

MODES = ("amp_fp16", "amp_fp16_scaler", "fp32")


def train_epoch(model, loader, criterion, optimizer, device, mode, scaler):
    model.train()
    total, n = 0.0, 0
    use_amp = mode.startswith("amp")
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(images), targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def grad_stats(model):
    """Fraction of gradient elements that are exactly zero — the underflow signal."""
    zeros = total = 0
    for p in model.parameters():
        if p.grad is not None:
            zeros += int((p.grad == 0).sum())
            total += p.grad.numel()
    return zeros / max(total, 1)


def run_one(arch: str, mode: str, epochs: int, cache_root: str | None, overrides: list[str]):
    cfg = load_config("configs/default.yaml", overrides)
    if cache_root:
        cfg["data"] = {**cfg["data"], "cache_root": cache_root}
    cfg["performance"] = {"num_workers": 8, "persistent_workers": False,
                          "pin_memory": True, "prefetch_factor": 4}
    cfg["model"] = {**cfg["model"], "architecture": arch}
    device = resolve_device("auto")

    set_seed(42)
    train_loader, val_loader, _, _ = build_loaders(cfg, train_seed=42, split_seed=42,
                                                   device=device)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler(device) if mode == "amp_fp16_scaler" else None

    print(f"\n{arch} / {mode}")
    losses = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, device, mode, scaler)
        zero_frac = grad_stats(model)
        val = evaluate(model, val_loader, {"dice": dice, "iou": iou}, device=device)
        losses.append(loss)
        print(f"  epoch {ep}  loss {loss:.4f}  val_dice {val['dice']:.4f}  "
              f"zero-grad {zero_frac * 100:5.1f}%  {time.time() - t0:5.1f}s")
    print(f"  loss change over {epochs} epochs: {losses[0] - losses[-1]:+.4f}")
    del model, optimizer, train_loader, val_loader
    if device == "cuda":
        torch.cuda.empty_cache()
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = ap.parse_args()

    print(f"device={resolve_device('auto')} epochs={args.epochs}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    results = {}
    for mode in args.modes:
        results[("resnet", mode)] = run_one("resnet", mode, args.epochs,
                                            args.cache_root, ["configs/ablations/resnet.yaml"])
    # Control: VGG under the reviewed settings, which does train on CUDA.
    results[("vgg", "amp_fp16")] = run_one("vgg", "amp_fp16", args.epochs,
                                           args.cache_root, [])

    print("\n" + "=" * 64)
    print(f"{'arch / mode':30s} {'first':>8s} {'last':>8s} {'change':>9s}")
    for (arch, mode), losses in results.items():
        print(f"{arch + ' / ' + mode:30s} {losses[0]:8.4f} {losses[-1]:8.4f} "
              f"{losses[0] - losses[-1]:+9.4f}")
    print("A regime where the loss barely moves is not training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
