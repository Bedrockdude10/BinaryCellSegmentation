"""
Measure inference latency and peak memory for each encoder at 256x256, batch size 1.
Run from final_project/: python scripts/measure_inference_cost.py
Replace the numbers in Table 3 (tab:inference_cost) with the output.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.encoders import VGGEncoder, ResNetEncoder, SwinEncoder
from src.models.unet import UNet

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONFIGS = [
    ("VGG U-Net",                 lambda: UNet(VGGEncoder())),
    ("ResNet U-Net",              lambda: UNet(ResNetEncoder())),
    ("Swin-T U-Net (pretrained)", lambda: UNet(SwinEncoder(pretrained=True))),
]

def bench(model, n_warmup=10, n_trials=100):
    model.eval().to(DEVICE)
    x = torch.randn(1, 3, 256, 256, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if DEVICE == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            model(x)
        if DEVICE == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) / n_trials * 1000  # ms
        peak_mb = torch.mps.current_allocated_memory() / (1024 ** 2) if DEVICE == "mps" else -1
    return elapsed, peak_mb

for name, build in CONFIGS:
    model = build()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    lat, mem = bench(model)
    print(f"{name:30s}  {n_params:5.1f}M  {lat:5.1f} ms  {mem:6.1f} MB")
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()