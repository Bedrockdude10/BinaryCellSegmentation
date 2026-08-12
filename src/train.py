# src/train.py
import torch
import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=10,
                    use_amp=None):
    """One training epoch.

    ``use_amp=None`` keeps the original behaviour (autocast fp16 on any non-CPU
    device). The sweep passes ``use_amp=False``: on CUDA, fp16 autocast without a
    GradScaler zeroes 92-98% of the ResNet encoder's gradients and it never
    trains (Dice 0.667 vs 0.849 published). The published runs were on the MPS
    backend where this autocast was inert, so fp32 is the regime those numbers
    actually came from. See scripts/diagnose_amp.py for the measurement.
    """
    model.train()
    total_loss = 0.0
    if use_amp is None:
        use_amp = (device != "cpu")
    if use_amp and device == "cuda":
        logger.warning(
            "AMP fp16 is enabled on CUDA with no GradScaler. This silently "
            "prevents the ResNet encoder from training — see scripts/diagnose_amp.py."
        )
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            logger.info(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / max(len(loader), 1)