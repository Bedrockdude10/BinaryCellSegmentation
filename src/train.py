# src/train.py
import torch
import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=10):
    model.train()
    total_loss = 0.0
    use_amp = (device != "cpu")
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