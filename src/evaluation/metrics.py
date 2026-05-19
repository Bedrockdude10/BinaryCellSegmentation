# src/evaluation/metrics.py
import torch


def dice(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    p = pred.view(-1).float()
    t = target.view(-1).float()
    inter = (p * t).sum()
    return float((2 * inter + smooth) / (p.sum() + t.sum() + smooth))


def iou(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    p = pred.view(-1).float()
    t = target.view(-1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + smooth) / (union + smooth))