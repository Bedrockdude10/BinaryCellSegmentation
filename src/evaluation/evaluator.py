# src/evaluation/evaluator.py
import torch


@torch.no_grad()
def evaluate(model, loader, metric_fns: dict, threshold=0.5, device="cpu",
             patches_per_image: list = None):
    """Run model on loader, compute metrics.

    If patches_per_image is provided, computes per-image Dice and averages —
    matching MedT's evaluation protocol. Otherwise computes global Dice across
    all patches pooled together.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        preds = (torch.sigmoid(model(images)) > threshold).float().cpu()
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)   # (N_patches, 1, H, W)
    all_targets = torch.cat(all_targets, dim=0)

    if patches_per_image is None:
        return {name: fn(all_preds, all_targets) for name, fn in metric_fns.items()}

    # Per-image Dice: reconstruct each image from its patches, compute metric
    per_image_metrics = {name: [] for name in metric_fns}
    start = 0
    for count in patches_per_image:
        img_preds = all_preds[start : start + count]
        img_targets = all_targets[start : start + count]
        for name, fn in metric_fns.items():
            per_image_metrics[name].append(fn(img_preds, img_targets))
        start += count

    return {name: sum(vals) / len(vals) for name, vals in per_image_metrics.items()}