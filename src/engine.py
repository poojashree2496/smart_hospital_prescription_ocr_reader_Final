from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import get_head


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, scheduler, device, scaler, use_amp: bool, head_only: bool, grad_clip: float = 1.0) -> tuple[float, float]:
    """One epoch of training. Returns (mean loss, accuracy).

    head_only=True keeps the frozen backbone in eval mode so its BatchNorm running
    statistics are NOT overwritten by small, augmented batches (a bug in the old code).
    """
    if head_only:
        model.eval()
        get_head(model).train()
    else:
        model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        n = targets.size(0)
        total_loss += loss.item() * n
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += n
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, device, topk: int = 5, keep_predictions: bool = False) -> dict:
    """Loss (plain cross-entropy), top-1 and top-k accuracy, optionally per-sample predictions."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, top1, top3_hits, top5_hits, total = 0.0, 0, 0, 0, 0
    predictions: list[dict] = []
    for images, targets, names, ids in loader:
        logits = model(images.to(device)).float()
        targets_dev = targets.to(device)
        total_loss += ce(logits, targets_dev).item()
        probs = torch.softmax(logits, dim=1)
        k = min(max(5, topk), probs.size(1))
        top_values, top_indices = torch.topk(probs, k=k, dim=1)
        top1 += int((top_indices[:, 0] == targets_dev).sum().item())
        top3_hits += int((top_indices[:, :min(3, k)] == targets_dev.unsqueeze(1)).any(dim=1).sum().item())
        top5_hits += int((top_indices[:, :min(5, k)] == targets_dev.unsqueeze(1)).any(dim=1).sum().item())
        total += targets.size(0)
        if keep_predictions:
            for i in range(targets.size(0)):
                predictions.append({
                    "id": ids[i],
                    "expected_index": int(targets[i]),
                    "expected": names[i],
                    "top_indices": top_indices[i].cpu().tolist(),
                    "top_probs": top_values[i].cpu().tolist(),
                })
    result = {
        "loss": total_loss / max(1, total),
        "top1": top1 / max(1, total),
        "top3": top3_hits / max(1, total),
        "top5": top5_hits / max(1, total),
        "samples": total,
    }
    if keep_predictions:
        result["predictions"] = predictions
    return result
