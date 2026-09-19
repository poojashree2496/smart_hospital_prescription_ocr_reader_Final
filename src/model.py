from __future__ import annotations

from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ("resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_small")
PREPROCESS_VERSION = 2  # v1 = old 64x160 / no tight crop; those checkpoints are incompatible


def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[-1].in_features, num_classes))
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture {arch!r}. Choose from {ARCHITECTURES}.")
    return model


def get_head(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in get_head(model).parameters():
        parameter.requires_grad = True


def parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict]:
    head_ids = {id(p) for p in get_head(model).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": backbone_lr},
        {"params": list(get_head(model).parameters()), "lr": head_lr},
    ]


def save_checkpoint(path: str | Path, model: nn.Module, labels: list[str], arch: str, image_size: tuple[int, int], extra: dict | None = None) -> None:
    """image_size is (height, width)."""
    checkpoint = {
        "model_state": model.state_dict(),
        "labels": labels,
        "num_classes": len(labels),
        "architecture": arch,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "preprocess_version": PREPROCESS_VERSION,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    output_dir = Path(path).parent.parent if Path(path).parent.name == "medicine_classifier" else Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_to_idx.json").write_text(
        json.dumps({label: index for index, label in enumerate(labels)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "idx_to_class.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(labels)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Returns (model in eval mode on `device`, checkpoint metadata dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("preprocess_version") != PREPROCESS_VERSION:
        raise RuntimeError(
            f"{path} was produced by the old pipeline (64x160 input, no tight crop) and cannot be "
            "used with the current preprocessing. Re-train with training/train_classifier.py."
        )
    labels = checkpoint["labels"]
    model = build_model(len(labels), arch=checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
