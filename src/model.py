"""
Model factory: transfer-learning CNNs for binary anemic / non-anemic
classification, per the Work Package A spec (ResNet-50 / EfficientNet).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(architecture: str, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_prefix = "fc"

    elif architecture in ("efficientnet_b0", "efficientnetb0"):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_prefix = "classifier"

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported: 'resnet50', 'efficientnet_b0'."
        )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith(head_prefix):
                param.requires_grad = False

    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params) — useful to sanity-check
    that freeze_backbone actually did what you expect."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
