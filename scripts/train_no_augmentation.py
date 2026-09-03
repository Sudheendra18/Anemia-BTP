"""
Ablation Study: Train ResNet-50 and EfficientNet-B0 WITHOUT Data Augmentation.

Runs patient-level stratified k-fold cross-validation on both architectures
with all data augmentations turned off (no horizontal flip, no rotation,
no color jitter/saturation changes, no random erasing) to measure the
quantitative improvement gained from data augmentation.

Usage:
    # Train both ResNet-50 and EfficientNet-B0 without augmentations and compare:
    python scripts/train_no_augmentation.py

    # Train a single architecture:
    python scripts/train_no_augmentation.py --models resnet50
    python scripts/train_no_augmentation.py --models efficientnet_b0

    # Force re-train all folds from scratch:
    python scripts/train_no_augmentation.py --fresh

    # Fast smoke-test (1 epoch, 2 folds):
    python scripts/train_no_augmentation.py --quick-test

    # Prevent updating the main leaderboard (if you want to keep results isolated):
    python scripts/train_no_augmentation.py --no-leaderboard

    # Generate comparison report from completed runs without retraining:
    python scripts/train_no_augmentation.py --compare-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_no_augmentation import main  # noqa: E402

if __name__ == "__main__":
    main()
