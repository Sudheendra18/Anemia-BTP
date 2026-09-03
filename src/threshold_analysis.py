"""
Threshold analysis — finds a better decision cutoff than the default 0.5,
using models you've ALREADY trained. No retraining involved: this loads each
fold's best_model.pt, runs a single inference pass over that fold's held-out
validation images, and pools the predictions across all folds.

Why pooling across folds is valid: in k-fold CV, each image is in exactly one
fold's validation set. Concatenating every fold's validation predictions
gives you one "out-of-fold" (OOF) prediction per image across the whole
dataset — every image scored by a model that never saw it during training.
That's the right set to tune a threshold against.

Why this matters here specifically: the training harness's reported metrics
use the implicit 0.5 cutoff (argmax on a 2-class softmax). Class-weighting
the loss function encourages the model to output higher probabilities for
the minority class, but it does NOT change where the 0.5 decision boundary
sits — those are two different levers. If sensitivity is high and
specificity is low (as it was here), a big part of that gap is closeable
just by moving the cutoff, with no architecture or data changes at all.

Usage:
    python scripts/analyze_thresholds.py
    python scripts/analyze_thresholds.py --sensitivity-floor 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ConjunctivaDataset, build_transforms          # noqa: E402
from model import build_model                                      # noqa: E402
from train import load_metadata_and_splits, get_device             # noqa: E402
from metrics import (                                              # noqa: E402
    sweep_thresholds,
    recommend_thresholds,
    plot_threshold_sweep,
    apply_threshold,
    plot_confusion_matrix,
)
from utils import load_config, resolve_path, setup_logging, ensure_parent_dir  # noqa: E402


@torch.no_grad()
def predict_fold(fold_idx: int, val_df: pd.DataFrame, image_col: str, config: dict, device, logger) -> pd.DataFrame | None:
    train_cfg = config["training"]
    fold_dir = resolve_path(train_cfg["output_dir"]) / f"fold_{fold_idx}"
    checkpoint_path = fold_dir / "best_model.pt"

    if not checkpoint_path.exists():
        logger.warning(f"[fold {fold_idx}] No best_model.pt found at {checkpoint_path} — skipping this fold. Was it trained yet?")
        return None

    val_tf = build_transforms(train_cfg["image_size"], train=False)
    val_ds = ConjunctivaDataset(val_df, image_col, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

    # pretrained=False: we're about to overwrite every weight with the
    # checkpoint anyway, no point downloading ImageNet weights first.
    model = build_model(train_cfg["architecture"], num_classes=train_cfg["num_classes"], pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_prob = []
    for images, _labels in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_prob.extend(probs.cpu().numpy().tolist())

    out = val_df[["patient_id", "image_id", "label", "label_text"]].copy().reset_index(drop=True)
    out["fold"] = fold_idx
    out["y_prob"] = all_prob
    logger.info(f"[fold {fold_idx}] Scored {len(out)} validation images using {checkpoint_path.name}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Find a better decision threshold using already-trained fold checkpoints — no retraining.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--sensitivity-floor", type=float, default=0.90,
        help="For the 'max_specificity_at_sensitivity_floor' recommendation: the minimum sensitivity to hold while maximizing specificity. Default 0.90.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(log_file="outputs/logs/analyze_thresholds.log")
    device = get_device(logger)

    df, image_col, splits = load_metadata_and_splits(config, logger)

    fold_predictions = []
    for fold_idx, (_train_idx, val_idx) in enumerate(splits, start=1):
        val_df = df.iloc[val_idx]
        preds = predict_fold(fold_idx, val_df, image_col, config, device, logger)
        if preds is not None:
            fold_predictions.append(preds)

    if not fold_predictions:
        logger.error(
            "No fold checkpoints found under outputs/training/. Run scripts/train_baseline.py "
            "first — this script only analyzes models that already exist, it doesn't train any."
        )
        return

    oof_df = pd.concat(fold_predictions, ignore_index=True)
    n_expected = len(df)
    if len(oof_df) < n_expected:
        logger.warning(
            f"Only {len(oof_df)}/{n_expected} images have out-of-fold predictions "
            f"(some folds' checkpoints are missing). Recommendations below are based on "
            f"the {len(oof_df)} that are available."
        )

    output_dir = resolve_path(config["training"]["output_dir"])
    oof_csv = ensure_parent_dir(output_dir / "oof_predictions.csv")
    oof_df.to_csv(oof_csv, index=False)
    logger.info(f"Wrote out-of-fold predictions -> {oof_csv}")

    y_true = oof_df["label"].to_numpy()
    y_prob = oof_df["y_prob"].to_numpy()

    sweep_df = sweep_thresholds(y_true, y_prob)
    sweep_csv = ensure_parent_dir(output_dir / "threshold_sweep.csv")
    sweep_df.to_csv(sweep_csv, index=False)
    logger.info(f"Wrote threshold sweep table -> {sweep_csv}")

    recs = recommend_thresholds(sweep_df, sensitivity_floor=args.sensitivity_floor)

    default_row = sweep_df.iloc[(sweep_df["threshold"] - 0.5).abs().idxmin()]
    logger.info("=" * 70)
    logger.info(f"Default threshold (0.50): sensitivity={default_row['sensitivity']:.3f}  specificity={default_row['specificity']:.3f}  f2={default_row['f2']:.3f}")
    logger.info("-" * 70)
    for name, row in recs.items():
        logger.info(
            f"{name:42s} thresh={row['threshold']:.2f}  "
            f"sens={row['sensitivity']:.3f}  spec={row['specificity']:.3f}  "
            f"f2={row['f2']:.3f}  acc={row['accuracy']:.3f}"
        )
    logger.info("=" * 70)

    recommended = recs["max_specificity_at_sensitivity_floor"]["threshold"]
    plot_path = ensure_parent_dir(output_dir / "threshold_sweep.png")
    plot_threshold_sweep(sweep_df, recommended_threshold=recommended, save_path=plot_path, title="Sensitivity / Specificity / F2 vs Threshold (pooled out-of-fold)")
    logger.info(f"Wrote threshold sweep plot -> {plot_path}")

    # Confusion matrix at the recommended threshold, for a direct visual
    # comparison against the default-threshold confusion matrices already
    # saved per fold.
    y_pred_recommended = apply_threshold(y_prob, recommended)
    cm_path = ensure_parent_dir(output_dir / "confusion_matrix_recommended_threshold.png")
    plot_confusion_matrix(
        y_true, y_pred_recommended, cm_path,
        title=f"Pooled OOF — Confusion Matrix @ threshold={recommended:.2f}",
    )
    logger.info(f"Wrote confusion matrix at recommended threshold -> {cm_path}")

    logger.info(
        f"\nRecommended operating point (sensitivity floor {args.sensitivity_floor:.2f}): "
        f"threshold={recommended:.2f}. To use it, call metrics.apply_threshold(y_prob, {recommended:.2f}) "
        f"wherever you currently use argmax for a real prediction (e.g. any future inference/deployment code)."
    )


if __name__ == "__main__":
    main()
