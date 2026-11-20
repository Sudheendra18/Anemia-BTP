"""Train and evaluate CNNs with patient-grouped cross-validation and per-fold checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ConjunctivaDataset, build_transforms          # noqa: E402
from model import build_model, count_trainable_params              # noqa: E402
from metrics import (                                              # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_training_curves,
    aggregate_fold_metrics,
)
from utils import load_config, read_metadata_csv, resolve_path, setup_logging  # noqa: E402
from model_contract import training_identity, input_format, check_run_manifest, file_hash


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(logger) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info(f"Using device: cuda ({name}, {total_mem_gb:.1f} GB)")
        # Fixed input shapes allow cuDNN to benchmark convolution algorithms.
        torch.backends.cudnn.benchmark = True
        return device

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        logger.info("Using device: mps (Apple Silicon GPU)")
        return torch.device("mps")

    logger.info("Using device: cpu (no CUDA or MPS GPU detected)")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp: bool) -> float:
    model.train()
    running_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        n += images.size(0)
    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    running_loss = 0.0
    n = 0
    all_true, all_pred, all_prob = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        n += images.size(0)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]  # P(Anemic)
        preds = torch.argmax(logits, dim=1)

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    avg_loss = running_loss / max(n, 1)
    return avg_loss, np.array(all_true), np.array(all_pred), np.array(all_prob)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)  # Avoid division by zero when a class is absent.
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_single_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, image_col: str, config: dict, device, logger, fresh: bool = False) -> dict:
    train_cfg = config["training"]
    fold_dir = resolve_path(train_cfg["output_dir"]) / f"fold_{fold_idx}"

    if fresh and fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    config_fingerprint = training_identity(config, train_df, val_df, image_col)
    run_manifest = dict(version=1, fingerprint=config_fingerprint,
                        input_format=input_format(pd.concat([train_df, val_df]), image_col))
    check_run_manifest(fold_dir, run_manifest)
    (fold_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    train_tf = build_transforms(train_cfg["image_size"], train=True, augmentation=train_cfg.get("augmentation"))
    val_tf = build_transforms(train_cfg["image_size"], train=False)

    train_ds = ConjunctivaDataset(train_df, image_col, transform=train_tf)
    val_ds = ConjunctivaDataset(val_df, image_col, transform=val_tf)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], drop_last=len(train_ds) > train_cfg["batch_size"],
        pin_memory=pin_memory, persistent_workers=train_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin_memory, persistent_workers=train_cfg["num_workers"] > 0,
    )

    model = build_model(
        train_cfg["architecture"], num_classes=train_cfg["num_classes"],
        pretrained=train_cfg["pretrained"], freeze_backbone=train_cfg["freeze_backbone"],
    ).to(device)

    trainable, total = count_trainable_params(model)
    logger.info(f"[fold {fold_idx}] Model: {train_cfg['architecture']}  trainable params: {trainable:,} / {total:,}")

    if train_cfg["use_class_weights"]:
        class_weights = compute_class_weights(train_df["label"].to_numpy(), train_cfg["num_classes"]).to(device)
        logger.info(f"[fold {fold_idx}] Class weights (Non-anemic, Anemic): {class_weights.tolist()}")
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if train_cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    elif train_cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")

    lr_scheduler_name = train_cfg.get("lr_scheduler", "none")
    if lr_scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["num_epochs"])
    elif lr_scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    elif lr_scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: '{lr_scheduler_name}'. Supported: 'none', 'cosine', 'plateau'.")

    use_amp = device.type == "cuda" and train_cfg.get("mixed_precision", True)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    if device.type == "cuda":
        logger.info(f"[fold {fold_idx}] Mixed precision (AMP): {'on' if use_amp else 'off'}")

    primary_metric = train_cfg["primary_metric"]
    best_metric_value = -np.inf
    best_epoch = None
    best_metrics_dict = None
    patience_counter = 0
    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], f"val_{primary_metric}": []}

    resume_path = fold_dir / "resume_state.pt"
    if resume_path.exists():
        # Resume files contain optimizer and training state. Load only trusted local files.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("config_fingerprint") == config_fingerprint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if use_amp and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            history = checkpoint["history"]
            best_metric_value = checkpoint["best_metric_value"]
            best_epoch = checkpoint["best_epoch"]
            best_metrics_dict = checkpoint["best_metrics_dict"]
            patience_counter = checkpoint["patience_counter"]
            start_epoch = checkpoint["epoch"] + 1
            logger.info(
                f"[fold {fold_idx}] Resuming from a saved checkpoint at epoch {checkpoint['epoch']} "
                f"(best so far: epoch {best_epoch}, {primary_metric}={best_metric_value:.4f}). "
                f"Continuing from epoch {start_epoch}."
            )
        else:
            raise ValueError(f"Resume checkpoint does not match this dataset/settings in {fold_dir}; use a new output or --fresh")

    for epoch in range(start_epoch, train_cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, device, use_amp)
        val_metrics = compute_metrics(y_true, y_pred, y_prob)
        current = val_metrics[primary_metric]
        current_for_compare = -np.inf if np.isnan(current) else current

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history[f"val_{primary_metric}"].append(current)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[fold {fold_idx}] epoch {epoch:>3}/{train_cfg['num_epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_{primary_metric}={current:.4f} lr={current_lr:.2e}"
        )

        if scheduler is not None:
            if lr_scheduler_name == "plateau":
                scheduler.step(current_for_compare)
            else:
                scheduler.step()

        if current_for_compare > best_metric_value:
            best_metric_value = current_for_compare
            best_epoch = epoch
            best_metrics_dict = val_metrics
            patience_counter = 0

            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, fold_dir / "best_model.pt")
            contract = dict(version=1, input_format=run_manifest["input_format"],
                            image_size=train_cfg["image_size"], label_map=config["dataset"]["label_map"],
                            preprocessing="rgb_imagenet_resize_v1", training_fingerprint=config_fingerprint,
                            checkpoint_sha256=file_hash(fold_dir / "best_model.pt"))
            (fold_dir / "checkpoint_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
            plot_confusion_matrix(y_true, y_pred, fold_dir / "confusion_matrix.png", title=f"Fold {fold_idx} — Confusion Matrix")
            plot_roc_curve(y_true, y_prob, fold_dir / "roc_curve.png", title=f"Fold {fold_idx} — ROC Curve")
            plot_pr_curve(y_true, y_prob, fold_dir / "pr_curve.png", title=f"Fold {fold_idx} — Precision-Recall Curve")
            with open(fold_dir / "metrics.json", "w") as f:
                json.dump({"best_epoch": epoch, **val_metrics}, f, indent=2)
        else:
            patience_counter += 1

        torch.save(
            {
                "config_fingerprint": config_fingerprint,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_metric_value": best_metric_value,
                "best_epoch": best_epoch,
                "best_metrics_dict": best_metrics_dict,
                "patience_counter": patience_counter,
            },
            resume_path,
        )

        if patience_counter >= train_cfg["early_stopping_patience"]:
            logger.info(f"[fold {fold_idx}] Early stopping at epoch {epoch} (no {primary_metric} improvement for {patience_counter} epochs)")
            break

    plot_training_curves(history, fold_dir / "training_history.png", title=f"Fold {fold_idx} — Training History")
    logger.info(f"[fold {fold_idx}] Best epoch: {best_epoch}  {primary_metric}={best_metric_value:.4f}")

    if best_epoch is not None:
        best_idx = best_epoch - 1
        train_loss_at_best = history["train_loss"][best_idx]
        val_loss_at_best = history["val_loss"][best_idx]
        gap = val_loss_at_best - train_loss_at_best
        if gap > 0.15:
            logger.warning(
                f"[fold {fold_idx}] Train/val loss gap at best epoch: {train_loss_at_best:.4f} vs "
                f"{val_loss_at_best:.4f} (gap {gap:.4f}) — some overfitting. Consider more "
                f"augmentation, higher weight_decay, or freeze_backbone if this is consistent across folds."
            )

    resume_path.unlink(missing_ok=True)

    return best_metrics_dict


def load_metadata_and_splits(config: dict, logger) -> tuple[pd.DataFrame, str, list]:
    """Load metadata and generate the patient-grouped splits shared by training and analysis."""
    train_cfg = config["training"]

    if train_cfg["image_source"] == "with_roi":
        metadata_csv = resolve_path(config["roi_extraction"]["output_csv"])
        image_col = "roi_image_path"
    elif train_cfg["image_source"] == "raw":
        metadata_csv = resolve_path(config["metadata_builder"]["output_csv"])
        image_col = "image_path"
    else:
        raise ValueError(f"training.image_source must be 'with_roi' or 'raw', got '{train_cfg['image_source']}'")

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {metadata_csv}. Run scripts/run_pipeline.py first "
            f"(Stage 1, and Stage 2 if training.image_source is 'with_roi')."
        )

    df = read_metadata_csv(metadata_csv)
    before = len(df)
    df = df[df[image_col].notna()].reset_index(drop=True)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing '{image_col}' (e.g. unreadable images).")
    if df["patient_id"].str.strip().eq("").any():
        raise ValueError("Metadata contains empty patient IDs; assign them before patient-level splitting")
    logger.info(f"Loaded {len(df)} images from {metadata_csv} (image_col='{image_col}')")
    logger.info(f"Class balance: \n{df['label_text'].value_counts().to_string()}")

    k = train_cfg["k_folds"]
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.get("random_seed", 42))
    splits = list(sgkf.split(df, df["label"], groups=df["patient_id"]))

    return df, image_col, splits


def run_kfold_cv(config: dict, logger, fresh: bool = False) -> pd.DataFrame:
    train_cfg = config["training"]
    set_seed(config.get("random_seed", 42))
    device = get_device(logger)

    df, image_col, splits = load_metadata_and_splits(config, logger)
    k = len(splits)

    fold_metrics = []
    hashes = {}
    # Validate every existing fold before training or overwriting any of them.
    roi_format = input_format(df, image_col)
    if not fresh:
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            expected = dict(version=1, fingerprint=training_identity(config, df.iloc[train_idx], df.iloc[val_idx], image_col, hashes),
                            input_format=roi_format)
            check_run_manifest(resolve_path(train_cfg["output_dir"]) / f"fold_{fold_idx}", expected)
    output = resolve_path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dir = resolve_path(train_cfg["output_dir"]) / f"fold_{fold_idx}"
        existing_metrics_path = fold_dir / "metrics.json"
        resume_state_path = fold_dir / "resume_state.pt"

        # Best-epoch metrics can exist before a fold finishes.
        fold_is_complete = (existing_metrics_path.exists() and not resume_state_path.exists()
                            and (fold_dir / "best_model.pt").is_file())

        if not fresh and fold_is_complete:
            with open(existing_metrics_path) as f:
                saved = json.load(f)
            saved.pop("best_epoch", None)
            logger.info(f"=== Fold {fold_idx}/{k} === already completed (found {existing_metrics_path.name}) — skipping. Use --fresh to retrain anyway.")
            fold_metrics.append(saved)
            continue

        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        overlap = set(train_df["patient_id"]) & set(val_df["patient_id"])
        assert not overlap, f"Patient leakage across fold {fold_idx}: {len(overlap)} patient_id(s) in both train and val"

        logger.info(
            f"=== Fold {fold_idx}/{k} === train={len(train_df)} "
            f"({train_df['label_text'].value_counts().to_dict()})  "
            f"val={len(val_df)} ({val_df['label_text'].value_counts().to_dict()})"
        )

        metrics = run_single_fold(fold_idx, train_df, val_df, image_col, config, device, logger, fresh=fresh)
        fold_metrics.append(metrics)

    results_df = aggregate_fold_metrics(fold_metrics)

    output_dir = resolve_path(train_cfg["output_dir"])
    results_csv = output_dir / "baseline_results_table.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Wrote aggregated results table -> {results_csv}")
    logger.info("\n" + results_df.to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate the CP-AnemiC baseline model with patient-level stratified k-fold CV.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing checkpoints/completed folds under outputs/training and retrain everything from scratch.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(log_file="outputs/logs/train_baseline.log")

    run_kfold_cv(config, logger, fresh=args.fresh)


if __name__ == "__main__":
    main()
