"""
Standard results module: accuracy, sensitivity, specificity, F1, F2, AUC,
plus confusion-matrix / ROC / precision-recall plots — per the Work
Package A spec.

Convention throughout: label 1 = Anemic (positive class), label 0 =
Non-anemic. Sensitivity/specificity are defined relative to the Anemic class,
which is the clinically meaningful framing for a screening tool.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

CLASS_NAMES = ["Non-anemic", "Anemic"]
PLOT_DPI = 300


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_prob is the predicted probability of the positive (Anemic) class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # a.k.a. recall on Anemic
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # happens if y_true is single-class in this split (e.g. a tiny debug run)
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "f2": float(f2),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_true)),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=PLOT_DPI)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path, title: str = "ROC Curve") -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=PLOT_DPI)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.3f}")
    except ValueError:
        ax.text(0.5, 0.5, "Undefined\n(single class in this split)", ha="center", va="center")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path, title: str = "Precision-Recall Curve") -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=PLOT_DPI)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color="#d62728", lw=2, label=f"AP = {ap:.3f}")
    except ValueError:
        ax.text(0.5, 0.5, "Undefined\n(single class in this split)", ha="center", va="center")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, save_path: str | Path, title: str = "Training History") -> None:
    """history: {'train_loss': [...], 'val_loss': [...], 'val_<primary_metric>': [...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=PLOT_DPI)

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    metric_key = [k for k in history if k.startswith("val_") and k != "val_loss"][0]
    metric_name = metric_key.replace("val_", "").upper()
    axes[1].plot(history[metric_key], color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f"Validation {metric_name}")

    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Turns positive-class probabilities into 0/1 predictions at a custom
    cutoff, instead of the implicit 0.5 cutoff that torch.argmax gives you.
    Anything that makes real (non-0.5) predictions should route through this,
    so there's exactly one place that encodes what the decision rule is."""
    return (np.asarray(y_prob) >= threshold).astype(int)


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Computes the full metric set at each candidate threshold. One row per
    threshold — this is the raw material for picking an operating point
    other than the default 0.5, and for plotting the sensitivity/specificity
    tradeoff."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []
    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        m = compute_metrics(y_true, y_pred, y_prob)
        m["threshold"] = float(t)
        rows.append(m)

    return pd.DataFrame(rows)[["threshold", "accuracy", "sensitivity", "specificity", "f1", "f2", "auc", "tn", "fp", "fn", "tp"]]


def recommend_thresholds(sweep_df: pd.DataFrame, sensitivity_floor: float = 0.90) -> dict:
    """Three different, defensible ways to pick an operating point, since
    "optimal" depends on what you're optimizing for:
      - max_f2: whatever the training harness already optimizes for.
      - max_youden_j: sensitivity + specificity - 1, maximized. The standard
        "best overall balance" criterion in diagnostic-test literature.
      - max_specificity_at_sensitivity_floor: the most clinically motivated
        for a screening tool — hold sensitivity at or above `sensitivity_floor`
        (don't miss more anemic cases than that), and get the best specificity
        available without violating it. Falls back to the highest-sensitivity
        row if no threshold actually reaches the floor.
    """
    df = sweep_df.copy()
    df["youden_j"] = df["sensitivity"] + df["specificity"] - 1

    recommendations = {}

    best_f2_row = df.loc[df["f2"].idxmax()]
    recommendations["max_f2"] = best_f2_row.to_dict()

    best_j_row = df.loc[df["youden_j"].idxmax()]
    recommendations["max_youden_j"] = best_j_row.to_dict()

    eligible = df[df["sensitivity"] >= sensitivity_floor]
    if len(eligible) > 0:
        best_floor_row = eligible.loc[eligible["specificity"].idxmax()]
    else:
        # Nothing hit the floor — best we can do is get as close as possible.
        best_floor_row = df.loc[df["sensitivity"].idxmax()]
    recommendations["max_specificity_at_sensitivity_floor"] = best_floor_row.to_dict()

    return recommendations


def plot_threshold_sweep(sweep_df: pd.DataFrame, recommended_threshold: float, save_path: str | Path, title: str = "Threshold Sweep") -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=PLOT_DPI)
    ax.plot(sweep_df["threshold"], sweep_df["sensitivity"], label="Sensitivity", color="#1f77b4")
    ax.plot(sweep_df["threshold"], sweep_df["specificity"], label="Specificity", color="#d62728")
    ax.plot(sweep_df["threshold"], sweep_df["f2"], label="F2", color="#2ca02c", linestyle="--")
    ax.axvline(recommended_threshold, color="gray", linestyle=":", label=f"Recommended ({recommended_threshold:.2f})")
    ax.axvline(0.5, color="lightgray", linestyle=":", label="Default (0.50)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim([0, 1.05])
    ax.set_title(title)
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def aggregate_fold_metrics(fold_metrics: list[dict]) -> pd.DataFrame:
    """Turns a list of per-fold metric dicts into a results table: one row
    per fold plus a 'mean' and 'std' summary row, for the metrics that make
    sense to average (skips raw confusion-matrix counts)."""
    df = pd.DataFrame(fold_metrics)
    df.insert(0, "fold", range(1, len(df) + 1))

    avg_cols = ["accuracy", "sensitivity", "specificity", "f1", "f2", "auc"]
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for col in avg_cols:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()
    for col in ["tn", "fp", "fn", "tp", "n_samples"]:
        mean_row[col] = df[col].sum() if col == "n_samples" else np.nan
        std_row[col] = np.nan

    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return summary_df
