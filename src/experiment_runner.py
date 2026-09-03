"""
Runs multiple training configurations (architecture / learning-rate
combinations, defined in `experiments.runs` in the config) back to back, and
produces a leaderboard ranking them by `training.primary_metric`.

This is the systematic version of "try full fine-tuning" — instead of one
educated guess, it runs a small, defensible search across the two things
most likely to matter at this dataset size: architecture (ResNet-50 vs the
much lighter EfficientNet-B0, which may generalize better on 710 images) and
learning rate.

Reuses run_kfold_cv() completely unchanged for each experiment — same
crash-resilient per-fold checkpointing, same resume-on-rerun behavior, just
pointed at an isolated output directory per experiment
(outputs/experiments/<name>/). If this whole script gets interrupted,
rerunning it picks up exactly where it left off: experiments that finished
are skipped instantly, the interrupted experiment resumes mid-fold exactly
like a single training run would — nothing extra needed for that, it falls
out of run_kfold_cv's existing behavior for free.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --fresh   # ignore all existing progress, retrain everything
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import run_kfold_cv                                     # noqa: E402
from utils import load_config, resolve_path, setup_logging, ensure_parent_dir  # noqa: E402


def build_experiment_config(base_config: dict, name: str, overrides: dict) -> dict:
    """A full deep copy of the base config with this experiment's training
    overrides applied and its output_dir isolated from every other
    experiment (and from a plain, non-experiment training run)."""
    cfg = copy.deepcopy(base_config)
    cfg["training"].update(overrides)
    cfg["training"]["output_dir"] = str(Path(base_config["experiments"]["output_dir"]) / name)
    return cfg


def run_experiments(config: dict, logger, fresh: bool = False) -> pd.DataFrame:
    exp_cfg = config["experiments"]
    runs = exp_cfg["runs"]
    experiments_root = resolve_path(exp_cfg["output_dir"])
    primary_metric = config["training"]["primary_metric"]

    leaderboard_rows = []
    for i, run in enumerate(runs, start=1):
        name = run["name"]
        overrides = run.get("overrides", {})
        exp_config = build_experiment_config(config, name, overrides)

        exp_dir = resolve_path(exp_config["training"]["output_dir"])
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Saved so analyze_thresholds.py (or anything else) can be pointed
        # straight at this experiment later with --config, without the user
        # having to hand-reconstruct what overrides were used.
        effective_config_path = exp_dir / "effective_config.yaml"
        with open(effective_config_path, "w") as f:
            yaml.safe_dump(exp_config, f, sort_keys=False)

        logger.info("#" * 70)
        logger.info(f"# EXPERIMENT {i}/{len(runs)}: {name}   overrides={overrides}")
        logger.info("#" * 70)

        results_df = run_kfold_cv(exp_config, logger, fresh=fresh)

        mean_row = results_df[results_df["fold"] == "mean"].iloc[0].to_dict()
        std_row = results_df[results_df["fold"] == "std"].iloc[0].to_dict()
        leaderboard_rows.append(
            {
                "experiment": name,
                **{f"{k}_mean": v for k, v in mean_row.items() if k != "fold"},
                **{f"{k}_std": v for k, v in std_row.items() if k != "fold"},
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    sort_col = f"{primary_metric}_mean"
    if sort_col in leaderboard_df.columns:
        leaderboard_df = leaderboard_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    leaderboard_csv = ensure_parent_dir(experiments_root / "leaderboard.csv")
    leaderboard_df.to_csv(leaderboard_csv, index=False)
    logger.info(f"\nWrote leaderboard -> {leaderboard_csv}")
    logger.info("\n" + leaderboard_df.to_string(index=False))

    if len(leaderboard_df) > 0:
        winner = leaderboard_df.iloc[0]["experiment"]
        winner_config_path = experiments_root / winner / "effective_config.yaml"
        logger.info(
            f"\nBest experiment by {primary_metric}: '{winner}'.\n"
            f"To run threshold tuning against it:\n"
            f"  python scripts/analyze_thresholds.py --config {winner_config_path}"
        )

    return leaderboard_df


def main():
    parser = argparse.ArgumentParser(description="Run multiple training configurations back to back and rank them.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the base config YAML (must have an 'experiments:' section).")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore all existing checkpoints/completed folds across every experiment and retrain everything from scratch.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if "experiments" not in config or not config["experiments"].get("runs"):
        raise ValueError(
            "No 'experiments.runs' found in the config. Add an 'experiments:' section — "
            "see docs/TRAINING.md for the format — or use scripts/train_baseline.py for a single run."
        )

    logger = setup_logging(log_file="outputs/logs/run_experiments.log")
    run_experiments(config, logger, fresh=args.fresh)


if __name__ == "__main__":
    main()
