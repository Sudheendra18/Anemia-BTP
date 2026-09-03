"""
Run the full Stage 1 -> Stage 2 pipeline end-to-end from one config file.

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/config.yaml
    python scripts/run_pipeline.py --stage metadata     # only Stage 1
    python scripts/run_pipeline.py --stage roi           # only Stage 2 (needs Stage 1 output to exist)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset_loader import build_metadata          # noqa: E402
from roi_extraction import run_batch                # noqa: E402
from utils import load_config, setup_logging, ensure_parent_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the CP-AnemiC data pipeline end-to-end.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--stage",
        choices=["all", "metadata", "roi"],
        default="all",
        help="Which stage(s) to run. Default: all.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(log_file="outputs/logs/run_pipeline.log")

    if args.stage in ("all", "metadata"):
        logger.info("=" * 60)
        logger.info("STAGE 1: Dataset loader / metadata builder")
        logger.info("=" * 60)
        metadata_df, issues_df = build_metadata(config, logger)

        out_csv = ensure_parent_dir(config["metadata_builder"]["output_csv"])
        metadata_df.to_csv(out_csv, index=False)
        logger.info(f"Wrote metadata CSV -> {out_csv}  ({len(metadata_df)} rows)")

        issues_csv = ensure_parent_dir(config["metadata_builder"]["issues_csv"])
        issues_df.to_csv(issues_csv, index=False)
        logger.info(f"Wrote issues CSV   -> {issues_csv}  ({len(issues_df)} rows)")

    if args.stage in ("all", "roi"):
        logger.info("=" * 60)
        logger.info("STAGE 2: Conjunctiva ROI extraction")
        logger.info("=" * 60)
        roi_df = run_batch(config, logger)

        out_csv = ensure_parent_dir(config["roi_extraction"]["output_csv"])
        roi_df.to_csv(out_csv, index=False)
        logger.info(f"Wrote metadata + ROI paths -> {out_csv}")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
