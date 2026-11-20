"""Evaluate labeled ROI masks or saved ensemble predictions without changing captures."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capture_validation import roi_audit, ensemble_report, replay_capture_predictions
from utils import load_config


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["roi", "ensemble", "replay"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-metadata", help="All training patients; required for ensemble validation")
    parser.add_argument("--include-unscored", action="store_true",
                        help="Include failed and unsaved attempts in ensemble capture coverage")
    parser.add_argument("--config", default="configs/config.yaml", help="Capture config for replay")
    parser.add_argument("--output", required=True, help="New JSON report path (existing files are never overwritten)")
    args = parser.parse_args()
    try:
        rows = read_rows(args.manifest)
        base = Path(args.manifest).resolve().parent
        for row in rows:
            for key in ("eye_image_path", "reference_mask_path", "metrics_path"):
                if row.get(key):
                    row[key] = str((base / row[key]).resolve())
        if args.mode == "roi":
            report = roi_audit(rows)
        elif args.mode == "replay":
            options = load_config(args.config).get("capture", {}).get("inference")
            report = replay_capture_predictions(rows, base, options)
        else:
            if not args.training_metadata:
                raise ValueError("--training-metadata is required to check patient separation")
            patients = {r["patient_id"].strip() for r in read_rows(args.training_metadata)}
            if not patients or "" in patients:
                raise ValueError("Training metadata must contain nonempty patient IDs")
            report = ensemble_report(rows, patients, include_unscored=args.include_unscored)
        with Path(args.output).open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(f"Saved validation report: {Path(args.output).resolve()}")
        if args.mode == "replay" and report["replay_counts"].get("failed", 0):
            parser.exit(1, "Some saved predictions could not be reproduced; see the report.\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Validation failed: {error}\n")


if __name__ == "__main__":
    main()
