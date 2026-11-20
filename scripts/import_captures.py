"""Join reviewed captures to explicit clinical labels for the existing trainer."""
from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config, read_metadata_csv, resolve_path


def join_labels(captures, labels, label_map):
    required = {"image_id", "patient_id", "label_text"}
    if not required.issubset(labels.columns):
        raise ValueError("Labels CSV needs image_id, patient_id, label_text columns")
    conflicts = (set(labels.columns) & set(captures.columns)) - {"image_id", "patient_id"}
    if conflicts:
        raise ValueError(f"Labels cannot override capture fields: {sorted(conflicts)}")
    if labels.empty:
        raise ValueError("No labels supplied")
    for column in required:
        if labels[column].isna().any() or labels[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Labels contain empty {column}")
    if labels.image_id.duplicated().any() or captures.image_id.duplicated().any():
        raise ValueError("Duplicate image_id in labels or captures")
    if not labels.label_text.isin(label_map).all():
        raise ValueError(f"label_text must be one of {list(label_map)}")
    missing = set(labels.image_id) - set(captures.image_id)
    if missing:
        raise ValueError(f"Labels refer to unknown captures: {sorted(missing)}")
    joined = captures.merge(labels, on="image_id", how="inner", validate="one_to_one",
                            suffixes=("_capture", ""))
    stored = joined.patient_id_capture.fillna("").astype(str)
    if ((stored != "") & (stored != joined.patient_id)).any():
        raise ValueError("Patient ID conflicts with the ID recorded at capture")
    good = joined.crop_ok.astype(str).str.lower().isin(["true", "1"])
    if not (good & joined.roi_status.eq("ok")).all():
        raise ValueError("Labelled captures include failed quality gates or fallback/no-eye ROIs; review and retake")
    for column in ("image_path", "roi_image_path"):
        if not joined[column].map(lambda p: Path(str(p)).is_file()).all():
            raise ValueError(f"Missing files in {column}")
    joined["label"] = joined.label_text.map(label_map).astype(int)
    return joined.drop(columns=["patient_id_capture"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", default="outputs/captures/dataset.csv")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="outputs/metadata/captures_with_roi.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    try:
        manifest = resolve_path(args.captures)
        captures = read_metadata_csv(manifest)
        if "capture_dir" in captures and "image_path" not in captures:
            captures["image_path"] = captures["capture_dir"].map(
                lambda folder: (manifest.parent / str(folder) / "frame.png").as_posix()
            )
            for column in ("roi_image_path", "prediction_input_path"):
                if column in captures:
                    captures[column] = captures[column].map(
                        lambda path: (manifest.parent / str(path)).as_posix() if pd.notna(path) and path else path
                    )
        labels = pd.read_csv(resolve_path(args.labels), dtype=str, keep_default_na=False)
        result = join_labels(captures, labels, load_config(args.config)["dataset"]["label_map"])
        output = resolve_path(args.output)
        if output.resolve() in {resolve_path(args.captures).resolve(), resolve_path(args.labels).resolve()}:
            raise ValueError("Output must differ from the input CSVs")
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    except (ValueError, OSError, KeyError) as e:
        parser.exit(1, f"Import failed: {e}\n")
    print(f"Wrote {len(result)} labelled captures to {output}")


if __name__ == "__main__":
    main()
