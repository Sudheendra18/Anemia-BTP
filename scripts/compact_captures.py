"""Remove saved recordings and duplicate candidates, preserving review/ROI files."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capture_storage import _load_manifest, _manifest_lock, _verified_prediction, _write_manifest, prediction_columns
from frame_selection import CRITERIA
from utils import resolve_path


def compact(output):
    output = Path(output).resolve()
    removed = 0
    preserved_recordings = []
    with _manifest_lock(output):
        rows = _load_manifest(output / "dataset.csv")
        for row in rows:
            folder = Path(row["capture_dir"]).resolve()
            if not folder.is_relative_to(output) or len(folder.relative_to(output).parts) not in (1, 4):
                raise ValueError(f"Capture folder is outside the expected output layout: {folder}")
            redundant = [folder / f"candidate_{name}.png" for name in CRITERIA
                         if (folder / f"candidate_{name}_review.png").is_file()]
            metrics_path = folder / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
            geometry = metrics.get("eye_geometry") if metrics else None
            can_repair = (isinstance(geometry, dict)
                          and all(key in geometry for key in ("box", "oval", "side", "method"))
                          and (folder / "eye.png").is_file())
            recordings = [p for p in folder.glob("recording.*") if p.is_file()]
            if can_repair:
                redundant += recordings
            else:
                preserved_recordings.extend(p.as_posix() for p in recordings)
            for path in redundant:
                if not path.resolve().is_relative_to(output):
                    raise ValueError(f"Refusing to remove a file outside the capture output: {path}")
            if metrics is not None:
                capture = metrics["capture"]
                masked = folder / "roi_masked.png"
                row["roi_image_path"] = masked.as_posix() if row["roi_status"] == "ok" and masked.is_file() else ""
                if not row["roi_image_path"]:
                    row["crop_ok"] = False
                capture["roi_image_path"] = row["roi_image_path"]
                capture["roi_masked_path"] = row["roi_image_path"]
                capture["crop_ok"] = str(row["crop_ok"]).lower() in ("true", "1")
                capture["roi_status"] = row["roi_status"]
                prediction = _verified_prediction(metrics.get("prediction", {}), folder, capture["crop_ok"])
                if prediction:
                    metrics["prediction"] = prediction
                columns = prediction_columns(prediction, folder)
                row.update(columns)
                capture.update(columns)
                if can_repair:
                    capture["video_path"] = ""
                temporary = folder / f".metrics_{uuid4().hex}.tmp"
                try:
                    temporary.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    temporary.replace(metrics_path)
                finally:
                    temporary.unlink(missing_ok=True)
            for path in redundant:
                if path.exists():
                    path.unlink()
                    removed += 1
        _write_manifest(output / "dataset.csv", rows)
    return dict(captures=len(rows), removed_files=removed, preserved_recordings=preserved_recordings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/captures")
    args = parser.parse_args()
    print(json.dumps(compact(resolve_path(args.out))))
