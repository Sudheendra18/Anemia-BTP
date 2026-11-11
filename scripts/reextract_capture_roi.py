"""Regenerate a saved capture's ROI, mask and QA preview, preserving a backup."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capture_roi import extract_capture_roi
from capture_storage import _load_manifest, _manifest_lock, _write_image, _write_manifest
from frame_selection import CaptureConfig, select_best_frame
from roi_extraction import _save_debug_composite
from utils import load_config, resolve_path


def repair_capture(folder, manifest, config):
    folder, manifest = Path(folder).resolve(), Path(manifest).resolve()
    with _manifest_lock(manifest.parent):
        rows = _load_manifest(manifest)
        matches = [r for r in rows if Path(r["image_path"]).resolve().parent == folder]
        if len(matches) != 1:
            raise ValueError("Expected exactly one CSV entry for this capture folder")
        row = matches[0]
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        image = cv2.imread(str(folder / "eye.png"))
        if image is None:
            raise ValueError("This capture has no readable saved eye crop")
        if "eye_geometry" in metrics:
            eye = dict(metrics["eye_geometry"], crop=image)
            search_path = folder / "eye_search_mask.png"
            if search_path.exists():
                search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
                if search is None or search.shape != image.shape[:2]:
                    raise ValueError("Saved eye search mask does not match the crop")
                eye["roi_search_mask"] = search
        else:
            # Older exports omitted geometry. Replay the original tracker and
            # require an exact crop match rather than guessing anatomical bounds.
            settings = metrics.get("quality_config", {})
            cfg = CaptureConfig(**{f.name: settings[f.name] for f in fields(CaptureConfig) if f.name in settings})
            selection = select_best_frame(metrics["capture"]["video_path"], cfg)
            best = selection["candidates"][selection["best"]]
            eye = best["eye"]
            if (eye is None or best["idx"] != int(metrics["capture"]["best_frame_idx"])
                    or not np.array_equal(eye["crop"], image)):
                raise ValueError("Cannot reproduce the saved eye geometry exactly; recapture using the updated app")
        roi = extract_capture_roi(eye, config.get("roi_extraction"))
        row["roi_status"] = roi["status"]
        row["roi_image_path"] = (folder / "roi.png").as_posix() if roi["status"] == "ok" else ""
        if roi["status"] != "ok":
            row["crop_ok"] = False
        metrics["capture"].update(roi_status=row["roi_status"], roi_image_path=row["roi_image_path"],
                                    crop_ok=str(row["crop_ok"]).lower() in ("true", "1"),
                                    roi_masked_path=(folder / "roi_masked.png").as_posix() if roi["status"] == "ok" else "")
        metrics["eye_geometry"] = {k: eye[k] for k in ("box", "oval", "side", "method")}
        metrics["roi_extraction"] = {k: roi[k] for k in ("status", "bbox", "method", "reason")}
        names = ("roi.png", "roi_masked.png", "roi_mask.png", "roi_debug.png", "eye_search_mask.png", "metrics.json")
        backup = folder / ("roi_backup_" + uuid4().hex[:8])
        backup.mkdir()
        for name in names:
            if (folder / name).exists():
                shutil.copy2(folder / name, backup / name)
        shutil.copy2(manifest, backup / "dataset.csv")
        with tempfile.TemporaryDirectory(prefix=".roi_", dir=folder) as temp:
            staged = Path(temp)
            if eye.get("roi_search_mask") is not None:
                _write_image(staged / "eye_search_mask.png", eye["roi_search_mask"])
            if roi["status"] == "ok":
                for name, key in (("roi.png", "roi"), ("roi_masked.png", "roi_masked"), ("roi_mask.png", "mask")):
                    _write_image(staged / name, roi[key])
                _save_debug_composite(roi, staged / "roi_debug.png")
            (staged / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            try:
                for name in names:
                    if (staged / name).exists():
                        (staged / name).replace(folder / name)
                    else:
                        (folder / name).unlink(missing_ok=True)
                _write_manifest(manifest, rows)
            except Exception:
                for name in names:
                    if (backup / name).exists():
                        shutil.copy2(backup / name, folder / name)
                    else:
                        (folder / name).unlink(missing_ok=True)
                raise
        return dict(status=roi["status"], bbox=roi["bbox"], backup=str(backup),
                    preview=str(folder / "roi_debug.png") if roi["status"] == "ok" else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--manifest", default="outputs/captures/dataset.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    try:
        result = repair_capture(resolve_path(args.capture_dir), resolve_path(args.manifest), load_config(args.config))
    except (ValueError, OSError, RuntimeError, KeyError) as e:
        parser.exit(1, f"ROI regeneration failed: {e}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
