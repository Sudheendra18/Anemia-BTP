"""Save clean capture artifacts and a pipeline-compatible, unlabelled manifest."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from capture_ui import annotate
from frame_selection import CRITERIA
from capture_roi import extract_capture_roi
from roi_extraction import _save_debug_composite

LEGACY_FIELDS = ["image_id", "patient_id", "device", "distance", "lighting", "captured_at",
          "image_path", "video_path", "eye_image_path", "roi_image_path", "roi_masked_path",
          "roi_status", "crop_ok", "best_frame_idx", "sharpness", "brightness", "glare",
          "conj_coverage", "conj_score", "crop_method", "review_image_path", "metrics_path"]
FULL_FIELDS = ["serial_id", *LEGACY_FIELDS[:6], "capture_dir", "sharpness_path", "exposure_path",
          "glare_path", *LEGACY_FIELDS[6:]]
FIELDS = ["serial_id", "image_id", "patient_id", "device", "distance", "lighting",
          "sharpness_path", "exposure_path", "glare_path", "image_path", "eye_image_path",
          "roi_image_path", "roi_status", "crop_ok", "sharpness", "brightness", "glare"]


@contextmanager
def _manifest_lock(output):
    """Serialize saves across processes; the OS releases the lock on a crash."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".dataset.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise OSError("Another capture is saving to this folder. Retry Save in a moment.") from e
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_manifest(manifest, rows):
    temporary = manifest.with_name(f".dataset_{uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, manifest)
    except PermissionError as e:
        raise PermissionError(f"Cannot update {manifest}. Close it in Excel and retry Save.") from e
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(manifest):
    if not manifest.exists():
        _write_manifest(manifest, [])
        return []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    if header in (FIELDS, FULL_FIELDS):
        serials = [int(r["serial_id"]) for r in rows]
        if any(n < 1 for n in serials) or len(serials) != len(set(serials)):
            raise ValueError("dataset.csv contains invalid or duplicate serial IDs")
        if header == FIELDS:
            return rows
    elif header != LEGACY_FIELDS:
        raise ValueError("dataset.csv has an unknown schema; choose a different --out folder")
    # Backfill the original export without renaming any captured images or IDs.
    if header == LEGACY_FIELDS:
        for serial, row in enumerate(rows, start=1):
            row["serial_id"] = serial
            folder = Path(row["image_path"]).parent
            for name in ("sharpness", "exposure", "glare"):
                path = folder / f"candidate_{name}.png"
                row[f"{name}_path"] = path.as_posix() if path.is_file() else ""
    reason = "serial_ids" if header == LEGACY_FIELDS else "cleanup"
    backup = manifest.with_name(f"dataset.before_{reason}_{uuid4().hex[:8]}.csv")
    shutil.copy2(manifest, backup)
    _write_manifest(manifest, rows)
    return [{key: row.get(key, "") for key in FIELDS} for row in rows]


def prepare_manifest(output):
    """Create the CSV or upgrade the prior capture index, preserving a backup."""
    output = Path(output).resolve()
    with _manifest_lock(output):
        _load_manifest(output / "dataset.csv")
    return output / "dataset.csv"


def _next_patient_id(rows):
    numbers = []
    for row in rows:
        match = re.fullmatch(r"P?(\d+)", str(row.get("patient_id", "")), re.IGNORECASE)
        if match:
            numbers.append(int(match[1]))
    return f"P{max(numbers, default=0) + 1:06d}"


def next_patient_id(output):
    """Display-only preview; the actual ID is allocated under the save lock."""
    output = Path(output).resolve()
    with _manifest_lock(output):
        return _next_patient_id(_load_manifest(output / "dataset.csv"))


def safe_component(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Labels must use 1-80 letters, digits, underscores or hyphens")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Label is a reserved Windows filename")
    return value


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def save_capture(result, video_path, output, device, distance, lighting, patient_id="", roi_cfg=None):
    """Save all images and one indexed row with an automatic, persistent serial."""
    for value in (device, distance, lighting):
        safe_component(value)
    if patient_id:
        safe_component(patient_id)
    output = Path(output).resolve()
    with _manifest_lock(output):
        return _save_locked(result, video_path, output, device, distance, lighting, patient_id, roi_cfg)


def _save_locked(result, video_path, output, device, distance, lighting, patient_id, roi_cfg):
    manifest = output / "dataset.csv"
    rows = _load_manifest(manifest)
    if not patient_id:
        patient_id = _next_patient_id(rows)
    serial = max((int(row["serial_id"]) for row in rows), default=0) + 1
    stamp = datetime.now(timezone.utc)
    image_id = f"CAP_{serial:06d}"
    base = output / device / distance / lighting
    final = base / image_id
    # If a crash published files before the index commit, never reuse their ID.
    while any(output.glob(f"*/*/*/{image_id}")) or any(row["image_id"] == image_id for row in rows):
        serial += 1
        image_id = f"CAP_{serial:06d}"
        final = base / image_id
    staging = base / (".pending_" + image_id + "_" + uuid4().hex[:8])
    staging.mkdir(parents=True)
    bc = result["candidates"][result["best"]]
    # Keep full diagnostics in metrics.json and the UI response, not in the CSV.
    row = dict.fromkeys(FULL_FIELDS, "")
    row.update(serial_id=serial, image_id=image_id, capture_dir=final.as_posix(),
               patient_id=patient_id, device=device, distance=distance,
               lighting=lighting, captured_at=stamp.isoformat(), roi_status="no_eye",
               crop_ok=bool(bc["ok"] and result["eye_found"]), best_frame_idx=bc["idx"],
               sharpness=bc["sharpness"], brightness=bc["brightness"], glare=bc["glare"],
               conj_coverage=bc["coverage"], conj_score=bc["score"])

    def image(field, name, data):
        _write_image(staging / name, data)
        row[field] = (final / name).as_posix()

    try:
        # Clean pixels are the only input to ROI extraction and downstream ML.
        image("image_path", "frame.png", bc["frame"])
        image("review_image_path", "review.png", annotate(bc["frame"], bc, "Selected", True))
        for name in CRITERIA:
            c = result["candidates"][name]
            _write_image(staging / f"candidate_{name}.png", annotate(c["frame"], c, name))
            if name in ("sharpness", "exposure", "glare"):
                row[f"{name}_path"] = (final / f"candidate_{name}.png").as_posix()
        suffix = Path(video_path).suffix.lower() or ".video"
        shutil.copy2(video_path, staging / ("recording" + suffix))
        row["video_path"] = (final / ("recording" + suffix)).as_posix()
        eye = bc["eye"]
        if eye is not None:
            image("eye_image_path", "eye.png", eye["crop"])
            if eye.get("roi_search_mask") is not None:
                _write_image(staging / "eye_search_mask.png", eye["roi_search_mask"])
            row["crop_method"] = eye["method"]
            roi = result.get("_roi_result")
            if roi is None:
                roi = extract_capture_roi(eye, roi_cfg)
            row["roi_status"] = roi["status"]
            if roi["status"] == "ok":
                image("roi_image_path", "roi.png", roi["roi"])
                image("roi_masked_path", "roi_masked.png", roi["roi_masked"])
                _write_image(staging / "roi_mask.png", roi["mask"])
                _save_debug_composite(roi, staging / "roi_debug.png")
            row["crop_ok"] = bool(row["crop_ok"] and roi["status"] == "ok")
        row["metrics_path"] = (final / "metrics.json").as_posix()
        metrics = {k: v for k, v in result.items() if k not in ("candidates", "_roi_result")}
        if eye is not None:
            metrics["eye_geometry"] = {key: eye[key] for key in ("box", "oval", "side", "method")}
            metrics["roi_extraction"] = {key: roi[key] for key in ("status", "bbox", "method", "reason")}
        metrics["capture"] = row
        metrics["candidates"] = {
            name: {k: v for k, v in c.items() if k not in ("frame", "eye")}
            for name, c in result["candidates"].items()}
        (staging / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        staging.rename(final)
        try:
            _write_manifest(manifest, [*rows, row])
        except Exception:
            # Return only a complete save; failed CSV updates can be retried.
            final.rename(staging)
            raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(row, csv_path=manifest.as_posix(), next_patient_id=_next_patient_id([*rows, row]))
