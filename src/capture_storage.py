"""Save clean capture artifacts and a pipeline-compatible, unlabelled manifest."""
from __future__ import annotations

import csv
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from capture_ui import annotate
from frame_selection import CRITERIA
from capture_roi import choose_capture_roi, extract_capture_roi
from capture_prediction import SCORED_STATUSES
from roi_extraction import _save_debug_composite
from model_contract import file_hash
from capture_provenance import prediction_input_path, read_prediction_input
from image_io import read_image, write_image

LEGACY_FIELDS = ["image_id", "patient_id", "device", "distance", "lighting", "captured_at",
          "image_path", "video_path", "eye_image_path", "roi_image_path", "roi_masked_path",
          "roi_status", "crop_ok", "best_frame_idx", "sharpness", "brightness", "glare",
          "conj_coverage", "conj_score", "crop_method", "review_image_path", "metrics_path"]
FULL_FIELDS = ["serial_id", *LEGACY_FIELDS[:6], "capture_dir", "sharpness_path", "exposure_path",
          "glare_path", *LEGACY_FIELDS[6:]]
PREVIOUS_FIELDS = ["serial_id", "image_id", "patient_id", "device", "distance", "lighting",
          "sharpness_path", "exposure_path", "glare_path", "image_path", "eye_image_path",
          "roi_image_path", "roi_status", "crop_ok", "sharpness", "brightness", "glare"]
PREDICTION_FIELDS = ("prediction_status", "prediction_label", "anemia_score")
CAPTURE_PATH_FIELDS = ("capture_dir", "roi_image_path", "prediction_input_path")
PRIOR_CAPTURE_FIELDS = [*PREVIOUS_FIELDS[:6], "capture_dir", *PREVIOUS_FIELDS[11:], *PREDICTION_FIELDS]
FIELDS = [*PRIOR_CAPTURE_FIELDS, "prediction_input_path"]


def prediction_columns(prediction, folder):
    status = prediction.get("status", "not_run")
    scored = status in SCORED_STATUSES
    return dict(prediction_status=status,
                prediction_label=prediction.get("label", "") if scored else "",
                anemia_score=prediction.get("anemia_score", "") if scored else "",
                prediction_input_path=prediction_input_path(folder, prediction).as_posix() if scored else "")


def _manifest_row(row, output):
    record = {key: row.get(key, "") for key in FIELDS}
    for key in CAPTURE_PATH_FIELDS:
        if record[key]:
            record[key] = (output / record[key]).resolve().relative_to(output.resolve()).as_posix()
    return record


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


def _manifest_text(output, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(_manifest_row(row, output) for row in rows)
    return buffer.getvalue()


def _write_manifest(manifest, rows):
    temporary = manifest.with_name(f".dataset_{uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            f.write(_manifest_text(manifest.parent, rows))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, manifest)
    except PermissionError as e:
        raise PermissionError(f"Cannot update {manifest}. Close it in Excel and retry Save.") from e
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Cleanup must not report failure after the CSV was committed.


def _write_compaction_journal(staging, manifest, rows):
    journal = {
        "before": file_hash(manifest),
        "after": sha256(_manifest_text(manifest.parent, rows).encode("utf-8")).hexdigest(),
    }
    temporary = staging / "journal.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(journal, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(staging / "journal.json")


def _finish_compaction(staging):
    # Retire the stage before deleting files so a partial cleanup cannot look like a rollback.
    retired = staging.with_name(f".done{staging.name}")
    if staging.resolve() != staging or retired.exists() or retired.is_symlink():
        return staging.as_posix()
    try:
        staging.rename(retired)
    except OSError:
        return staging.as_posix()
    try:
        shutil.rmtree(retired)
    except OSError:
        return retired.as_posix()
    return None


def _recover_compaction(staging, manifest):
    if staging.resolve() != staging:
        raise ValueError(f"Unexpected compaction staging path: {staging}")
    journal = json.loads((staging / "journal.json").read_text(encoding="utf-8"))
    current = file_hash(manifest)
    before, after = journal["before"], journal["after"]
    if current == before and (before != after or not (staging / "committed").exists()):
        originals = staging / "original"
        backups = [p for p in originals.rglob("*") if p.is_file() or p.is_symlink()]
        for backup in backups:
            target = manifest.parent / backup.relative_to(originals)
            if (backup.parent.resolve() != backup.parent or target.parent.resolve() != target.parent
                    or len(target.relative_to(manifest.parent).parts) not in (2, 5)
                    or not (target.name == "metrics.json" or target.name.startswith("recording.")
                            or target.name in {f"candidate_{criterion}.png" for criterion in CRITERIA})):
                raise ValueError(f"Unexpected compaction backup path: {backup}")
        for backup in backups:
            backup.replace(manifest.parent / backup.relative_to(originals))
        action = "restored original captures"
    elif current == after:
        action = "kept committed compaction"
    else:
        raise ValueError(f"Capture CSV changed during compaction. Review the preserved files in {staging}")
    result = {"folder": staging.as_posix(), "action": action}
    cleanup = _finish_compaction(staging)
    if cleanup:
        result["cleanup_pending"] = cleanup
    return result


def _load_manifest(manifest, *, recovery=None):
    if recovery is not None:
        for retired in sorted(manifest.parent.glob(".done_compact_*")):
            recovery["needs_review"].append({"folder": retired.as_posix(), "reason": "Compaction stage cleanup pending"})
    for staging in sorted(manifest.parent.glob(".compact_*")):
        if not staging.is_dir():
            continue
        if not (staging / "journal.json").is_file():
            if recovery is not None:
                recovery["needs_review"].append({"folder": staging.as_posix(), "reason": "Unjournaled compaction stage retained for review"})
            continue
        try:
            result = _recover_compaction(staging, manifest.parent / "dataset.csv")
        except (KeyError, TypeError, OSError, ValueError) as error:
            raise ValueError(f"Cannot recover compaction at {staging}: {error}") from error
        if recovery is not None:
            recovery.setdefault("compactions", []).append(result)
    if not manifest.exists():
        _write_manifest(manifest, [])
        return []
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, strict=True)
            header, rows = reader.fieldnames, list(reader)
    except (csv.Error, UnicodeError) as e:
        raise ValueError("dataset.csv is not a valid UTF-8 CSV; repair it before saving") from e
    if header not in (FIELDS, PRIOR_CAPTURE_FIELDS, PREVIOUS_FIELDS, FULL_FIELDS, LEGACY_FIELDS):
        raise ValueError("dataset.csv has an unknown schema; choose a different --out folder")
    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"dataset.csv row {number} has missing or extra columns; repair it before saving")
    image_ids = [row["image_id"] for row in rows]
    if any(not value.strip() for value in image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("dataset.csv contains empty or duplicate image IDs")
    if header == LEGACY_FIELDS:
        for serial, row in enumerate(rows, start=1):
            row["serial_id"] = serial
    serials = [int(r["serial_id"]) for r in rows]
    if any(n < 1 for n in serials) or len(serials) != len(set(serials)):
        raise ValueError("dataset.csv contains invalid or duplicate serial IDs")
    if header == PRIOR_CAPTURE_FIELDS:
        _backfill_prediction_paths(manifest, rows)
    elif header != FIELDS:
        _migrate_layout(manifest, rows, "serial_ids" if header == LEGACY_FIELDS else "cleanup")
    for row in rows:
        for key in CAPTURE_PATH_FIELDS:
            if row[key]:
                row[key] = (manifest.parent / row[key]).resolve().as_posix()
    return rows


def _backfill_prediction_paths(manifest, rows):
    output = manifest.parent.resolve()
    for row in rows:
        row["prediction_input_path"] = ""
        if row["prediction_status"] not in SCORED_STATUSES or not row["capture_dir"]:
            continue
        folder = (output / row["capture_dir"]).resolve()
        if not folder.is_relative_to(output):
            continue
        try:
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            prediction = metrics["prediction"]
            if (metrics["capture"]["image_id"] != row["image_id"]
                    or prediction["status"] != row["prediction_status"]
                    or prediction["label"] != row["prediction_label"]
                    or float(prediction["anemia_score"]) != float(row["anemia_score"])):
                continue
            path, _, _ = read_prediction_input(folder, prediction)
            row["prediction_input_path"] = path.as_posix()
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Older captures may lack verified metadata. Preserve their original result.
    shutil.copy2(manifest, output / f"dataset.before_prediction_input_{uuid4().hex[:8]}.csv")
    _write_manifest(manifest, rows)


def _migrate_layout(manifest, rows, reason):
    output = manifest.parent.resolve()
    moves = []
    for row in rows:
        image_id = safe_component(row["image_id"])
        source = (output / row["image_path"]).resolve().parent
        target = output / image_id
        if target.is_symlink():
            raise ValueError(f"Capture destination is a symbolic link: {target}")
        if (source.name != image_id or not source.is_relative_to(output)
                or len(source.relative_to(output).parts) not in (1, 4)):
            raise ValueError(f"Unexpected capture folder: {source}")
        if source != target and not source.exists() and target.is_dir():
            source = target
        if source != target and target.exists():
            raise ValueError(f"Capture folder already exists: {target}")
        metadata_path = source / "metrics.json"
        original = metadata_path.read_bytes()
        metrics = json.loads(original)
        capture = metrics["capture"]
        if capture["image_id"] != image_id or not (source / "frame.png").is_file():
            raise ValueError(f"Capture identity or frame is missing: {source}")
        masked = row["roi_status"] == "ok" and (source / "roi_masked.png").is_file()
        row.update(capture_dir=target.as_posix(),
                   roi_image_path=(target / "roi_masked.png").as_posix() if masked else "")
        if not masked:
            row["crop_ok"] = False
        for key, value in capture.items():
            if key.endswith("_path") and value:
                capture[key] = (target / Path(value).name).as_posix()
        capture.update(capture_dir=row["capture_dir"], roi_image_path=row["roi_image_path"],
                       roi_masked_path=row["roi_image_path"], serial_id=int(row["serial_id"]),
                       crop_ok=str(row["crop_ok"]).lower() in ("true", "1"))
        prediction = _verified_prediction(metrics.get("prediction", {}), source, capture["crop_ok"])
        columns = prediction_columns(prediction, target)
        row.update(columns)
        capture.update(columns)
        if prediction:
            metrics["prediction"] = prediction
        moves.append((source, target, original, json.dumps(metrics, indent=2).encode("utf-8")))

    shutil.copy2(manifest, output / f"dataset.before_{reason}_{uuid4().hex[:8]}.csv")
    moved = []
    try:
        for source, target, original, updated in moves:
            if source != target:
                source.rename(target)
            moved.append((source, target, original))
            _replace_metrics(target, updated)
        _write_manifest(manifest, rows)
    except Exception:
        for source, target, original in reversed(moved):
            _replace_metrics(target, original)
            if source != target:
                target.rename(source)
        raise
    for source, _, _, _ in moves:
        for parent in source.parents:
            if parent == output:
                break
            try:
                parent.rmdir()
            except OSError:
                break


def _replace_metrics(folder, data):
    temp = folder / f".metrics_{uuid4().hex}.tmp"
    try:
        temp.write_bytes(data)
        temp.replace(folder / "metrics.json")
    finally:
        temp.unlink(missing_ok=True)


def prepare_manifest(output):
    """Create the CSV or upgrade the prior capture index, preserving a backup."""
    output = Path(output).resolve()
    with _manifest_lock(output):
        report = {"recovered": [], "needs_review": []}
        rows = _load_manifest(output / "dataset.csv", recovery=report)
        _recover_locked(output, rows, report)
        if any(report.values()):
            (output / "recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output / "dataset.csv"


def _recover_locked(output, rows, report):
    """Reindex complete published saves; never publish partial staging folders."""
    known = {str(row["image_id"]) for row in rows}
    serials = {int(row["serial_id"]) for row in rows}
    recovered = []
    folders = [*output.glob("CAP_*"), *output.glob("*/*/*/CAP_*")]
    for folder in sorted(folders):
        if not folder.is_dir() or folder.name in known:
            continue
        try:
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                raise ValueError("Capture metadata must be an object")
            row = metrics["capture"]
            if not isinstance(row, dict) or not set(PREVIOUS_FIELDS).issubset(row):
                raise ValueError("Incomplete capture metadata")
            if folder.resolve() != folder or PureWindowsPath(row["image_path"]).name != "frame.png":
                raise ValueError("Unexpected capture folder or frame name")
            safe_component(row["patient_id"])
            serial = int(row["serial_id"])
            if (folder.name != row["image_id"] or serial < 1 or serial in serials
                    or folder.name != f"CAP_{serial:06d}" or not str(row["patient_id"]).strip()):
                raise ValueError("Conflicting or incomplete capture identity")
            hashes = metrics.get("artifact_sha256", {})
            if not isinstance(hashes, dict):
                raise ValueError("Artifact inventory must be an object")
            if not isinstance(metrics.get("prediction", {}), dict):
                raise ValueError("Prediction metadata must be an object")
            required = {"frame.png", "review.png", *(f"candidate_{name}_review.png" for name in CRITERIA)}
            if row.get("eye_image_path"):
                required.add("eye.png")
            if row.get("roi_status") == "ok":
                required.update(["roi.png", "roi_masked.png", "roi_mask.png", "roi_debug.png"])
                if PureWindowsPath(row["roi_image_path"]).name not in ("roi.png", "roi_masked.png"):
                    raise ValueError("Unexpected ROI name in capture metadata")
            if not required.issubset(hashes):
                raise ValueError("No verified artifact inventory; manual review required")
            for name, digest in hashes.items():
                if (Path(name).name != name or not name.endswith(".png")
                        or (folder / name).resolve().parent != folder
                        or file_hash(folder / name) != digest or read_image(folder / name) is None):
                    raise ValueError("Missing or changed saved image")
            for field in ("image_path", "eye_image_path", "roi_image_path", "sharpness_path", "exposure_path", "glare_path"):
                value = row.get(field)
                if value:
                    filename = PureWindowsPath(value).name
                    if filename not in hashes:
                        raise ValueError("Capture paths do not match verified artifacts")
                    row[field] = (folder / filename).as_posix()
            row["capture_dir"] = folder.as_posix()
            prediction = _verified_prediction(metrics.get("prediction", {}), folder, row["crop_ok"] is True)
            row.update(prediction_columns(prediction, folder))
            recovered.append(row)
            known.add(row["image_id"])
            serials.add(serial)
            report["recovered"].append(row["image_id"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["needs_review"].append({"folder": str(folder), "reason": str(error)})
    pending = [*output.glob(".pending_CAP_*"), *output.glob("*/*/*/.pending_CAP_*")]
    for folder in sorted(pending):
        report["needs_review"].append({"folder": str(folder), "reason": "Interrupted staging folder; retained for manual review"})
    if recovered:
        manifest = output / "dataset.csv"
        shutil.copy2(manifest, output / f"dataset.before_recovery_{uuid4().hex[:8]}.csv")
        _write_manifest(manifest, sorted([*rows, *recovered], key=lambda row: int(row["serial_id"])))


def _next_patient_id(rows):
    numbers = []
    for row in rows:
        match = re.fullmatch(r"P?(\d+)", str(row.get("patient_id", "")), re.IGNORECASE)
        if match:
            numbers.append(int(match[1]))
    return f"P{max(numbers, default=0) + 1:06d}"


def _reserved_patients(output, rows):
    """Reserve patient IDs in published or staged captures absent from the CSV."""
    known = {row["image_id"] for row in rows}
    reserved = list(rows)
    for pattern in ("CAP_*/metrics.json", ".pending_CAP_*/metrics.json",
                    "*/*/*/CAP_*/metrics.json", "*/*/*/.pending_CAP_*/metrics.json"):
        for path in output.glob(pattern):
            if path.parent.name in known:
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))["capture"]
                patient = row["patient_id"]
                if not isinstance(patient, str) or not patient:
                    raise ValueError("missing patient ID")
                reserved.append({"patient_id": patient})
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Cannot recover patient numbering from {path}; repair this capture first") from error
    return reserved


def next_patient_id(output):
    """Display-only preview; the actual ID is allocated under the save lock."""
    output = Path(output).resolve()
    with _manifest_lock(output):
        rows = _load_manifest(output / "dataset.csv")
        return _next_patient_id(_reserved_patients(output, rows))


def available_patient_ids(output):
    output = Path(output).resolve()
    with _manifest_lock(output):
        return sorted({row["patient_id"] for row in _load_manifest(output / "dataset.csv")})


def safe_component(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Labels must use 1-80 letters, digits, underscores or hyphens")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Label is a reserved Windows filename")
    return value


def _verified_prediction(prediction, folder, approved):
    status = prediction.get("status")
    if status not in SCORED_STATUSES:
        return prediction
    if not approved and status != "review_only":
        return dict(status="stale", reason="Capture approval does not match the prediction; rerun inference")
    try:
        _, _, digest = read_prediction_input(folder, prediction, require_file_hash=False)
    except ValueError as error:
        return dict(status="stale", reason=str(error))
    return dict(prediction, input_file_sha256=digest)


def save_capture(result, output, device, distance, lighting, patient_id="", roi_cfg=None, patient_mode=""):
    """Save all images and one indexed row with an automatic, persistent serial."""
    for value in (device, distance, lighting):
        safe_component(value)
    if patient_id:
        safe_component(patient_id)
    if "roi_selection" not in result:
        choose_capture_roi(result, roi_cfg)
    output = Path(output).resolve()
    with _manifest_lock(output):
        return _save_locked(result, output, device, distance, lighting, patient_id, roi_cfg, patient_mode)


def _save_locked(result, output, device, distance, lighting, patient_id, roi_cfg, patient_mode=""):
    manifest = output / "dataset.csv"
    rows = _load_manifest(manifest)
    reserved = _reserved_patients(output, rows)
    if patient_mode not in ("", "new", "existing"):
        raise ValueError("Choose New patient, Same patient, or Existing patient")
    if patient_mode == "existing" and patient_id not in {row["patient_id"] for row in rows}:
        raise ValueError("Existing patient ID was not found in this capture folder")
    if patient_mode == "new" and patient_id:
        raise ValueError("New patient IDs are assigned automatically")
    if not patient_id:
        patient_id = _next_patient_id(reserved)
    serial = max((int(row["serial_id"]) for row in rows), default=0) + 1
    stamp = datetime.now(timezone.utc)
    image_id = f"CAP_{serial:06d}"
    final = output / image_id
    # Reserve IDs from captures published before a failed CSV commit.
    while final.exists() or any(output.glob(f"*/*/*/{image_id}")):
        serial += 1
        image_id = f"CAP_{serial:06d}"
        final = output / image_id
    staging = output / (".pending_" + image_id + "_" + uuid4().hex[:8])
    staging.mkdir(parents=True)
    bc = result["candidates"][result["best"]]
    row = dict.fromkeys(FULL_FIELDS, "")
    row.update(serial_id=serial, image_id=image_id, capture_dir=final.as_posix(),
               patient_id=patient_id, device=device, distance=distance,
               lighting=lighting, captured_at=stamp.isoformat(), roi_status="no_eye",
               crop_ok=bool(bc["ok"] and result["eye_found"]), best_frame_idx=bc["idx"],
               sharpness=bc["sharpness"], brightness=bc["brightness"], glare=bc["glare"],
               conj_coverage=bc["coverage"], conj_score=bc["score"])

    def image(field, name, data):
        write_image(staging / name, data)
        row[field] = (final / name).as_posix()

    try:
        image("image_path", "frame.png", bc["frame"])
        image("review_image_path", "review.png", annotate(bc["frame"], bc, "Selected", True))
        for name in CRITERIA:
            c = result["candidates"][name]
            write_image(staging / f"candidate_{name}_review.png", annotate(c["frame"], c, name))
            if name in ("sharpness", "exposure", "glare"):
                row[f"{name}_path"] = (final / f"candidate_{name}_review.png").as_posix()
        eye = bc["eye"]
        if eye is not None:
            image("eye_image_path", "eye.png", eye["crop"])
            if eye.get("roi_search_mask") is not None:
                write_image(staging / "eye_search_mask.png", eye["roi_search_mask"])
            row["crop_method"] = eye["method"]
            roi = result.get("_roi_result")
            if roi is None:
                roi = extract_capture_roi(eye, roi_cfg)
            row["roi_status"] = roi["status"]
            if roi["status"] == "ok":
                write_image(staging / "roi.png", roi["roi"])
                image("roi_masked_path", "roi_masked.png", roi["roi_masked"])
                row["roi_image_path"] = row["roi_masked_path"]
                write_image(staging / "roi_mask.png", roi["mask"])
                _save_debug_composite(roi, staging / "roi_debug.png")
            row["crop_ok"] = bool(row["crop_ok"] and roi["status"] == "ok" and result.get("roi_confirmed", False))
        row["metrics_path"] = (final / "metrics.json").as_posix()
        metrics = {k: v for k, v in result.items() if k not in ("candidates", "_roi_result", "_shortlist")}
        if eye is not None:
            metrics["eye_geometry"] = {key: eye[key] for key in ("box", "oval", "side", "method")}
            metrics["roi_extraction"] = {key: roi[key] for key in ("status", "bbox", "method", "reason")}
            if "polygon_normalized" in roi:
                metrics["roi_extraction"]["polygon_normalized"] = roi["polygon_normalized"]
        metrics["capture"] = row
        capture_status = "approved_for_testing" if row["crop_ok"] else "saved_for_review"
        metrics["capture_status"] = capture_status
        metrics["camera_settings"] = result.get("camera_settings", {"source": "unknown"})
        prediction = _verified_prediction(metrics.get("prediction", {}), staging, row["crop_ok"])
        if prediction:
            metrics["prediction"] = prediction
        row.update(prediction_columns(prediction, final))
        metrics["artifact_sha256"] = {path.name: file_hash(path) for path in staging.glob("*.png")}
        metrics["candidates"] = {
            name: {k: v for k, v in c.items() if k not in ("frame", "eye")}
            for name, c in result["candidates"].items()}
        (staging / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        staging.rename(final)
        try:
            _write_manifest(manifest, [*rows, row])
        except Exception:
            final.rename(staging)
            raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(
        row, capture_status=capture_status, csv_path=manifest.as_posix(),
        next_patient_id=_next_patient_id([*reserved, row]),
        folds_disagree=prediction.get("status") == "uncertain" or bool(prediction.get("folds_disagree")),
    )
