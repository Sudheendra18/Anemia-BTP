"""Offline ROI audits and patient-separated evaluation of saved ensemble scores."""
from collections import Counter
import json
from math import isclose
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from capture_roi import extract_capture_roi
from frame_selection import RegionDetector
from capture_prediction import CapturePredictor, DEFAULT_THRESHOLD, SCORED_STATUSES
from capture_provenance import read_prediction_input


def roi_audit(rows):
    """Measure ROI overlap against reference masks aligned to the eye crops."""
    records = []
    for row in rows:
        image = cv2.imread(str(row["eye_image_path"]))
        truth = cv2.imread(str(row["reference_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None or truth.shape != image.shape[:2] or not np.any(truth):
            raise ValueError(f"Eye crop {row['image_id']} needs a nonempty, matching reference mask")
        eye = RegionDetector([0, 0, 1, 1])(image)[0]
        eye.update(crop=image, roi_search_mask=eye["roi_mask"])
        roi = extract_capture_roi(eye)
        predicted = np.zeros(truth.shape, dtype=bool)
        if roi["status"] == "ok":
            x0, y0, x1, y1 = roi["bbox"]
            predicted[y0:y1, x0:x1] = roi["mask"] > 0
        reference = truth > 0
        intersection = int(np.count_nonzero(predicted & reference))
        union = int(np.count_nonzero(predicted | reference))
        reference_area = int(reference.sum())
        predicted_area = int(predicted.sum())
        records.append({
            "image_id": row["image_id"],
            "group": row.get("group", "unspecified"),
            "status": roi["status"],
            "iou": intersection / union,
            "tissue_recall": intersection / reference_area,
            "dice": 2 * intersection / (predicted_area + reference_area),
        })
    if not records:
        raise ValueError("Provide at least one labeled eye crop")
    groups = {}
    for name in sorted({record["group"] for record in records}):
        selected = [record for record in records if record["group"] == name]
        groups[name] = {"count": len(selected)}
        for metric in ("iou", "dice", "tissue_recall"):
            groups[name][metric] = float(np.mean([record[metric] for record in selected]))
    return dict(scope="Segmentation on supplied eye crops; does not measure automatic eye detection",
                groups=groups, images=records)


def replay_capture_predictions(rows, manifest_dir, inference_config):
    """Verify the CSV and rerun saved predictions through the capture predictor."""
    predictor = CapturePredictor(inference_config)
    records = []
    for row in rows:
        record = dict(image_id=row["image_id"], status="unscored")
        records.append(record)
        status = row["prediction_status"]
        if status not in SCORED_STATUSES:
            record["reason"] = status or "not_run"
            continue
        try:
            folder = (manifest_dir / row["capture_dir"]).resolve()
            if not row["capture_dir"] or not folder.is_relative_to(manifest_dir.resolve()):
                raise ValueError("Capture folder is outside the manifest directory")
            metadata = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            prediction = metadata["prediction"]
            if (metadata["capture"]["image_id"] != row["image_id"]
                    or prediction["status"] != status
                    or prediction["label"] != row["prediction_label"]
                    or float(prediction["anemia_score"]) != float(row["anemia_score"])):
                raise ValueError("CSV prediction does not match capture metadata")
            path, image, _ = read_prediction_input(folder, prediction)
            if not row["prediction_input_path"] or (manifest_dir / row["prediction_input_path"]).resolve() != path.resolve():
                raise ValueError("CSV prediction input path does not match the verified saved image")
            predictor.threshold = prediction.get("threshold", DEFAULT_THRESHOLD)
            # The saved format is checked against the one the configured predictor actually used.
            result = predictor.predict_roi({"roi": image, "roi_masked": image})
            if result["status"] not in SCORED_STATUSES:
                raise ValueError(result["reason"])
            expected_checkpoints = [item["sha256"] for item in prediction["checkpoints"]]
            actual_checkpoints = [item["sha256"] for item in result["checkpoints"]]
            if result["input_format"] != prediction["input_format"] or expected_checkpoints != actual_checkpoints:
                raise ValueError("Configured checkpoints or input format differ from the saved prediction")
            if (result["label"] != prediction["label"]
                    or not isclose(result["anemia_score"], prediction["anemia_score"], rel_tol=0, abs_tol=1e-6)):
                raise ValueError("Replayed prediction differs from the saved label or score")
            if "fold_scores" in prediction:
                expected_folds = np.asarray(prediction["fold_scores"], dtype=float)
                actual_folds = np.asarray(result["fold_scores"], dtype=float)
                if (expected_folds.shape != actual_folds.shape
                        or not np.allclose(expected_folds, actual_folds, rtol=0, atol=1e-6)):
                    raise ValueError("Replayed fold scores differ from the saved fold scores")
            if status != "review_only" or "folds_disagree" in prediction:
                expected_disagreement = prediction.get("folds_disagree", status == "uncertain")
                if (result["status"] == "uncertain") != expected_disagreement:
                    raise ValueError("Replayed fold disagreement differs from the saved result")
            record["status"] = "matched"
        except (OSError, ValueError, TypeError, KeyError) as error:
            record.update(status="failed", reason=str(error))
    return dict(
        scope="Saved captures only; discarded attempts are outside the CSV",
        saved_captures=len(rows),
        extracted_rois=sum(row["roi_status"] == "ok" for row in rows),
        passed_quality=sum(row["crop_ok"].lower() in ("true", "1") for row in rows),
        replay_counts=dict(Counter(record["status"] for record in records)),
        captures=records,
    )


def ensemble_report(rows, training_patient_ids, *, include_unscored=False):
    """Fit calibration only on calibration patients; report untouched test patients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
    records = []
    attempts = []
    provenance = None
    raw_threshold = None
    seen_paths = set()
    for row in rows:
        patient = str(row["patient_id"]).strip()
        if not patient or patient in training_patient_ids:
            raise ValueError("Evaluation patients must have stable IDs absent from all training data")
        if row["split"] not in ("calibration", "test") or str(row["label"]) not in ("0", "1"):
            raise ValueError("Use split calibration/test and confirmed label 0/1")
        attempt = dict(patient_id=patient, split=row["split"], label=int(row["label"]),
                       outcome="not_saved", quality_passed=False)
        attempts.append(attempt)
        if include_unscored and not row["metrics_path"]:
            continue
        path = Path(row["metrics_path"]).resolve()
        if path in seen_paths:
            raise ValueError("Duplicate capture in evaluation")
        seen_paths.add(path)
        capture = json.loads(path.read_text(encoding="utf-8"))
        saved = capture.get("capture", {})
        if patient != str(saved.get("patient_id", "")).strip():
            raise ValueError("Evaluation patient ID does not match the saved capture patient ID")
        if row.get("image_id") and str(row["image_id"]) != str(saved.get("image_id", "")):
            raise ValueError("Evaluation image ID does not match the saved capture")
        prediction = capture.get("prediction", {})
        confirmed = capture.get("roi_confirmed") is True
        quality_passed = saved.get("crop_ok") is True and saved.get("roi_status") == "ok"
        scored = prediction.get("status") in ("ok", "uncertain")
        attempt["quality_passed"] = quality_passed and confirmed
        if include_unscored and not (attempt["quality_passed"] and scored):
            attempt["outcome"] = "not_eligible"
            continue
        if not (confirmed and scored):
            raise ValueError("Evaluation requires confirmed ROI and saved ensemble scores")
        if not quality_passed:
            raise ValueError("Evaluation requires a capture that passed quality checks")
        read_prediction_input(path.parent, prediction)
        scores = np.asarray(prediction["fold_scores"], dtype=float)
        if scores.shape != (5,) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Expected five finite fold scores between zero and one")
        hashes = tuple(checkpoint["sha256"] for checkpoint in prediction["checkpoints"])
        if len(hashes) != 5 or len(set(hashes)) != 5 or not all(hashes):
            raise ValueError("Five distinct checkpoint hashes are required")
        identity = (hashes, prediction["input_format"])
        if provenance is not None and identity != provenance:
            raise ValueError("Do not mix checkpoints or model input formats")
        provenance = identity
        cutoff = float(prediction.get("threshold", DEFAULT_THRESHOLD))
        if not 0 < cutoff < 1:
            raise ValueError("Saved prediction threshold must be between 0 and 1")
        if raw_threshold is not None and cutoff != raw_threshold:
            raise ValueError("Do not mix predictions saved with different thresholds")
        raw_threshold = cutoff
        attempt["outcome"] = "scored"
        records.append({
            "patient_id": patient,
            "split": row["split"],
            "label": int(row["label"]),
            "score": float(scores.mean()),
        })
    if not records:
        raise ValueError("Provide separate calibration and test patients")
    attempts_df = pd.DataFrame(attempts)
    patient_assignments = attempts_df.groupby("patient_id")[["split", "label"]].nunique()
    if patient_assignments.gt(1).any().any():
        raise ValueError("A patient cannot cross splits or have conflicting labels")
    # Average captures per patient before computing metrics.
    df = pd.DataFrame(records).groupby(["patient_id", "split", "label"], as_index=False)["score"].mean()
    calibration, test = [df[df.split == split] for split in ("calibration", "test")]
    for subset in (calibration, test):
        if set(subset.label) != {0, 1} or subset.groupby("label").size().min() < 2:
            raise ValueError("Each split requires at least two different patients per class; larger cohorts are needed for reliable estimates")

    def logits(values):
        p = np.clip(np.asarray(values), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    model = LogisticRegression(random_state=42).fit(logits(calibration.score), calibration.label)
    cal_scores = model.predict_proba(logits(calibration.score))[:, 1]
    thresholds = np.unique(np.r_[0.5, cal_scores])
    threshold = float(max(
        thresholds,
        key=lambda cutoff: (
            balanced_accuracy_score(calibration.label, cal_scores >= cutoff),
            -abs(cutoff - 0.5),
        ),
    ))
    probabilities = model.predict_proba(logits(test.score))[:, 1]

    def metrics(scores, cutoff):
        predicted = scores >= cutoff
        return {
            "balanced_accuracy": float(balanced_accuracy_score(test.label, predicted)),
            "roc_auc": float(roc_auc_score(test.label, scores)),
            "brier": float(brier_score_loss(test.label, scores)),
            "confusion_matrix": confusion_matrix(test.label, predicted, labels=[0, 1]).tolist(),
        }

    coverage = {}
    for split, subset in attempts_df.groupby("split"):
        scored = subset[subset.outcome == "scored"]
        coverage[split] = dict(
            attempted_captures=len(subset), scored_captures=len(scored), scored_fraction=len(scored) / len(subset),
            passed_quality=int(subset.quality_passed.sum()), quality_pass_fraction=float(subset.quality_passed.mean()),
            patients_enrolled=subset.patient_id.nunique(), patients_scored=scored.patient_id.nunique(),
            outcomes=dict(Counter(subset.outcome)),
        )

    return dict(scope="Patient-mean ensemble score; exploratory evaluation, not deployment approval",
                calibration_patients=len(calibration), test_patients=len(test),
                capture_coverage=coverage,
                raw_threshold=raw_threshold, raw_test=metrics(test.score.to_numpy(), raw_threshold),
                calibrated_test=metrics(probabilities, threshold),
                calibration=dict(coefficient=float(model.coef_[0,0]), intercept=float(model.intercept_[0]), threshold=threshold),
                checkpoint_hashes=list(provenance[0]), input_format=provenance[1],
                note="Settings are not installed into capture automatically. Stable patient identity across datasets is required.")
