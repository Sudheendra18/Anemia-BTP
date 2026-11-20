"""Capture provenance, patient selection, and interrupted-save recovery checks."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from capture import camera_metadata, validate_details
from capture_prediction import CapturePredictor
from capture_provenance import pixel_hash
from capture_roi import choose_capture_roi
from capture_storage import save_capture, prepare_manifest, FIELDS, PRIOR_CAPTURE_FIELDS
from capture_validation import ensemble_report, replay_capture_predictions, roi_audit
from frame_selection import select_best_frame
from model_contract import file_hash
from test_capture import video, tissue_frame, eye_detector


class CaptureSavingTests(unittest.TestCase):
    def make_capture(self, root, confirmed=False, predicted=False):
        source = root / "video.avi"
        video(source, [tissue_frame()])
        result = select_best_frame(source, detector=eye_detector)
        choose_capture_roi(result)
        result["roi_confirmed"] = confirmed
        result["camera_settings"] = camera_metadata(dict(camera_name="Test camera", width=1280,
            height=720, frame_rate=30, mime_type="video/webm", video_bits_per_second=8000000))
        if predicted:
            result["prediction"] = dict(status="ok", label="Anemic", anemia_score=.8, input_format="rectangular",
                input_pixel_sha256=pixel_hash(result["_roi_result"]["roi"]))
        return result

    def test_confirmed_roi_prediction_is_saved_when_quality_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root, confirmed=True)
            result["candidates"][result["best"]]["ok"] = False
            prediction = dict(status="uncertain", label="Anemic", anemia_score=.6,
                              input_format="rectangular",
                              input_pixel_sha256=pixel_hash(result["_roi_result"]["roi"]))
            predictor = CapturePredictor()
            with patch.object(predictor, "predict_roi", return_value=prediction):
                result["prediction"] = predictor.predict(result, allow_review=True)
            row = save_capture(result, root / "saved", "test", "10cm", "indoor")
            with Path(row["csv_path"]).open(newline="") as handle:
                record = next(csv.DictReader(handle))
            self.assertEqual(record["crop_ok"], "False")
            self.assertEqual(record["prediction_status"], "review_only")
            self.assertEqual(record["prediction_label"], "Anemic")
            self.assertEqual(record["anemia_score"], "0.6")
            saved = json.loads(Path(row["metrics_path"]).read_text())["prediction"]
            self.assertTrue(saved["folds_disagree"])
            self.assertTrue(row["folds_disagree"])
            self.assertEqual(saved["input_file_sha256"], file_hash(Path(row["capture_dir"]) / "roi.png"))

    def test_configured_inference_on_an_unconfirmed_roi_saves_for_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root)
            result["candidates"][result["best"]]["ok"] = True
            predictor = CapturePredictor({"require_roi_confirmation": False})
            prediction = dict(status="ok", label="Non-anemic", anemia_score=.2,
                              input_format="rectangular",
                              input_pixel_sha256=pixel_hash(result["_roi_result"]["roi"]))
            with patch.object(predictor, "predict_roi", return_value=prediction):
                result["prediction"] = predictor.predict(result)
            row = save_capture(result, root / "saved", "test", "10cm", "indoor")
            self.assertEqual(row["prediction_status"], "review_only")
            self.assertEqual(row["anemia_score"], .2)
            self.assertFalse(row["crop_ok"])

    def test_unconfirmed_save_is_review_only_and_confirmed_prediction_is_bound_to_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root)
            row = save_capture(result, root / "saved", "test", "10cm", "indoor")
            self.assertEqual(row["capture_status"], "saved_for_review")
            self.assertFalse(row["crop_ok"])
            result["roi_confirmed"] = True
            # This test isolates persistence from the synthetic video's focus score.
            result["candidates"][result["best"]]["ok"] = True
            result["prediction"] = dict(status="ok", input_format="rectangular",
                input_pixel_sha256=pixel_hash(result["_roi_result"]["roi"]))
            result["prediction"].update(label="Anemic", anemia_score=.8)
            row = save_capture(result, root / "saved", "test", "10cm", "indoor")
            saved = json.loads(Path(row["metrics_path"]).read_text())
            self.assertTrue(row["crop_ok"])
            self.assertEqual(saved["capture_status"], "approved_for_testing")
            self.assertEqual(saved["camera_settings"]["width"], 1280)
            self.assertEqual(saved["prediction"]["input_file_sha256"], file_hash(Path(row["capture_dir"]) / "roi.png"))
            result["prediction"]["input_pixel_sha256"] = "outdated"
            stale = save_capture(result, root / "saved", "test", "10cm", "indoor")
            self.assertEqual(json.loads(Path(stale["metrics_path"]).read_text())["prediction"]["status"], "stale")
            with Path(row["csv_path"]).open(newline="") as handle:
                unscored, predicted, invalidated = csv.DictReader(handle)
            self.assertEqual(unscored["prediction_label"], "")
            self.assertEqual(predicted["prediction_label"], "Anemic")
            self.assertEqual(predicted["anemia_score"], "0.8")
            self.assertEqual(predicted["capture_dir"], row["image_id"])
            self.assertEqual(predicted["roi_image_path"], f'{row["image_id"]}/roi_masked.png')
            self.assertEqual(predicted["prediction_input_path"], f'{row["image_id"]}/roi.png')
            self.assertEqual(invalidated["prediction_status"], "stale")
            self.assertEqual(invalidated["prediction_label"], "")
            self.assertEqual(invalidated["prediction_input_path"], "")

    def test_backfill_preserves_results_and_requires_verified_prediction_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root, confirmed=True, predicted=True)
            result["candidates"][result["best"]]["ok"] = True
            roi = result["_roi_result"]["roi_masked"]
            result["prediction"].update(input_format="masked", input_pixel_sha256=pixel_hash(roi))
            result["prediction"]["checkpoints"] = [{"sha256": str(i)} for i in range(5)]
            result["prediction"]["fold_scores"] = [.8] * 5
            output = root / "saved"
            good = save_capture(result, output, "test", "10cm", "indoor")
            changed = save_capture(result, output, "test", "10cm", "indoor")
            (Path(changed["capture_dir"]) / "roi_masked.png").write_bytes(b"changed")
            manifest = Path(good["csv_path"])
            with manifest.open(newline="", encoding="utf-8") as handle:
                original_rows = list(csv.DictReader(handle))
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PRIOR_CAPTURE_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(original_rows)
            prepare_manifest(output)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["prediction_input_path"], f'{good["image_id"]}/roi_masked.png')
            self.assertEqual(rows[1]["prediction_input_path"], "")
            for before, after in zip(original_rows, rows):
                self.assertEqual({k: after[k] for k in PRIOR_CAPTURE_FIELDS},
                                 {k: before[k] for k in PRIOR_CAPTURE_FIELDS})
            with patch.object(CapturePredictor, "predict_roi", return_value=result["prediction"]) as predict:
                report = replay_capture_predictions(rows, output, {})
                self.assertEqual(report["replay_counts"], {"matched": 1, "failed": 1})
                predict.assert_called_once()
                np.testing.assert_array_equal(predict.call_args.args[0]["roi_masked"], roi)
                predict.return_value = dict(result["prediction"], fold_scores=[.79, .81, .8, .8, .8])
                report = replay_capture_predictions(rows[:1], output, {})
                self.assertEqual(report["replay_counts"], {"failed": 1})
                self.assertIn("fold scores differ", report["captures"][0]["reason"])
                rows[0]["prediction_input_path"] = rows[0]["prediction_input_path"].replace("roi_masked.png", "roi.png")
                predict.reset_mock()
                report = replay_capture_predictions(rows, output, {})
                self.assertEqual(report["replay_counts"], {"failed": 2})
                predict.assert_not_called()
            prepare_manifest(output)
            self.assertEqual(len(list(output.glob("dataset.before_prediction_input_*.csv"))), 1)

    def test_recovery_reindexes_complete_capture_once_and_preserves_partial_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root)
            output = root / "saved"
            row = save_capture(result, output, "test", "10cm", "indoor")
            manifest = Path(row["csv_path"])
            with manifest.open("w", newline="") as handle:
                csv.writer(handle).writerow(FIELDS)
            staging = Path(row["capture_dir"]).parent / ".pending_CAP_000002_test"
            staging.mkdir()
            (staging / "partial.txt").write_text("preserve")
            prepare_manifest(output)
            with manifest.open(newline="") as handle:
                recovered = list(csv.DictReader(handle))
            self.assertEqual([record["image_id"] for record in recovered], [row["image_id"]])
            report = json.loads((output / "recovery_report.json").read_text())
            self.assertEqual(report["recovered"], [row["image_id"]])
            self.assertEqual(len(report["needs_review"]), 1)
            self.assertTrue((staging / "partial.txt").is_file())
            self.assertEqual(len(list(output.glob("dataset.before_recovery_*.csv"))), 1)
            prepare_manifest(output)
            self.assertEqual(len(list(output.glob("dataset.before_recovery_*.csv"))), 1)

    def test_recovery_does_not_index_damaged_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root)
            row = save_capture(result, root / "saved", "test", "10cm", "indoor")
            Path(row["image_path"]).write_bytes(b"damaged")
            with Path(row["csv_path"]).open("w", newline="") as handle:
                csv.writer(handle).writerow(FIELDS)
            prepare_manifest(root / "saved")
            with Path(row["csv_path"]).open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            self.assertTrue(Path(row["capture_dir"]).is_dir())
            report = json.loads((root / "saved" / "recovery_report.json").read_text())
            self.assertFalse(report["recovered"])
            self.assertEqual(len(report["needs_review"]), 1)

    def test_recovery_reports_malformed_metadata_without_blocking_startup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root)
            output = root / "saved"
            row = save_capture(result, output, "test", "10cm", "indoor")
            path = Path(row["metrics_path"])
            original = json.loads(path.read_text())
            malformed = [[], dict(original, artifact_sha256=list(original["artifact_sha256"])),
                         dict(original, prediction=[])]
            with Path(row["csv_path"]).open("w", newline="") as handle:
                csv.writer(handle).writerow(FIELDS)
            for metadata in malformed:
                with self.subTest(metadata_type=type(metadata).__name__):
                    path.write_text(json.dumps(metadata))
                    before = path.read_bytes()
                    prepare_manifest(output)
                    report = json.loads((output / "recovery_report.json").read_text())
                    self.assertEqual(report["recovered"], [])
                    self.assertEqual(len(report["needs_review"]), 1)
                    self.assertIn("must be an object", report["needs_review"][0]["reason"])
                    self.assertEqual(path.read_bytes(), before)
                    with Path(row["csv_path"]).open(newline="") as handle:
                        self.assertEqual(list(csv.DictReader(handle)), [])

    def test_existing_patient_must_exist_and_camera_settings_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.make_capture(root)
            with self.assertRaisesRegex(ValueError, "not found"):
                save_capture(result, root / "saved", "test", "10cm", "indoor", "P123", patient_mode="existing")
            row = save_capture(result, root / "saved", "test", "10cm", "indoor", patient_mode="new")
            repeat = save_capture(result, root / "saved", "test", "10cm", "indoor", row["patient_id"], patient_mode="existing")
            self.assertEqual(row["patient_id"], repeat["patient_id"])
        for value in (-1, float("nan"), True):
            with self.assertRaises(ValueError):
                camera_metadata({"width": value})
        with self.assertRaises(ValueError):
            validate_details({"patient_mode": "existing"})

    def test_evaluation_rejects_patient_mismatch_and_replaced_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = np.full((10, 10, 3), 75, np.uint8)
            cv2.imwrite(str(root / "roi.png"), image)
            metrics = dict(roi_confirmed=True, capture=dict(patient_id="P1", crop_ok=True, roi_status="ok"),
                prediction=dict(status="ok", input_format="rectangular", input_pixel_sha256=pixel_hash(image),
                                input_file_sha256=file_hash(root / "roi.png")))
            path = root / "metrics.json"
            path.write_text(json.dumps(metrics))
            row = dict(patient_id="other", split="test", label="0", metrics_path=str(path))
            with self.assertRaisesRegex(ValueError, "patient ID does not match"):
                ensemble_report([row], set())
            row["patient_id"] = "P1"
            cv2.imwrite(str(root / "roi.png"), image + 1)
            with self.assertRaisesRegex(ValueError, "Saved image does not match"):
                ensemble_report([row], set())

    def test_roi_audit_counts_failed_segmentation_as_zero_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            image, mask = Path(temp)/"eye.png", Path(temp)/"truth.png"
            cv2.imwrite(str(image), np.full((40,60,3),190,np.uint8))
            cv2.imwrite(str(mask), np.full((40,60),255,np.uint8))
            report=roi_audit([dict(image_id="pale",group="pale",eye_image_path=str(image),reference_mask_path=str(mask))])
            self.assertEqual(report["groups"]["pale"]["tissue_recall"],0)

    def test_validation_blocks_patient_leakage_and_keeps_test_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=[]
            for i in range(8):
                label=i%2
                folder=Path(temp)/str(i)
                folder.mkdir()
                pixels=np.full((20,20,3), i*20, np.uint8)
                cv2.imwrite(str(folder/"roi.png"),pixels)
                path=folder/"metrics.json"
                path.write_text(json.dumps(dict(roi_confirmed=True,capture=dict(patient_id=f"p{i}",crop_ok=True,roi_status="ok"),prediction=dict(status="ok",input_format="rectangular",
                    input_pixel_sha256=pixel_hash(pixels),input_file_sha256=file_hash(folder/"roi.png"),
                    fold_scores=[.2+.6*label]*5,checkpoints=[dict(sha256=str(j)) for j in range(5)]))))
                rows.append(dict(patient_id=f"p{i}",split="calibration" if i<4 else "test",label=str(label),metrics_path=str(path)))
            with self.assertRaisesRegex(ValueError,"absent"):
                ensemble_report(rows,{"p0"})
            report=ensemble_report(rows,{"training-patient"})
            self.assertEqual(report["test_patients"],4)
            self.assertEqual(report["raw_threshold"], .5)
            self.assertEqual(report["raw_test"]["balanced_accuracy"],1.)
            failed_path = Path(temp) / "failed.json"
            failed_path.write_text(json.dumps(
                dict(roi_confirmed=False, capture=dict(patient_id="p4", crop_ok=False, roi_status="failed"))
            ))
            extra = [dict(patient_id="p4", split="test", label="0", metrics_path=str(failed_path)),
                     dict(patient_id="p8", split="test", label="0", metrics_path="")]
            report = ensemble_report(rows + extra, set(), include_unscored=True)
            coverage = report["capture_coverage"]["test"]
            self.assertEqual(coverage["outcomes"], {"scored": 4, "not_eligible": 1, "not_saved": 1})
            self.assertEqual(coverage["quality_pass_fraction"], 4 / 6)
            self.assertEqual((coverage["patients_enrolled"], coverage["patients_scored"]), (5, 4))
            self.assertEqual(report["raw_test"]["balanced_accuracy"], 1.)
            rows[-1]["patient_id"]="p0"
            with self.assertRaisesRegex(ValueError,"does not match"):
                ensemble_report(rows,set())
            rows[-1]["patient_id"] = "p7"
            last_path = Path(rows[-1]["metrics_path"])
            last_capture = json.loads(last_path.read_text())
            last_capture["roi_confirmed"] = "False"
            last_path.write_text(json.dumps(last_capture))
            with self.assertRaisesRegex(ValueError, "requires confirmed ROI"):
                ensemble_report(rows, set())
            last_capture["roi_confirmed"] = True
            rows[-1]["patient_id"] = "p0"
            last_capture["capture"]["patient_id"] = "p0"
            last_path.write_text(json.dumps(last_capture))
            with self.assertRaisesRegex(ValueError, "cross splits"):
                ensemble_report(rows, set())
            last_capture["capture"]["patient_id"] = "p7"
            last_path.write_text(json.dumps(last_capture))
            rows[-1]["patient_id"]="p7"

            last_capture["prediction"]["threshold"] = .85
            last_path.write_text(json.dumps(last_capture))
            with self.assertRaisesRegex(ValueError, "different thresholds"):
                ensemble_report(rows, set())
            for row in rows[:-1]:
                path = Path(row["metrics_path"])
                capture = json.loads(path.read_text())
                capture["prediction"]["threshold"] = .85
                path.write_text(json.dumps(capture))
            report = ensemble_report(rows, set())
            self.assertEqual(report["raw_threshold"], .85)
            self.assertEqual(report["raw_test"]["balanced_accuracy"], .5)

            path = Path(rows[0]["metrics_path"])
            capture = json.loads(path.read_text())
            checkpoints = capture["prediction"]["checkpoints"]
            checkpoints[1] = checkpoints[0]
            path.write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, "distinct checkpoint hashes"):
                ensemble_report(rows, set())


if __name__ == "__main__":
    unittest.main()
