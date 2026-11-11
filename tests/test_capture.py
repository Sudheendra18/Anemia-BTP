import json
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from frame_selection import CaptureConfig, EyeTracker, RegionDetector, quality, select_best_frame
from capture_storage import FIELDS, FULL_FIELDS, LEGACY_FIELDS, next_patient_id, prepare_manifest, save_capture, safe_component
from import_captures import join_labels
from roi_extraction import extract_roi, extract_roi_array
from capture_roi import extract_capture_roi
from reextract_capture_roi import repair_capture
from capture import validate_details


def eye_detector(frame, allow_guess=False):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[h//4:3*h//4, w//4:3*w//4] = 255
    return [dict(box=(w//4, h//4, 3*w//4, 3*h//4), roi_mask=mask, ref_mask=None,
                 oval=dict(cx=w/2, cy=h/2, rx=w/4, ry=h/4, angle=0.),
                 side="single", method="test")]


def textured():
    rng = np.random.default_rng(42)
    gray = rng.integers(75, 185, (180, 320), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def tissue_frame():
    """Known pink crescent surrounded by warm skin and a large distracting finger."""
    frame = np.full((180, 320, 3), (70, 100, 145), np.uint8)
    cv2.ellipse(frame, (160, 80), (65, 40), 0, 0, 360, (115, 100, 185), -1)
    cv2.ellipse(frame, (160, 69), (64, 38), 0, 0, 360, (170, 170, 175), -1)
    cv2.circle(frame, (160, 68), 25, (40, 35, 30), -1)
    cv2.rectangle(frame, (70, 145), (270, 179), (130, 160, 210), -1)
    return frame


def video(path, frames):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (w, h))
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder unavailable")
    for frame in frames:
        writer.write(frame)
    writer.release()


class CaptureTests(unittest.TestCase):
    def test_automatic_patient_sequence_and_explicit_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"eye.avi"
            video(path, [tissue_frame()])
            result = select_best_frame(path, detector=eye_detector)
            self.assertEqual(next_patient_id(output), "P000001")
            first = save_capture(result, path, output, "test", "10cm", "indoor")
            self.assertEqual(first["patient_id"], "P000001")
            second = save_capture(result, path, output, "test", "10cm", "indoor")
            self.assertEqual(second["patient_id"], "P000002")
            repeated = save_capture(result, path, output, "test", "10cm", "indoor", second["patient_id"])
            self.assertEqual(repeated["patient_id"], "P000002")
            self.assertEqual(repeated["serial_id"], 3)
            self.assertEqual(next_patient_id(output), "P000003")
            save_capture(result, path, output, "test", "10cm", "indoor", "P009")
            self.assertEqual(next_patient_id(output), "P000010")
            restarted = save_capture(result, path, output, "test", "10cm", "indoor")
            self.assertEqual(restarted["patient_id"], "P000010")

    def test_optional_capture_details_and_validation(self):
        self.assertEqual(validate_details({}), dict(device="webcam", distance="unspecified",
                                                    lighting="unspecified", patient_id=""))
        details = validate_details(dict(device=" laptop ", patient_id="001", distance="15cm", lighting="daylight"))
        self.assertEqual(details["device"], "laptop")
        self.assertEqual(details["patient_id"], "001")
        with self.assertRaisesRegex(ValueError, "Device"):
            validate_details(dict(device="../unsafe"))

    def test_manual_region_coordinates_and_validation(self):
        frame = textured()
        tracker = EyeTracker(detector=RegionDetector([.25, .25, .5, .5]))
        _, eye, q = tracker.update(frame)
        self.assertEqual(eye["box"], (80, 45, 240, 135))
        self.assertEqual(eye["method"], "manual_region")
        self.assertTrue(q["valid"])
        self.assertEqual(tracker.detection_calls, 0)
        for invalid in ([0, 0, 2, 1], [0, 0, 0, 1], [0, 0, .5], [-.1, 0, .5, .5]):
            with self.assertRaises(ValueError):
                RegionDetector(invalid)

    def test_blur_exposure_and_glare(self):
        sharp = textured()
        cfg = CaptureConfig()
        eye = eye_detector(sharp)[0]
        q = quality(sharp, eye, cfg)
        blurry = quality(cv2.GaussianBlur(sharp, (15, 15), 4), eye, cfg)
        self.assertGreater(q["score"], blurry["score"])
        self.assertTrue(q["ok"])
        self.assertFalse(blurry["ok"])
        for value in (0, 255):
            poor = quality(np.full_like(sharp, value), eye, cfg)
            self.assertFalse(poor["ok"])
        # A pale/neutral ROI is eligible; redness is diagnostic only.
        self.assertEqual(q["coverage"], 0)

    def test_streaming_selection_and_original_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.avi"
            sharp = cv2.resize(textured(), (1280, 720), interpolation=cv2.INTER_NEAREST)
            blur = cv2.GaussianBlur(sharp, (41, 41), 12)
            video(path, [blur, sharp, np.zeros_like(sharp)])
            result = select_best_frame(path, CaptureConfig(stride=1, detect_interval=1), eye_detector)
            best = result["candidates"]["overall"]
            self.assertEqual(best["idx"], 2)
            self.assertEqual(best["frame"].shape, sharp.shape)
            self.assertEqual(best["eye"]["box"], (320, 180, 960, 540))
            self.assertTrue(result["eye_found"])
            self.assertLessEqual(result["retained_frames"], 4)

    def test_no_eye_never_produces_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.avi"
            video(path, [textured()] * 4)
            result = select_best_frame(path, detector=lambda *a, **kw: [])
            self.assertFalse(result["eye_found"])
            self.assertEqual(result["analyzed_frames"], 2)
            self.assertTrue(all(c["eye"] is None and not c["ok"] for c in result["candidates"].values()))
            row = save_capture(result, path, Path(temp)/"saved", "test", "10cm", "indoor")
            self.assertEqual(row["roi_status"], "no_eye")
            self.assertEqual(row["roi_image_path"], "")

    def test_flow_reduces_detection_and_recovers_on_loss(self):
        tracker = EyeTracker(CaptureConfig(detect_interval=5), eye_detector)
        frame = textured()
        for _ in range(10):
            tracker.update(frame)
        self.assertEqual(tracker.detection_calls, 2)
        self.assertEqual(tracker.flow_frames, 8)
        tracker.detector = lambda *a, **kw: []
        self.assertFalse(tracker.live(np.zeros_like(frame))["found"])
        self.assertIsNone(tracker.eye)

    def test_save_clean_frame_and_roi_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.avi"
            video(path, [tissue_frame()])
            result = select_best_frame(path, detector=eye_detector)
            row = save_capture(result, path, Path(temp)/"saved", "test", "10cm", "indoor", "001")
            np.testing.assert_array_equal(cv2.imread(row["image_path"]), result["candidates"]["overall"]["frame"])
            self.assertTrue(Path(row["roi_image_path"]).is_file())
            self.assertNotIn("label", row)
            metadata = json.loads(Path(row["metrics_path"]).read_text())
            self.assertEqual(metadata["capture"]["patient_id"], "001")
            self.assertEqual(row["roi_status"], "ok")
            for name in ("roi_mask.png", "roi_debug.png"):
                self.assertTrue((Path(row["capture_dir"])/name).is_file())
            self.assertEqual(row["serial_id"], 1)
            self.assertEqual(row["image_id"], "CAP_000001")
            for key in ("sharpness_path", "exposure_path", "glare_path", "eye_image_path", "roi_image_path"):
                self.assertTrue(Path(row[key]).is_file(), key)
            second = save_capture(result, path, Path(temp)/"saved", "phone", "20cm", "daylight")
            self.assertEqual(second["serial_id"], 2)
            self.assertEqual(second["image_id"], "CAP_000002")
            with Path(row["csv_path"]).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["serial_id"] for r in rows], ["1", "2"])
            self.assertEqual(rows[0]["sharpness_path"], row["sharpness_path"])

    def test_segmentation_rejects_skin_and_does_not_fabricate_fallback(self):
        frame = tissue_frame()
        eye = eye_detector(frame)[0]
        # Match the stored candidate convention: mask and crop are local to box.
        x0, y0, x1, y1 = eye["box"]
        eye["crop"] = frame[y0:y1, x0:x1]
        eye["roi_search_mask"] = eye["roi_mask"][y0:y1, x0:x1]
        result = extract_capture_roi(eye)
        self.assertEqual(result["status"], "ok")
        rx0, ry0, rx1, ry1 = result["bbox"]
        self.assertLess(ry1+y0, 145, "The finger must be outside the extracted ROI")
        self.assertGreater(rx1-rx0, ry1-ry0)
        self.assertTrue(np.all(result["roi_masked"][result["mask"] == 0] == 0))
        np.testing.assert_array_equal(result["roi_masked"][result["mask"] > 0],
                                      result["roi"][result["mask"] > 0])
        eye["crop"] = np.full_like(eye["crop"], (70, 100, 145))
        failed = extract_capture_roi(eye)
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["roi"])

    def test_failed_segmentation_leaves_no_roi_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"gray.avi"
            video(path, [textured()])
            result = select_best_frame(path, detector=eye_detector)
            row = save_capture(result, path, Path(temp)/"saved", "test", "10cm", "indoor")
            self.assertEqual(row["roi_status"], "failed")
            self.assertFalse(row["crop_ok"])
            self.assertEqual(row["roi_image_path"], "")
            self.assertTrue(Path(row["eye_image_path"]).is_file())
            self.assertFalse((Path(row["capture_dir"])/"roi.png").exists())

    def test_legacy_csv_upgrade_preserves_images_and_continues_serials(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"test.avi"
            video(path, [textured()])
            result = select_best_frame(path, detector=eye_detector)
            original = save_capture(result, path, output, "test", "10cm", "indoor")
            frame_bytes = Path(original["image_path"]).read_bytes()
            # Recreate the previous release's CSV layout.
            with (output/"dataset.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LEGACY_FIELDS)
                writer.writeheader()
                writer.writerow({k: original[k] for k in LEGACY_FIELDS})
            prepare_manifest(output)
            with (output/"dataset.csv").open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, FIELDS)
            self.assertEqual(rows[0]["serial_id"], "1")
            self.assertEqual(rows[0]["image_id"], original["image_id"])
            for key in ("sharpness_path", "exposure_path", "glare_path"):
                self.assertTrue(Path(rows[0][key]).is_file())
            self.assertEqual(Path(original["image_path"]).read_bytes(), frame_bytes)
            self.assertEqual(len(list(output.glob("dataset.before_serial_ids_*.csv"))), 1)
            prepare_manifest(output)
            self.assertEqual(len(list(output.glob("dataset.before_serial_ids_*.csv"))), 1)
            next_row = save_capture(result, path, output, "test", "10cm", "indoor")
            self.assertEqual(next_row["serial_id"], 2)

    def test_failed_csv_write_can_be_retried_without_duplicate_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"test.avi"
            video(path, [textured()])
            result = select_best_frame(path, detector=eye_detector)
            prepare_manifest(output)
            with patch("capture_storage.os.replace", side_effect=PermissionError("File in use")):
                with self.assertRaisesRegex(PermissionError, "Close it in Excel"):
                    save_capture(result, path, output, "test", "10cm", "indoor")
            self.assertEqual(list(output.glob("*/*/*/CAP_*")), [])
            row = save_capture(result, path, output, "test", "10cm", "indoor")
            self.assertEqual(row["serial_id"], 1)

    def test_csv_cleanup_preserves_useful_fields_and_future_save_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"test.avi"
            video(path, [textured()])
            result = select_best_frame(path, detector=eye_detector)
            original = save_capture(result, path, output, "test", "10cm", "indoor", "001")
            metadata_before = Path(original["metrics_path"]).read_bytes()
            with (output/"dataset.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FULL_FIELDS)
                writer.writeheader()
                writer.writerow({k: original[k] for k in FULL_FIELDS})
            prepare_manifest(output)
            save_capture(result, path, output, "phone", "20cm", "daylight")
            with (output/"dataset.csv").open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, FIELDS)
            self.assertEqual(len(FIELDS), 17)
            self.assertEqual([r["serial_id"] for r in rows], ["1", "2"])
            self.assertEqual(rows[0]["patient_id"], "001")
            for key in FIELDS:
                self.assertEqual(rows[0][key], str(original[key]))
            self.assertEqual(Path(original["metrics_path"]).read_bytes(), metadata_before)
            self.assertEqual(len(list(output.glob("dataset.before_cleanup_*.csv"))), 1)

    def test_empty_video_invalid_settings_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"empty.avi"
            path.touch()
            with self.assertRaises(RuntimeError):
                select_best_frame(path)
        for kwargs in ({"stride": 0}, {"detect_interval": 0}, {"max_glare": 2}, {"min_sharpness": float("nan")}):
            with self.assertRaises(ValueError):
                CaptureConfig(**kwargs)
        for label in ("../escape", "CON", "foo/bar", ""):
            with self.assertRaises(ValueError):
                safe_component(label)

    def test_repair_regenerates_roi_and_preserves_source_and_serial(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"eye.avi"
            video(path, [tissue_frame()])
            result = select_best_frame(path, detector=eye_detector)
            row = save_capture(result, path, output, "test", "10cm", "indoor")
            expected = Path(row["roi_image_path"]).read_bytes()
            original_frame = Path(row["image_path"]).read_bytes()
            cv2.imwrite(row["roi_image_path"], np.zeros((20, 20, 3), np.uint8))
            repaired = repair_capture(row["capture_dir"], output/"dataset.csv", {})
            self.assertEqual(repaired["status"], "ok")
            self.assertEqual(Path(row["roi_image_path"]).read_bytes(), expected)
            self.assertEqual(Path(row["image_path"]).read_bytes(), original_frame)
            self.assertTrue((Path(repaired["backup"])/"roi.png").is_file())
            with (output/"dataset.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["serial_id"], "1")

    def test_roi_array_matches_file_and_tiny_input(self):
        frame = textured()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"frame.png"
            cv2.imwrite(str(path), frame)
            a, b = extract_roi(str(path)), extract_roi_array(frame)
            self.assertEqual(a["bbox"], b["bbox"])
            np.testing.assert_array_equal(a["roi"], b["roi"])
        result = extract_roi_array(np.zeros((1, 1, 3), np.uint8))
        self.assertEqual(result["roi"].shape, (1, 1, 3))

    def test_label_import_preserves_groups_and_rejects_failed_crops(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"image.png"
            cv2.imwrite(str(path), textured())
            captures = pd.DataFrame([dict(image_id=f"img{i}", patient_id="001", crop_ok=True,
                                         roi_status="ok", image_path=str(path), roi_image_path=str(path))
                                     for i in range(2)])
            labels = pd.DataFrame([dict(image_id=f"img{i}", patient_id="001", label_text="Anemic")
                                   for i in range(2)])
            joined = join_labels(captures, labels, {"Anemic": 1})
            self.assertEqual(list(joined.patient_id), ["001", "001"])
            self.assertEqual(list(joined.label), [1, 1])
            captures.loc[0, "roi_status"] = "fallback"
            with self.assertRaises(ValueError):
                join_labels(captures, labels, {"Anemic": 1})
            captures.loc[0, "roi_status"] = "ok"
            labels.loc[0, "patient_id"] = "002"
            with self.assertRaises(ValueError):
                join_labels(captures, labels, {"Anemic": 1})


if __name__ == "__main__":
    unittest.main()
