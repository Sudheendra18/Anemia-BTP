import json
import csv
import base64
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
from frame_selection import (
    CRITERIA, CaptureConfig, EyeTracker, RegionDetector, _candidate, quality,
    select_best_frame, source_quality,
)
from capture_ui import _png_b64
from capture_storage import FIELDS, FULL_FIELDS, LEGACY_FIELDS, next_patient_id, prepare_manifest, save_capture, safe_component
from import_captures import join_labels
from model_contract import file_hash
from roi_extraction import extract_roi, extract_roi_array
from capture_roi import choose_capture_roi, extract_capture_roi, manual_capture_roi, tissue_quality
from reextract_capture_roi import repair_capture
from capture import validate_details


def eye_detector(frame):
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
    def test_native_crop_scoring_preserves_fine_detail(self):
        frame = np.full((1080, 1920, 3), 120, np.uint8)
        rng = np.random.default_rng(42)
        frame[400:600, 700:1000] = rng.integers(75, 185, (200, 300, 1), dtype=np.uint8)
        detector = RegionDetector([.4, .4, .08, .08])
        tracker = EyeTracker(detector=detector)
        small, eye, preview = tracker.update(frame)
        native = source_quality(frame, small, eye, tracker.cfg)
        self.assertGreater(native["sharp"], preview["sharp"] * 4)
        live_tracker = EyeTracker(CaptureConfig(min_sharpness=500), detector)
        self.assertTrue(live_tracker.live(frame)["ok"], "Live guidance must score source pixels too")
        blurred = cv2.GaussianBlur(frame, (15, 15), 3)
        self.assertGreater(native["sharp"], source_quality(blurred, small, eye, tracker.cfg)["sharp"] * 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "detail.avi"
            video(path, [blurred, frame])
            result = select_best_frame(path, CaptureConfig(stride=1), detector)
            self.assertEqual(result["best"], "overall")
            self.assertEqual(result["candidates"]["overall"]["idx"], 2)
            self.assertEqual(result["scoring_source"], "original_eye_crop")
            self.assertEqual(result["candidates"]["overall"]["frame"].shape, frame.shape)

    def test_lossless_preview_preserves_original_pixels(self):
        original = cv2.resize(tissue_frame(), (1280, 720), interpolation=cv2.INTER_NEAREST)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(_png_b64(original)), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, original)

    def test_roi_refinement_excludes_native_glare_and_preserves_tissue_pixels(self):
        frame = cv2.resize(tissue_frame(), (1280, 720), interpolation=cv2.INTER_NEAREST)
        # Tiny highlight disappears into the coarse segmentation's blur.
        cv2.circle(frame, (640, 432), 2, (255, 255, 255), -1)
        eye = eye_detector(frame)[0]
        x0, y0, x1, y1 = eye["box"]
        eye.update(crop=frame[y0:y1, x0:x1], roi_search_mask=eye["roi_mask"][y0:y1, x0:x1])
        roi = extract_capture_roi(eye, {"resize_long_side": 160})
        self.assertEqual(roi["status"], "ok")
        a, b, c, d = roi["bbox"]
        self.assertEqual(roi["mask"][432-y0-b, 640-x0-a], 0)
        np.testing.assert_array_equal(roi["roi"], eye["crop"][b:d, a:c])
        np.testing.assert_array_equal(roi["roi_masked"][roi["mask"] > 0], roi["roi"][roi["mask"] > 0])
        self.assertFalse(np.any((roi["mask"] > 0) & (eye["roi_search_mask"][b:d, a:c] == 0)))

    def test_uniform_selected_tissue_does_not_reject_its_own_threshold(self):
        frame = np.full((100, 200, 3), (115, 100, 185), np.uint8)
        eye = RegionDetector([0, 0, 1, 1])(frame)[0]
        eye.update(crop=frame, roi_search_mask=eye["roi_mask"])
        roi = extract_capture_roi(eye)
        self.assertEqual(roi["status"], "ok")
        np.testing.assert_array_equal(roi["roi"], frame)
        eye["crop"] = np.full_like(frame, (70, 100, 145))
        self.assertEqual(extract_capture_roi(eye)["status"], "failed")
        # Blue iris pixels can satisfy Lab a-b but are not pink tissue.
        eye["crop"] = np.full_like(frame, (130, 80, 95))
        self.assertEqual(extract_capture_roi(eye)["status"], "failed")

    def test_roi_failure_selects_another_candidate_and_saves_matching_pixels(self):
        cfg = CaptureConfig()
        bad_frame, good_frame = textured(), tissue_frame()
        def candidate(frame, idx, score):
            eye = eye_detector(frame)[0]
            q = quality(frame, eye, cfg)
            q.update(score=score, ok=True)
            return _candidate(frame, frame, eye, q, idx)
        bad, good = candidate(bad_frame, 1, .9), candidate(good_frame, 2, .8)
        result = dict(best="overall", eye_found=True, n_frames=2,
                      candidates=dict(overall=bad, sharpness=good, exposure=good, glare=bad))
        with patch("capture_roi.extract_capture_roi", wraps=extract_capture_roi) as extract:
            choose_capture_roi(result)
            self.assertEqual(extract.call_count, 2)
        self.assertEqual(result["best"], "sharpness")
        self.assertTrue(result["roi_selection"]["fallback_used"])
        self.assertEqual(result["_roi_result"]["status"], "ok")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"test.avi"
            video(path, [bad_frame, good_frame])
            with patch("capture_storage.extract_capture_roi", side_effect=AssertionError("ROI must be cached")):
                row = save_capture(result, Path(temp)/"saved", "test", "10cm", "indoor")
            np.testing.assert_array_equal(cv2.imread(row["image_path"]), good_frame)
            self.assertEqual(row["best_frame_idx"], 2)
            metadata = json.loads(Path(row["metrics_path"]).read_text())
            self.assertEqual(metadata["roi_selection"]["quality_best"], "overall")
            self.assertEqual(metadata["best"], "sharpness")

    def test_roi_selection_deduplicates_failed_frames_and_preserves_quality_flags(self):
        frame = textured()
        eye = eye_detector(frame)[0]
        bad = _candidate(frame, frame, eye, quality(frame, eye, CaptureConfig()), 1)
        result = dict(best="overall", candidates={name: bad for name in ("overall", "sharpness", "exposure", "glare")})
        with patch("capture_roi.extract_capture_roi", wraps=extract_capture_roi) as extract:
            choose_capture_roi(result)
            self.assertEqual(extract.call_count, 1)
        self.assertEqual(result["best"], "overall")
        self.assertEqual(result["_roi_result"]["status"], "failed")
        good_frame = tissue_frame()
        good_eye = eye_detector(good_frame)[0]
        q = quality(good_frame, good_eye, CaptureConfig())
        q.update(ok=False, score=.1)
        good = _candidate(good_frame, good_frame, good_eye, q, 2)
        result = dict(best="overall", candidates=dict(overall=bad, exposure=good))
        choose_capture_roi(result)
        self.assertEqual(result["best"], "exposure")
        self.assertFalse(result["candidates"][result["best"]]["ok"])

    def test_successful_overall_roi_needs_only_one_extraction(self):
        frame = tissue_frame()
        eye = eye_detector(frame)[0]
        candidate = _candidate(frame, frame, eye, quality(frame, eye, CaptureConfig()), 1)
        result = dict(best="overall", candidates=dict(overall=candidate, sharpness=candidate))
        with patch("capture_roi.extract_capture_roi", wraps=extract_capture_roi) as extract:
            choose_capture_roi(result)
            self.assertEqual(extract.call_count, 1)
        self.assertEqual(result["best"], "overall")
        self.assertFalse(result["roi_selection"]["fallback_used"])

    def test_detector_scores_each_eye_once(self):
        tracker = EyeTracker(CaptureConfig(), lambda frame, **kwargs: eye_detector(frame)*2)
        frame = textured()
        with patch("frame_selection.quality", wraps=quality) as score:
            _, eye, measurement = tracker.update(frame)
            self.assertEqual(score.call_count, 2)
            self.assertEqual(measurement, quality(frame, eye, tracker.cfg))
            tracker.update(frame)
            self.assertEqual(score.call_count, 3, "Optical-flow frames need one new measurement")

    def test_automatic_patient_sequence_and_explicit_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"eye.avi"
            video(path, [tissue_frame()])
            result = select_best_frame(path, detector=eye_detector)
            self.assertEqual(next_patient_id(output), "P000001")
            first = save_capture(result, output, "test", "10cm", "indoor")
            self.assertEqual(first["patient_id"], "P000001")
            second = save_capture(result, output, "test", "10cm", "indoor")
            self.assertEqual(second["patient_id"], "P000002")
            repeated = save_capture(result, output, "test", "10cm", "indoor", second["patient_id"])
            self.assertEqual(repeated["patient_id"], "P000002")
            self.assertEqual(repeated["serial_id"], 3)
            self.assertEqual(next_patient_id(output), "P000003")
            save_capture(result, output, "test", "10cm", "indoor", "P009")
            self.assertEqual(next_patient_id(output), "P000010")
            restarted = save_capture(result, output, "test", "10cm", "indoor")
            self.assertEqual(restarted["patient_id"], "P000010")

    def test_optional_capture_details_and_validation(self):
        self.assertEqual(validate_details({}), dict(patient_mode="", device="webcam", distance="unspecified",
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
            self.assertIsNone(best["eye"]["roi_search_mask"].base)
            self.assertEqual(best["eye"]["roi_search_mask"].shape, best["eye"]["crop"].shape[:2])
            self.assertTrue(result["eye_found"])
            self.assertLessEqual(result["retained_frames"], 12)

    def test_no_eye_never_produces_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.avi"
            video(path, [textured()] * 4)
            result = select_best_frame(path, detector=lambda *a, **kw: [])
            self.assertFalse(result["eye_found"])
            self.assertEqual(result["scoring_source"], "whole_guide_preview")
            self.assertEqual(result["analyzed_frames"], 2)
            self.assertTrue(all(c["eye"] is None and not c["ok"] for c in result["candidates"].values()))
            row = save_capture(result, Path(temp)/"saved", "test", "10cm", "indoor")
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
            row = save_capture(result, Path(temp)/"saved", "test", "10cm", "indoor", "001")
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
            for name in ("sharpness", "exposure", "glare"):
                self.assertTrue(row[name+"_path"].endswith(f"candidate_{name}_review.png"))
                self.assertFalse((Path(row["capture_dir"])/f"candidate_{name}.png").exists())
                self.assertTrue((Path(row["capture_dir"])/f"candidate_{name}_review.png").is_file())
            self.assertFalse(list(Path(row["capture_dir"]).glob("recording.*")))
            self.assertTrue(row["roi_image_path"].endswith("roi_masked.png"))
            self.assertTrue((Path(row["capture_dir"])/"roi.png").is_file())
            second = save_capture(result, Path(temp)/"saved", "phone", "20cm", "daylight")
            self.assertEqual(second["serial_id"], 2)
            self.assertEqual(second["image_id"], "CAP_000002")
            with Path(row["csv_path"]).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["serial_id"] for r in rows], ["1", "2"])
            self.assertEqual(rows[0]["roi_image_path"], "CAP_000001/roi_masked.png")

    def test_segmentation_rejects_skin_and_does_not_fabricate_fallback(self):
        frame = tissue_frame()
        eye = eye_detector(frame)[0]
        # The mask and crop use coordinates local to the eye box.
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
            row = save_capture(result, Path(temp)/"saved", "test", "10cm", "indoor")
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
            original = save_capture(result, output, "test", "10cm", "indoor")
            frame_bytes = Path(original["image_path"]).read_bytes()
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
            self.assertEqual(rows[0]["capture_dir"], original["image_id"])
            for name in ("sharpness", "exposure", "glare"):
                self.assertTrue((output / rows[0]["capture_dir"] / f"candidate_{name}_review.png").is_file())
            self.assertEqual(Path(original["image_path"]).read_bytes(), frame_bytes)
            self.assertEqual(len(list(output.glob("dataset.before_serial_ids_*.csv"))), 1)
            prepare_manifest(output)
            self.assertEqual(len(list(output.glob("dataset.before_serial_ids_*.csv"))), 1)
            next_row = save_capture(result, output, "test", "10cm", "indoor")
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
                    save_capture(result, output, "test", "10cm", "indoor")
            self.assertEqual(list(output.glob("CAP_*")), [])
            row = save_capture(result, output, "test", "10cm", "indoor")
            self.assertEqual(row["serial_id"], 1)

    def test_csv_cleanup_preserves_useful_fields_and_future_save_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"saved"
            path = Path(temp)/"test.avi"
            video(path, [textured()])
            result = select_best_frame(path, detector=eye_detector)
            original = save_capture(result, output, "test", "10cm", "indoor", "001")
            folder = Path(original["capture_dir"])
            nested = output / "test" / "10cm" / "indoor" / folder.name
            nested.parent.mkdir(parents=True)
            folder.rename(nested)
            frame = (nested / "frame.png").read_bytes()
            old_paths = {key: val.replace(folder.as_posix(), nested.as_posix())
                         for key, val in original.items() if isinstance(val, str)}
            with (output/"dataset.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FULL_FIELDS)
                writer.writeheader()
                writer.writerow({k: old_paths.get(k, original[k]) for k in FULL_FIELDS})
            prepare_manifest(output)
            save_capture(result, output, "phone", "20cm", "daylight")
            with (output/"dataset.csv").open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, FIELDS)
            self.assertEqual({key for key in FIELDS if key.endswith("_path")},
                             {"roi_image_path", "prediction_input_path"})
            self.assertEqual([r["serial_id"] for r in rows], ["1", "2"])
            self.assertEqual(rows[0]["patient_id"], "001")
            self.assertEqual(rows[0]["capture_dir"], folder.name)
            self.assertFalse(nested.exists())
            self.assertEqual((folder / "frame.png").read_bytes(), frame)
            capture = json.loads((folder / "metrics.json").read_text())["capture"]
            self.assertEqual(capture["image_path"], original["image_path"])
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
            row = save_capture(result, output, "test", "10cm", "indoor")
            expected = Path(row["roi_image_path"]).read_bytes()
            original_frame = Path(row["image_path"]).read_bytes()
            cv2.imwrite(row["roi_image_path"], np.zeros((20, 20, 3), np.uint8))
            repaired = repair_capture(row["capture_dir"], output/"dataset.csv", {})
            self.assertEqual(repaired["status"], "ok")
            self.assertEqual(Path(row["roi_image_path"]).read_bytes(), expected)
            self.assertEqual(Path(row["image_path"]).read_bytes(), original_frame)
            self.assertTrue((Path(repaired["backup"])/"roi_masked.png").is_file())
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
            records = []
            for i in range(2):
                folder = Path(temp) / f"img{i}"
                folder.mkdir()
                frame, roi = folder / "frame.png", folder / "roi_masked.png"
                for path in (frame, roi):
                    cv2.imwrite(str(path), textured())
                record = dict(image_id=f"img{i}", patient_id="001", crop_ok=True, roi_status="ok",
                              image_path=str(frame), roi_image_path=str(roi))
                metrics = dict(capture=record, artifact_sha256={p.name: file_hash(p) for p in (frame, roi)})
                (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                records.append(record)
            captures = pd.DataFrame(records)
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
            labels.loc[0, "patient_id"] = "001"
            cv2.imwrite(captures.loc[0, "roi_image_path"], np.zeros((20, 20, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "saved hash"):
                join_labels(captures, labels, {"Anemic": 1})

    def test_shortlist_rescues_roi_discarded_by_category_winners(self):
        def candidate(frame, idx):
            eye = eye_detector(frame)[0]
            return _candidate(frame, frame, eye, quality(frame, eye, CaptureConfig()), idx)
        bad, good = candidate(textured(), 1), candidate(tissue_frame(), 2)
        result = dict(best="overall", candidates={name: bad for name in CRITERIA}, _shortlist=[bad, good])
        choose_capture_roi(result)
        self.assertEqual(result["best"], "alternative_2")
        self.assertEqual(result["_roi_result"]["status"], "ok")
        self.assertNotIn("_shortlist", result)
        self.assertFalse(result["candidates"][result["best"]]["ok"])

    def test_manual_outline_preserves_pale_pixels_and_original_colors(self):
        image = np.full((60, 100, 3), (190, 195, 200), np.uint8)
        eye = dict(crop=image)
        roi = manual_capture_roi(eye, [[.1,.2],[.9,.2],[.9,.8],[.1,.8]])
        self.assertEqual(roi["method"], "manual_polygon_v1")
        np.testing.assert_array_equal(roi["roi_masked"][roi["mask"]>0], np.tile([190,195,200], (np.count_nonzero(roi["mask"]),1)))
        self.assertFalse(tissue_quality(roi)["ok"])
        for points in ([[0,0],[1,1]], [[0,0],[2,1],[1,0]], [[0,0],[float('nan'),1],[1,0]]):
            with self.assertRaises(ValueError):
                manual_capture_roi(eye, points)

    def test_orphan_patient_number_is_reserved_on_preview_and_save(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            prepare_manifest(output)
            folder = output / "phone" / "x" / "y" / "CAP_000001"
            folder.mkdir(parents=True)
            (folder / "metrics.json").write_text(json.dumps({"capture":{"patient_id":"P000041"}}))
            self.assertEqual(next_patient_id(output), "P000042")
            frame = tissue_frame()
            eye = eye_detector(frame)[0]
            candidate = _candidate(frame, frame, eye, quality(frame, eye, CaptureConfig()), 1)
            result = dict(best="overall", eye_found=True, candidates={name:candidate for name in CRITERIA})
            row = save_capture(result, output, "phone", "x", "y")
            self.assertEqual(row["patient_id"], "P000042")
            self.assertEqual(row["image_id"], "CAP_000002")
            self.assertEqual(row["next_patient_id"], "P000043")
            (folder / "metrics.json").write_text("broken")
            with self.assertRaisesRegex(ValueError, "Cannot recover patient"):
                next_patient_id(output)


if __name__ == "__main__":
    unittest.main()
