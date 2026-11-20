"""Regression checks for CSV identity, damaged manifests and ROI repairs."""
import csv
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import import_captures
from capture_storage import FIELDS, prepare_manifest, save_capture
from frame_selection import CaptureConfig, select_best_frame
from model_contract import file_hash
from reextract_capture_roi import repair_capture
from test_capture import eye_detector, tissue_frame, video
from utils import read_metadata_csv
from compact_captures import compact


def write_manifest(path, rows, encoding="utf-8"):
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class CaptureIntegrityTests(unittest.TestCase):
    def test_compaction_keeps_legacy_recording_and_syncs_invalidated_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            path, output = Path(temp)/"eye.avi", Path(temp)/"saved"
            video(path, [tissue_frame()])
            row = save_capture(select_best_frame(path, detector=eye_detector), output, "test", "10cm", "indoor")
            folder = Path(row["capture_dir"])
            recording = folder/"recording.avi"
            recording.write_bytes(path.read_bytes())
            metrics_path = folder/"metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics.pop("eye_geometry")
            metrics["capture"].update(video_path=str(recording), crop_ok=True)
            metrics["prediction"] = {"status": "uncertain", "label": "Review needed"}
            metrics_path.write_text(json.dumps(metrics))
            row["roi_image_path"] = (folder/"roi.png").as_posix()
            row["crop_ok"] = True
            (folder/"roi_masked.png").unlink()
            write_manifest(output/"dataset.csv", [[row[key] for key in FIELDS]])
            result = compact(output)
            self.assertEqual(result["preserved_recordings"], [recording.as_posix()])
            self.assertTrue(recording.is_file())
            updated = json.loads(metrics_path.read_text())
            self.assertFalse(updated["capture"]["crop_ok"])
            self.assertEqual(updated["capture"]["video_path"], str(recording))
            self.assertEqual(updated["prediction"]["status"], "stale")

    def test_compaction_preserves_roi_and_review_and_migrates_masked_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path, output = Path(temp)/"eye.avi", Path(temp)/"saved"
            video(path, [tissue_frame()])
            row = save_capture(select_best_frame(path, detector=eye_detector), output, "test", "10cm", "indoor")
            folder = Path(row["capture_dir"])
            (folder/"candidate_glare.png").write_bytes((folder/"candidate_glare_review.png").read_bytes())
            (folder/"recording.avi").write_bytes(path.read_bytes())
            row["roi_image_path"] = (folder/"roi.png").as_posix()
            write_manifest(output/"dataset.csv", [[row[key] for key in FIELDS]])
            result = compact(output)
            self.assertEqual(result["removed_files"], 2)
            for name in ("roi.png", "roi_masked.png", "roi_mask.png", "roi_debug.png", "candidate_glare_review.png", "eye.png", "frame.png"):
                self.assertTrue((folder/name).is_file(), name)
            with (output/"dataset.csv").open(newline="", encoding="utf-8") as handle:
                saved = next(csv.DictReader(handle))
            self.assertTrue(saved["roi_image_path"].endswith("roi_masked.png"))
            self.assertEqual(compact(output)["removed_files"], 0)

    def test_metadata_preserves_ids_and_missing_image_paths(self):
        source = io.StringIO("image_id,patient_id,image_path,label\n001,NA,,1\n002,001,eye.png,0\n003,NULL,other.png,1\n004,,fourth.png,0\n")
        rows = read_metadata_csv(source)
        self.assertEqual(list(rows.image_id), ["001", "002", "003", "004"])
        self.assertEqual(list(rows.patient_id), ["NA", "001", "NULL", ""])
        self.assertTrue(pd.isna(rows.image_path.iloc[0]))
        self.assertEqual(list(rows.label), [1, 0, 1, 0])

    def test_malformed_manifest_is_rejected_without_rewriting(self):
        row = {key: "" for key in FIELDS}
        row.update(serial_id="1", image_id="CAP_000001", patient_id="P000001")
        values = [row[key] for key in FIELDS]
        for malformed in (values[:-1], values + ["unexpected extra column"]):
            with self.subTest(length=len(malformed)), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "dataset.csv"
                write_manifest(path, [malformed])
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "columns"):
                    prepare_manifest(temp)
                self.assertEqual(path.read_bytes(), before)

    def test_duplicate_capture_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.csv"
            rows = []
            for serial in (1, 2):
                row = dict.fromkeys(FIELDS, "")
                row.update(serial_id=str(serial), image_id="CAP_000001")
                rows.append([row[key] for key in FIELDS])
            write_manifest(path, rows)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "image IDs"):
                prepare_manifest(temp)
            self.assertEqual(path.read_bytes(), before)

    def test_excel_utf8_bom_manifest_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.csv"
            write_manifest(path, [], encoding="utf-8-sig")
            self.assertEqual(prepare_manifest(temp), path)

    def test_import_does_not_treat_literal_na_patient_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            image = base / "frame.png"
            image.touch()
            captures, labels, output = (base / name for name in ("captures.csv", "labels.csv", "out.csv"))
            pd.DataFrame([dict(image_id="001", patient_id="NA", crop_ok=True, roi_status="ok",
                               image_path=str(image), roi_image_path=str(image))]).to_csv(captures, index=False)
            pd.DataFrame([dict(image_id="001", patient_id="someone_else", label_text="Anemic")]).to_csv(labels, index=False)
            args = ["import_captures.py", "--captures", str(captures), "--labels", str(labels), "--output", str(output)]
            with patch.object(sys, "argv", args), patch("sys.stderr", new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit) as exit_status:
                    import_captures.main()
                self.assertEqual(exit_status.exception.code, 1)
                self.assertIn("Patient ID conflicts", errors.getvalue())
            self.assertFalse(output.exists())

    def test_repair_recovers_roi_status_without_erasing_quality_failure(self):
        for quality_ok in (True, False):
            with self.subTest(quality_ok=quality_ok), tempfile.TemporaryDirectory() as temp:
                path, output = Path(temp) / "eye.avi", Path(temp) / "saved"
                video(path, [tissue_frame()])
                result = select_best_frame(path, detector=eye_detector)
                for candidate in result["candidates"].values():
                    candidate["ok"] = quality_ok
                    candidate["band_ok"] = quality_ok
                failure = dict(status="failed", bbox=None, method="old_extractor", reason="Old segmentation failed")
                with patch("capture_roi.extract_capture_roi", return_value=failure):
                    row = save_capture(result, output, "test", "10cm", "indoor")
                self.assertFalse(row["crop_ok"])
                folder = Path(row["capture_dir"])
                metrics_path = folder / "metrics.json"
                previous_hashes = json.loads(metrics_path.read_text(encoding="utf-8"))["artifact_sha256"]
                # Mock tissue scoring to isolate recovery of the original quality gate.
                with patch("reextract_capture_roi.tissue_quality", return_value={"ok": True}):
                    repaired = repair_capture(row["capture_dir"], output / "dataset.csv", {})
                self.assertEqual(repaired["status"], "ok")
                with (output / "dataset.csv").open(newline="", encoding="utf-8") as handle:
                    saved = next(csv.DictReader(handle))
                self.assertEqual(saved["crop_ok"], str(quality_ok))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(metrics["capture_status"], "approved_for_testing" if quality_ok else "saved_for_review")
                self.assertEqual(metrics["artifact_sha256"]["frame.png"], previous_hashes["frame.png"])
                for name in ("roi.png", "roi_masked.png", "roi_mask.png", "roi_debug.png"):
                    self.assertEqual(metrics["artifact_sha256"][name], file_hash(folder / name))

    def test_capture_sampling_settings_require_integers(self):
        for name in ("analysis_width", "stride", "detect_interval"):
            for value in (1.5, float("nan"), float("inf"), True, "640", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    CaptureConfig(**{name: value})
        for name in ("min_sharpness", "brightness_min", "brightness_max", "max_glare", "max_clipped"):
            for value in (None, "35", True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    CaptureConfig(**{name: value})


if __name__ == "__main__":
    unittest.main()
