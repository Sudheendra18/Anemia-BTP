"""Regression checks for CSV identity, damaged manifests and ROI repairs."""
import csv
import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import import_captures
import capture
from capture_storage import FIELDS, prepare_manifest, save_capture
from frame_selection import CaptureConfig, select_best_frame
from image_io import read_image
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
    def test_capture_save_repair_and_recovery_support_unicode_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip = root / "test.avi"
            output = root / "captures_\u6d4b\u8bd5"
            video(clip, [tissue_frame()])
            row = save_capture(select_best_frame(clip, detector=eye_detector), output, "test", "10cm", "indoor")
            folder, manifest = Path(row["capture_dir"]), output / "dataset.csv"
            for name in ("frame.png", "eye.png", "roi.png", "roi_debug.png"):
                self.assertIsNotNone(read_image(folder / name), name)
            self.assertEqual(repair_capture(folder, manifest, {})["status"], "ok")
            write_manifest(manifest, [])
            prepare_manifest(output)
            with manifest.open(newline="", encoding="utf-8") as handle:
                recovered = list(csv.DictReader(handle))
            self.assertEqual([record["image_id"] for record in recovered], [row["image_id"]])

    def test_compaction_restores_all_files_when_csv_update_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip, output = root / "test.avi", root / "saved"
            video(clip, [tissue_frame()])
            for _ in range(2):
                row = save_capture(select_best_frame(clip, detector=eye_detector), output, "test", "10cm", "indoor")
                folder = Path(row["capture_dir"])
                (folder / "recording.avi").write_bytes(clip.read_bytes())
                (folder / "candidate_glare.png").write_bytes((folder / "candidate_glare_review.png").read_bytes())
            before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            with patch("compact_captures._write_manifest", side_effect=PermissionError("CSV is locked")):
                with self.assertRaisesRegex(PermissionError, "CSV is locked"):
                    compact(output)
            after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(after, before)
            self.assertEqual(compact(output)["removed_files"], 4)

    def test_startup_recovers_compaction_interrupted_before_or_after_csv_commit(self):
        child = """
import os, sys
sys.path.insert(0, 'scripts')
import compact_captures as command
write_manifest = command._write_manifest
def interrupted(path, rows):
    if sys.argv[2] == 'after':
        write_manifest(path, rows)
    os._exit(86)
command._write_manifest = interrupted
command.compact(sys.argv[1])
"""
        for moment in ("before", "after"):
            with self.subTest(moment=moment), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                clip, output = root / "test.avi", root / "saved"
                video(clip, [tissue_frame()])
                row = save_capture(select_best_frame(clip, detector=eye_detector), output, "test", "10cm", "indoor")
                folder = Path(row["capture_dir"])
                clip_bytes = clip.read_bytes()
                recording = folder / "recording.avi"
                recording.write_bytes(clip_bytes)
                manifest = output / "dataset.csv"
                row["roi_image_path"] = (folder / "roi.png").as_posix()
                write_manifest(manifest, [[row[field] for field in FIELDS]])
                original_csv = manifest.read_bytes()
                metrics = folder / "metrics.json"
                original_metrics = metrics.read_bytes()

                result = subprocess.run([sys.executable, "-c", child, str(output), moment], cwd=ROOT, timeout=45)
                self.assertEqual(result.returncode, 86)
                self.assertFalse(recording.exists())
                prepare_manifest(output)
                report = json.loads((output / "recovery_report.json").read_text(encoding="utf-8"))
                self.assertFalse(list(output.glob(".compact_*")))
                if moment == "before":
                    self.assertEqual(recording.read_bytes(), clip_bytes)
                    self.assertEqual(metrics.read_bytes(), original_metrics)
                    self.assertEqual(manifest.read_bytes(), original_csv)
                    self.assertEqual(report["compactions"][0]["action"], "restored original captures")
                else:
                    self.assertFalse(recording.exists())
                    self.assertIn(b"roi_masked.png", manifest.read_bytes())
                    self.assertEqual(report["compactions"][0]["action"], "kept committed compaction")

    def test_copied_capture_is_reindexed_only_when_local_artifact_hashes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip, source = root / "test.avi", root / "saved"
            video(clip, [tissue_frame()])
            capture = save_capture(select_best_frame(clip, detector=eye_detector), source, "test", "10cm", "indoor")
            for altered in (False, True):
                with self.subTest(altered=altered):
                    output = root / f"copied_{altered}"
                    folder = output / capture["image_id"]
                    shutil.copytree(capture["capture_dir"], folder)
                    if altered:
                        (folder / "frame.png").write_bytes(b"changed")
                    manifest = prepare_manifest(output)
                    with manifest.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    if altered:
                        self.assertEqual(rows, [])
                    else:
                        self.assertEqual(rows[0]["capture_dir"], capture["image_id"])
                        self.assertEqual(rows[0]["roi_image_path"], f'{capture["image_id"]}/roi_masked.png')

    def test_invalid_haar_detector_fails_before_capture_storage_is_created(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()) as stderr:
            output = Path(temp) / "captures"
            with patch.object(sys, "argv", ["capture.py", "--out", str(output)]), \
                    patch("eye_detection._haar", None), patch("eye_detection.cv2.CascadeClassifier", create=True) as classifier:
                classifier.return_value.empty.return_value = True
                with self.assertRaises(SystemExit) as error:
                    capture.main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn("OpenCV eye detector is unavailable", stderr.getvalue())
            self.assertFalse(output.exists())

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
