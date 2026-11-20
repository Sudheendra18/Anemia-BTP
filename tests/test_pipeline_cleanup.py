"""Regression checks for preprocessing, checkpoint loading, and ablation reuse."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from dataset import build_transforms
from image_io import read_image, write_image
from model_contract import check_run_manifest, input_format, training_identity
from test_capture import textured, tissue_frame
from predict import DirectInferenceDataset, load_ensemble_models
from train_no_augmentation import (
    load_augmented_baseline_metrics,
    run_no_augmentation_ablation,
)


class PipelineCleanupTests(unittest.TestCase):
    def test_disabled_augmentation_matches_evaluation_pixels(self):
        pixels = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
        image = Image.fromarray(pixels)
        expected = build_transforms(16, train=False)(image)
        for augmentation in (False, {"enabled": False, "color_jitter": 0.9}):
            with self.subTest(augmentation=augmentation):
                actual = build_transforms(16, train=True, augmentation=augmentation)(image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prediction_decodes_once_and_reuses_evaluation_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture_\u6d4b\u8bd5"
            folder.mkdir()
            path = folder / "eye.png"
            pixels = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
            write_image(path, pixels)
            crop = pixels[2:10, 3:15]
            dataset = DirectInferenceDataset([path], image_size=16, do_extract_roi=True)
            with patch("predict.read_image", wraps=read_image) as decode, patch(
                "predict.extract_roi_array", return_value={"status": "ok", "roi": crop}
            ) as extract:
                tensor, name, saved_path, status = dataset[0]
            decode.assert_called_once_with(path)
            np.testing.assert_array_equal(extract.call_args.args[0], pixels)
            expected = build_transforms(16, train=False)(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            )
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
            self.assertEqual((name, saved_path, status), (path.name, str(path.resolve()), "ok"))

    def test_single_fold_and_ensemble_load_the_selected_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for fold in range(1, 6):
                checkpoint = folder / f"fold_{fold}" / "best_model.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"test weights")
            for fold, expected_names in ((1, ["fold_1"]), (None, [f"fold_{i}" for i in range(1, 6)])):
                with self.subTest(fold=fold), patch(
                    "predict.build_model", side_effect=lambda *args, **kwargs: Mock()
                ) as build, patch("predict.torch.load", return_value={}) as load:
                    models = load_ensemble_models(folder, fold, "resnet50", 2, torch.device("cpu"))
                    self.assertEqual([name for name, _ in models], expected_names)
                    self.assertEqual(build.call_count, len(expected_names))
                    self.assertEqual(
                        [call.args[0] for call in load.call_args_list],
                        [folder / name / "best_model.pt" for name in expected_names],
                    )
                    for _, model in models:
                        model.load_state_dict.assert_called_once_with({})
                        model.eval.assert_called_once()
            (folder / "fold_3" / "best_model.pt").unlink()
            for fold in (None, 3):
                with self.assertRaisesRegex(FileNotFoundError, "fold_3"), patch("predict.build_model") as build:
                    load_ensemble_models(folder, fold, "resnet50", 2, torch.device("cpu"))
                build.assert_not_called()

    def test_rejected_ablation_resume_preserves_saved_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            saved_config = folder / "resnet50" / "effective_config.yaml"
            saved_config.parent.mkdir()
            saved_config.write_text("training: {architecture: original}", encoding="utf-8")
            original = saved_config.read_bytes()
            config = {"training": {}, "ablation": {"output_dir": str(folder)}}
            with patch("train_no_augmentation.run_kfold_cv", side_effect=ValueError("run mismatch")):
                with self.assertRaisesRegex(ValueError, "run mismatch"):
                    run_no_augmentation_ablation(config, Mock(), models=["resnet50"])
            self.assertEqual(saved_config.read_bytes(), original)

    def test_augmented_baseline_excludes_no_augmentation_leaderboard_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = {
                "experiments": {"output_dir": str(folder)},
                "training": {"output_dir": str(folder / "baseline")},
            }
            pd.DataFrame([
                {"experiment": "resnet50_no_aug", "accuracy_mean": 0.99, "accuracy_std": 0.01},
                {"experiment": "resnet50_custom", "accuracy_mean": 0.75, "accuracy_std": 0.04},
            ]).to_csv(folder / "leaderboard.csv", index=False)
            result = load_augmented_baseline_metrics("resnet50", config, Mock())
            self.assertEqual(result["accuracy"], (0.75, 0.04))

            pd.DataFrame([
                {"experiment": "resnet50_no_aug", "accuracy_mean": 0.99, "accuracy_std": 0.01},
            ]).to_csv(folder / "leaderboard.csv", index=False)
            self.assertIsNone(load_augmented_baseline_metrics("resnet50", config, Mock()))

    def test_training_identity_changes_with_pixels_patient_split_and_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp)/"roi.png"
            cv2.imwrite(str(image), textured())
            config = {"training":{"image_size":224}, "dataset":{"label_map":{"Non-anemic":0,"Anemic":1}}}
            df = pd.DataFrame([dict(image_id="a",patient_id="p1",label=0,roi_image_path=str(image)),
                               dict(image_id="b",patient_id="p2",label=1,roi_image_path=str(image))])
            original = training_identity(config, df.iloc[:1], df.iloc[1:], "roi_image_path")
            self.assertNotEqual(original, training_identity(config, df.iloc[1:], df.iloc[:1], "roi_image_path"))
            cv2.imwrite(str(image), tissue_frame())
            self.assertNotEqual(original, training_identity(config, df.iloc[:1], df.iloc[1:], "roi_image_path"))
            manifest = dict(version=1,fingerprint=original,input_format="rectangular")
            folder=Path(temp)
            (folder/"metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"Cannot verify"):
                check_run_manifest(folder, manifest)
            (folder/"run_manifest.json").write_text(json.dumps(manifest))
            check_run_manifest(folder, manifest)
            with self.assertRaisesRegex(ValueError,"changed"):
                check_run_manifest(folder, dict(manifest,fingerprint="new"))
            df.loc[1,"roi_image_path"] = str(folder/"roi_masked.png")
            with self.assertRaisesRegex(ValueError,"mix"):
                input_format(df,"roi_image_path")

    def test_input_format_uses_batch_masked_path_column(self):
        df = pd.DataFrame({
            "roi_image_path": ["masked/eye_001.png", "masked/eye_002.png"],
            "roi_masked_path": ["masked/eye_001.png", "masked/eye_002.png"],
        })
        self.assertEqual(input_format(df, "roi_image_path"), "masked")
        df.loc[0, "roi_image_path"] = "rectangular/eye_001.png"
        with self.assertRaisesRegex(ValueError, "mix"):
            input_format(df, "roi_image_path")


if __name__ == "__main__":
    unittest.main()
