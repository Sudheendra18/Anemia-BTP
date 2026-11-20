import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capture_prediction import CapturePredictor


def selection(ok=True, status="ok"):
    return dict(best="overall", roi_confirmed=True, candidates={"overall": {"ok": ok}},
                _roi_result=dict(status=status, roi=np.zeros((32, 60, 3), np.uint8)))


class PredictionTests(unittest.TestCase):
    def test_failed_quality_or_roi_never_loads_model(self):
        predictor = CapturePredictor()
        with patch.object(predictor, "_load", side_effect=AssertionError("Must not load")):
            self.assertEqual(predictor.predict(selection(False))["status"], "skipped")
            self.assertEqual(predictor.predict(selection(status="failed"))["status"], "skipped")

    def test_preview_reports_disabled_inference_even_for_a_valid_roi(self):
        predictor = CapturePredictor({"enabled": False})
        with patch.object(predictor, "_load", side_effect=AssertionError("Must not load")):
            self.assertEqual(predictor.preview(selection()["_roi_result"])["status"], "disabled")

    def test_missing_weights_and_invalid_threshold_are_explicit(self):
        for threshold in ("not-a-number", None, [], float("inf")):
            predictor = CapturePredictor({"threshold": threshold})
            self.assertEqual(predictor.predict(selection())["status"], "unavailable")
        predictor = CapturePredictor({"model_dir": "nonexistent-model"})
        self.assertEqual(predictor.predict(selection())["status"], "unavailable")
        predictor.threshold = float("nan")
        self.assertEqual(predictor.predict(selection())["status"], "unavailable")

    def test_ensemble_score_mapping_and_cached_models(self):
        import torch
        from dataset import build_transforms
        predictor = CapturePredictor()
        predictor.transform = build_transforms(224, train=False)
        predictor.provenance = []
        observed = []
        def model(tensor):
            observed.append(tensor.shape)
            self.assertFalse(torch.is_grad_enabled())
            return torch.tensor([[0., 2.]])
        predictor.models = [model] * 5
        with patch.object(predictor, "_load", side_effect=AssertionError("Already loaded")):
            result = predictor.predict(selection())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["label"], "Anemic")
        self.assertAlmostEqual(result["anemia_score"], .880797, places=5)
        self.assertEqual(observed, [(1, 3, 224, 224)] * 5)

    def test_input_contract_selects_masked_input_and_rejects_changed_weights(self):
        import yaml
        from model_contract import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=dict(training=dict(architecture="efficientnet_b0",image_source="with_roi",image_size=224),
                        dataset=dict(label_map={"Non-anemic":0,"Anemic":1}))
            (root/"effective_config.yaml").write_text(yaml.safe_dump(config))
            for i in range(1,6):
                folder=root/f"fold_{i}"
                folder.mkdir()
                weights=folder/"best_model.pt"
                weights.write_bytes(b"test checkpoint")
                contract=dict(version=1,input_format="masked",image_size=224,label_map=config["dataset"]["label_map"],
                              preprocessing="rgb_imagenet_resize_v1",checkpoint_sha256=file_hash(weights))
                (folder/"checkpoint_contract.json").write_text(json.dumps(contract))
            predictor=CapturePredictor({"model_dir":str(root)})
            with patch("model.build_model") as build, patch("torch.load",return_value={}):
                predictor._load()
                self.assertEqual(build.call_count,5)
            self.assertEqual(predictor.input_format,"masked")
            (root/"fold_1"/"best_model.pt").write_bytes(b"different checkpoint")
            with patch("model.build_model",side_effect=AssertionError("Must verify contract before load")):
                with self.assertRaisesRegex(ValueError,"does not match"):
                    CapturePredictor({"model_dir":str(root)})._load()

    def test_model_requires_review_and_reports_disagreement(self):
        import torch
        from dataset import build_transforms
        predictor = CapturePredictor()
        data = selection()
        data["roi_confirmed"] = False
        with patch.object(predictor, "_load", side_effect=AssertionError("Must not load")):
            self.assertEqual(predictor.predict(data)["status"], "needs_review")
        data["roi_confirmed"] = True
        data["_roi_result"]["roi_masked"] = np.full((32,60,3), 255, np.uint8)
        predictor.input_format = "masked"
        predictor.transform = build_transforms(224, train=False)
        predictor.provenance = []
        observed = []
        def model(tensor, value):
            observed.append(float(tensor.mean()))
            return torch.tensor([[0., value]])
        predictor.models = [lambda tensor: model(tensor, -2.)] + [lambda tensor: model(tensor, 2.)]*4
        result = predictor.predict(data)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["input_format"], "masked")
        self.assertTrue(all(x > 1 for x in observed))
