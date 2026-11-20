"""Optional, cached inference using the winning experiment's five saved folds."""
import json
import math

from model_contract import file_hash
from utils import load_config, resolve_path
from capture_provenance import pixel_hash

DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL_DIR = "outputs/experiments/efficientnet_b0_lr1e-4"
ENSEMBLE_FOLDS = range(1, 6)
SCORED_STATUSES = ("ok", "uncertain", "review_only")


class CapturePredictor:
    def __init__(self, options=None):
        options = options or {}
        self.enabled = options.get("enabled", True)
        self.folder = resolve_path(options.get("model_dir", DEFAULT_MODEL_DIR))
        self.threshold = options.get("threshold", DEFAULT_THRESHOLD)
        self.models = None
        self.input_format = options.get("legacy_input_format", "rectangular")
        self.require_confirmation = options.get("require_roi_confirmation", True)

    def checkpoint_paths(self):
        paths = [self.folder / f"fold_{i}" / "best_model.pt" for i in ENSEMBLE_FOLDS]
        required = [self.folder / "effective_config.yaml", *paths]
        missing = [p.relative_to(self.folder).as_posix() for p in required if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Missing or empty model files in {self.folder}: {', '.join(missing)}. "
                                    "Extract the bundled checkpoints using First-time setup in README.md.")
        return paths

    def _load(self):
        paths = self.checkpoint_paths()
        import torch
        # Limit CPU contention with the live camera.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        from dataset import build_transforms
        from model import build_model
        config = load_config(self.folder / "effective_config.yaml")
        training = config["training"]
        if training["architecture"] != "efficientnet_b0" or training["image_source"] != "with_roi":
            raise ValueError("Expected an EfficientNet-B0 ROI-trained experiment")
        if config["dataset"]["label_map"] != {"Non-anemic": 0, "Anemic": 1}:
            raise ValueError("Checkpoint label mapping does not match capture inference")
        models, provenance = [], []
        contracts = [path.with_name("checkpoint_contract.json") for path in paths]
        if any(p.exists() for p in contracts) and not all(p.exists() for p in contracts):
            raise ValueError("Checkpoint input contracts are incomplete; do not mix model versions")
        formats = set()
        for path, contract_path in zip(paths, contracts):
            digest = file_hash(path)
            if contract_path.exists():
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                if (contract.get("version") != 1 or contract.get("checkpoint_sha256") != digest
                        or contract.get("image_size") != training["image_size"]
                        or contract.get("label_map") != config["dataset"]["label_map"]
                        or contract.get("preprocessing") != "rgb_imagenet_resize_v1"):
                    raise ValueError("Checkpoint does not match its input contract")
                formats.add(contract.get("input_format"))
            model = build_model("efficientnet_b0", pretrained=False)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            models.append(model)
            provenance.append(dict(path=path.as_posix(), sha256=digest))
        if formats:
            if len(formats) != 1:
                raise ValueError("Fold models expect different ROI formats")
            self.input_format = formats.pop()
        if self.input_format not in ("rectangular", "masked"):
            raise ValueError("Model input must be rectangular or masked ROI")
        self.transform = build_transforms(training["image_size"], train=False)
        self.provenance = provenance
        self.models = models

    def _input_status(self, roi):
        if not self.enabled:
            return dict(status="disabled", reason="Model inference is disabled")
        if not roi or roi["status"] != "ok":
            return dict(status="skipped", reason="Retake: no usable ROI was extracted")
        return None

    def preview(self, roi, *, corrected=False):
        """Explain the next browser action without loading the model."""
        blocked = self._input_status(roi)
        if blocked:
            return blocked
        reason = ("Review the corrected mask, then confirm to run the model" if corrected
                  else "Check the tissue mask, correct it if needed, then confirm to run the model")
        return dict(status="needs_review", reason=reason)

    def predict(self, selection, *, allow_review=False):
        roi = selection.get("_roi_result")
        blocked = self._input_status(roi)
        if blocked:
            return blocked
        candidate = selection["candidates"][selection["best"]]
        quality_passed = bool(candidate["ok"])
        confirmed = bool(selection.get("roi_confirmed"))
        if not quality_passed and not (allow_review and confirmed):
            return dict(status="skipped", reason="Capture quality did not pass")
        if self.require_confirmation and not confirmed:
            return self.preview(roi)
        prediction = self.predict_roi(roi)
        if prediction["status"] in SCORED_STATUSES and not (quality_passed and confirmed):
            reason = ("The ROI was not confirmed" if not confirmed else "The capture failed the quality check")
            prediction = dict(prediction, folds_disagree=prediction["status"] == "uncertain",
                              status="review_only", reason=reason + ". Prediction is for review only.")
        return prediction

    def predict_roi(self, roi):
        """Score an extracted ROI for offline review without setting capture approval."""
        if not self.enabled:
            return dict(status="disabled", reason="Model inference is disabled")
        try:
            threshold = float(self.threshold)
            if not math.isfinite(threshold) or not 0 < threshold < 1:
                raise ValueError("Prediction threshold must be between 0 and 1")
            import cv2
            import torch
            from PIL import Image
            if self.models is None:
                self._load()
            source = "roi_masked" if self.input_format == "masked" else "roi"
            rgb = cv2.cvtColor(roi[source], cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0)
            with torch.inference_mode():
                scores = [float(torch.softmax(model(tensor), dim=1)[0, 1]) for model in self.models]
            if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
                raise ValueError("Model returned an invalid score")
            score = sum(scores) / len(scores)
            disagreement = min(scores) < threshold <= max(scores)
            label = "Anemic" if score >= threshold else "Non-anemic"
            return dict(status="uncertain" if disagreement else "ok",
                        label=label,
                        reason="Fold models disagree across the decision threshold" if disagreement else "",
                        anemia_score=score, threshold=threshold, fold_scores=scores, folds_disagree=disagreement,
                        score_spread=max(scores) - min(scores), calibrated=False,
                        model="efficientnet_b0", experiment=self.folder.name,
                        input=f"{source}.png (original colors)", input_format=self.input_format, checkpoints=self.provenance,
                        input_pixel_sha256=pixel_hash(roi[source]),
                        note="Research model prediction, not a diagnosis; scores are not calibrated clinical probabilities")
        except Exception as error:
            return dict(status="unavailable", reason=str(error))
