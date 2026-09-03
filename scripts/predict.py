"""
Direct Inference / Prediction Script

Run your trained EfficientNet model directly on ANY folder of images (or a single image)
without needing to organize them into subfolders or provide labels.

Outputs:
  - Prints predicted class (Anemic vs Non-anemic) & confidence probability for each image.
  - Generates a summary count (e.g. 15 Anemic, 10 Non-anemic).
  - Saves a CSV with image names, paths, predicted probabilities, and classifications.

Usage examples:
    # 1. Run on a folder of cropped conjunctiva images:
    python scripts/predict.py --input path/to/your/images

    # 2. Run on a folder of raw uncropped eye photos (with automatic ROI extraction):
    python scripts/predict.py --input path/to/your/images --extract-roi

    # 3. Run on a single image:
    python scripts/predict.py --input path/to/image.jpg

    # 4. Use custom threshold or output location:
    python scripts/predict.py --input path/to/your/images --threshold 0.58 --output outputs/my_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import build_model
from dataset import IMAGENET_MEAN, IMAGENET_STD
from roi_extraction import extract_roi
from metrics import apply_threshold
from utils import load_config, resolve_path, setup_logging


class DirectInferenceDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        image_size: int = 224,
        do_extract_roi: bool = False,
        roi_cfg: dict | None = None,
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        self.do_extract_roi = do_extract_roi
        self.roi_cfg = roi_cfg or {}
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        roi_status = "original"
        if self.do_extract_roi:
            roi_res = extract_roi(str(img_path), self.roi_cfg)
            if roi_res["status"] != "unreadable" and roi_res.get("roi") is not None:
                img_bgr = roi_res["roi"]
                roi_status = roi_res["status"]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil)

        return tensor, img_path.name, str(img_path.resolve()), roi_status


def load_ensemble_models(model_dir: Path, fold: int | None, architecture: str, num_classes: int, device: torch.device):
    models = []
    if fold is not None:
        ckpt_path = model_dir / f"fold_{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = build_model(architecture, num_classes=num_classes, pretrained=False)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append((f"fold_{fold}", model))
    else:
        fold_dirs = sorted(model_dir.glob("fold_*"))
        for f_dir in fold_dirs:
            ckpt_path = f_dir / "best_model.pt"
            if ckpt_path.exists():
                model = build_model(architecture, num_classes=num_classes, pretrained=False)
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                models.append((f_dir.name, model))
        if not models:
            raise FileNotFoundError(f"No fold checkpoints found under: {model_dir}")
    return models


def find_images(input_path: Path, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        return []
    elif input_path.is_dir():
        return sorted([
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ])
    return []


def main():
    parser = argparse.ArgumentParser(description="Predict anemic vs non-anemic on images without folder categorization.")
    parser.add_argument("--input", required=True, help="Path to image file OR folder of images.")
    parser.add_argument("--output", default="outputs/predictions/predictions.csv", help="Output CSV path. Default: outputs/predictions/predictions.csv")
    parser.add_argument("--model-dir", default="outputs/experiments/efficientnet_b0_lr1e-4", help="Path to experiment folder containing fold checkpoints.")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold (1-5) to use. Default: None (5-fold ensemble).")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for Anemic classification (default: 0.50).")
    parser.add_argument("--extract-roi", action="store_true", help="Automatically crop conjunctiva region before prediction (for full eye photos).")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML for settings.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_csv = resolve_path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=str(output_csv.parent / "inference.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Discover image(s)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return

    exts = {e.lower() for e in config["dataset"]["image_extensions"]}
    image_paths = find_images(input_path, exts)

    if not image_paths:
        logger.error(f"No valid image files found in: {input_path} (looked for {sorted(exts)})")
        return

    logger.info(f"Found {len(image_paths)} image(s) to process.")

    # 2. Dataset & DataLoader
    img_size = config["training"].get("image_size", 224)
    roi_cfg = config.get("roi_extraction", {})
    ds = DirectInferenceDataset(
        image_paths=image_paths,
        image_size=img_size,
        do_extract_roi=args.extract_roi,
        roi_cfg=roi_cfg,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. Load Model(s)
    model_dir = resolve_path(args.model_dir)
    arch = config["training"].get("architecture", "efficientnet_b0")
    num_classes = config["training"].get("num_classes", 2)
    models = load_ensemble_models(model_dir, args.fold, arch, num_classes, device)
    logger.info(f"Loaded {'ensemble of ' + str(len(models)) + ' models' if args.fold is None else f'fold {args.fold}'} from {model_dir.name}")

    # 4. Run inference
    all_names = []
    all_paths = []
    all_probs = []
    all_roi_status = []

    with torch.no_grad():
        for images, names, paths, roi_statuses in loader:
            images = images.to(device, non_blocking=True)

            batch_probs = torch.zeros(images.size(0), device=device)
            for _, model in models:
                logits = model(images)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]  # P(Anemic)
                batch_probs += probs

            batch_probs /= len(models)

            all_probs.extend(batch_probs.cpu().numpy().tolist())
            all_names.extend(names)
            all_paths.extend(paths)
            all_roi_status.extend(roi_statuses)

    y_prob = np.array(all_probs)
    y_pred = apply_threshold(y_prob, args.threshold)

    results_df = pd.DataFrame({
        "image_name": all_names,
        "image_path": all_paths,
        "prob_anemic": np.round(y_prob, 4),
        "confidence_percent": np.round(y_prob * 100, 1),
        "prediction": ["Anemic" if p == 1 else "Non-anemic" for p in y_pred],
        "roi_extraction": all_roi_status,
    })

    results_df.to_csv(output_csv, index=False)

    # 5. Display Summary
    n_anemic = int((y_pred == 1).sum())
    n_non_anemic = int((y_pred == 0).sum())

    print("\n" + "=" * 65)
    print(f"PREDICTION RESULTS (Threshold = {args.threshold:.2f})")
    print("=" * 65)
    print(f"{'Image Name':<30} {'Probability (Anemic)':<22} {'Prediction'}")
    print("-" * 65)
    for idx, row in results_df.head(20).iterrows():
        prob_str = f"{row['prob_anemic']:.4f} ({row['confidence_percent']}%)"
        print(f"{row['image_name']:<30} {prob_str:<22} {row['prediction']}")

    if len(results_df) > 20:
        print(f"... and {len(results_df) - 20} more images.")

    print("=" * 65)
    print(f"SUMMARY: Total: {len(results_df)} | Anemic: {n_anemic} ({n_anemic/len(results_df)*100:.1f}%) | Non-anemic: {n_non_anemic} ({n_non_anemic/len(results_df)*100:.1f}%)")
    print(f"Detailed CSV saved -> {output_csv}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
