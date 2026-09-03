"""
PyTorch Dataset + transforms for CP-AnemiC.

Reads directly from the metadata DataFrame produced by dataset_loader.py /
roi_extraction.py — no separate on-disk manifest needed.
"""

from __future__ import annotations

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ConjunctivaDataset(Dataset):
    """Wraps a metadata DataFrame slice. `image_col` picks which path column
    to read from (`image_path` for raw images, `roi_image_path` for crops)."""

    def __init__(self, df, image_col: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at {img_path} (row image_id={row.get('image_id')})")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        if self.transform is not None:
            img = self.transform(img)

        label = int(row["label"])
        return img, label


def build_transforms(image_size: int, train: bool, augmentation: dict | None = None) -> transforms.Compose:
    """Train gets light augmentation appropriate for close-up clinical photos
    (flips/rotation/color jitter — no crops that could cut into the already-
    tight conjunctiva ROI, and no vertical flip since these images have a
    consistent up/down orientation unlike a generic photo). Val/test is
    deterministic resize + normalize.

    `augmentation` (only used when train=True) lets experiments tune
    regularization strength without touching this file — see
    `training.augmentation` in the config. Defaults match the original
    fixed values if not provided."""
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    aug = augmentation or {}
    rotation_degrees = aug.get("rotation_degrees", 15)
    color_jitter = aug.get("color_jitter", 0.2)
    random_erasing_prob = aug.get("random_erasing_prob", 0.0)

    tf_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if random_erasing_prob > 0:
        # Must come after ToTensor — RandomErasing operates on tensors, not
        # PIL images. Blanks out a random rectangular patch each draw, which
        # discourages the model from over-relying on any single localized
        # region of the crop — a standard, well-established regularizer.
        tf_list.append(transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.15)))

    return transforms.Compose(tf_list)
