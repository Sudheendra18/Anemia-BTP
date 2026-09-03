"""
Stage 2 — Conjunctiva ROI extraction.

CP-AnemiC and smartphone conjunctiva images are close-up photos of a manually
pulled-down lower eyelid: mostly moist reddish/pink conjunctival tissue in the
center, framed by sclera, iris/pupil, skin, eyelashes, and flash glare.

This module provides an accurate, illumination-robust conjunctiva segmentation
and ROI extraction pipeline:
  1. Multi-space Chromaticity Analysis: Combines Lab a*, YCrCb Cr, HSV Red Hue
     and normalized excess red (R - G) to distinguish mucosal tissue from
     surrounding skin, sclera, iris, and glare.
  2. Eye-Zone Adaptive Thresholding: Dynamically calculates the mucosal redness
     threshold based on the chromatic distribution in the ocular region.
  3. Horizontal Crescent Morphological Bridging: Bridges vascular networks and
     mucosal folds across the horizontal palpebral arch.
  4. Anatomical & Geometric Component Scoring: Selects and merges components
     matching the horizontal aspect ratio, lower-central ocular position,
     and peak redness density of the palpebral conjunctiva.
  5. Tight Padded Bounding Box: Produces the rectangular crop (for standard CNN
     training) and the masked crop (pure mucosal tissue for color analysis).
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_config, resolve_path, setup_logging, ensure_parent_dir


# --------------------------------------------------------------------------
# Core segmentation algorithm
# --------------------------------------------------------------------------

def extract_roi(image_path: str, roi_cfg: dict | None = None) -> dict:
    """Run conjunctiva ROI extraction on a single image.

    Parameters:
      image_path: Path to the input image file.
      roi_cfg: Configuration dictionary (optional).

    Returns a dict with:
      status        'ok', 'fallback', or 'unreadable'
      roi           rectangular crop (original resolution BGR)
      roi_masked    same crop with non-conjunctiva pixels blacked out (BGR)
      mask          binary mask in the crop's coordinate frame (uint8, 0/255)
      bbox          (x0, y0, x1, y1) in original-image coordinates
      original      the original full-resolution image (BGR)
    """
    if roi_cfg is None:
        roi_cfg = {}

    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "status": "unreadable",
            "roi": None,
            "roi_masked": None,
            "mask": None,
            "bbox": None,
            "original": None,
        }

    orig_h, orig_w = img.shape[:2]

    # Standard processing resolution for speed, scale invariance, and noise suppression
    target_long = roi_cfg.get("resize_long_side", 800)
    scale = target_long / max(orig_h, orig_w) if target_long else 1.0
    pw, ph = int(orig_w * scale), int(orig_h * scale)
    small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()

    # 1. Color space transformations
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)

    _, a_ch, b_lab = cv2.split(lab)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    _, cr_ch, _ = cv2.split(ycrcb)

    b_f, g_f, r_f = cv2.split(small.astype(np.float32))
    sum_rgb = r_f + g_f + b_f + 1e-5
    r_norm = r_f / sum_rgb
    g_norm = g_f / sum_rgb
    rg_diff = r_norm - g_norm

    # 2. Eye Zone & Mucosal Chromaticity Filtering
    y_grid = np.arange(ph)[:, None]
    in_eye_zone = (y_grid >= int(0.20 * ph)) & (y_grid <= int(0.90 * ph))

    # Palpebral conjunctiva has Hue in red/magenta range and moderate-to-high saturation
    hsv_red = ((h_ch <= 26) | (h_ch >= 152)) & (s_ch >= 28)

    # Filter out specular glare highlights and dark shadows / eyelashes
    valid_lum = (v_ch >= 40) & ~((v_ch > 245) & (s_ch < 35))

    # Composite Mucosal Redness Index (MRI)
    # Combines Lab a* (redness), YCrCb Cr (chroma), normalized red-green contrast,
    # and HSV saturation, while subtracting epidermal yellowness (Lab b*).
    mri = (
        (a_ch.astype(np.float32) - 128.0) * 1.0
        + (cr_ch.astype(np.float32) - 128.0) * 1.0
        + (rg_diff * 80.0)
        + (s_ch.astype(np.float32) * 0.15)
        - ((b_lab.astype(np.float32) - 128.0) * 0.25)
    )
    mri = np.clip(mri, 0, None)

    # 3. Dynamic Thresholding
    eye_mri = mri[in_eye_zone & hsv_red & valid_lum]
    if len(eye_mri) > 50:
        peak_val = np.percentile(eye_mri, 95)
        thresh = max(30.0, min(peak_val * 0.52, np.percentile(eye_mri, 68)))
    else:
        thresh = 30.0

    raw_mask = (mri >= thresh) & hsv_red & valid_lum & in_eye_zone
    mask_u8 = raw_mask.astype(np.uint8) * 255

    # 4. Horizontal Crescent Morphological Bridging
    # The palpebral conjunctiva is a horizontal crescent spanning across the lower lid
    kw = max(9, int(pw * 0.045) | 1)
    kh = max(3, int(ph * 0.015) | 1)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_h)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 5. Connected Component Analysis with Geometric & Spatial Scoring
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

    best_mask = np.zeros_like(opened)
    status = "fallback"

    if n_lbl > 1:
        candidates = []
        min_area = roi_cfg.get("min_contour_area_frac", 0.003) * pw * ph

        for lbl in range(1, n_lbl):
            area = stats[lbl, cv2.CC_STAT_AREA]
            bx = stats[lbl, cv2.CC_STAT_LEFT]
            by = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[lbl]

            if area < min_area:
                continue

            blob_pix = (labels == lbl)
            mean_mri = np.mean(mri[blob_pix])
            aspect = bw / max(1, bh)
            rel_y = cy / ph

            # Anatomical position prior: lower-middle eye region
            if 0.35 <= rel_y <= 0.85:
                pos_weight = 1.0 - 1.5 * ((rel_y - 0.62) ** 2)
            else:
                pos_weight = 0.15

            aspect_weight = min(aspect, 3.5) if aspect >= 1.0 else (0.5 * aspect)
            score = area * pos_weight * aspect_weight * (mean_mri ** 1.3)
            candidates.append({
                "lbl": lbl, "area": area, "score": score,
                "bx": bx, "by": by, "bw": bw, "bh": bh, "cx": cx, "cy": cy
            })

        if candidates:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            primary = candidates[0]

            # Retain the primary component plus horizontally aligned crescent segments
            keep_lbls = [primary["lbl"]]
            prim_cy = primary["cy"]

            for c in candidates[1:]:
                # If within vertical alignment of the main crescent band
                if abs(c["cy"] - prim_cy) < 0.07 * ph:
                    keep_lbls.append(c["lbl"])

            for lbl in keep_lbls:
                best_mask[labels == lbl] = 255

            # Fill internal gaps & smooth boundaries
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                filled = np.zeros_like(best_mask)
                cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
                k_smooth = max(7, int(pw * 0.02) | 1)
                best_mask = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_smooth, k_smooth)))
                status = "ok"

    # 6. Bounding Box & Padded ROI Crop
    pad_w_frac = roi_cfg.get("bbox_padding_w_frac", 0.04)
    pad_h_frac = roi_cfg.get("bbox_padding_h_frac", 0.08)

    if status == "ok" and cv2.countNonZero(best_mask) > 0:
        ys, xs = np.where(best_mask > 0)
        sx0, sy0, sx1, sy1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        pw_pad = int((sx1 - sx0) * pad_w_frac)
        ph_pad = int((sy1 - sy0) * pad_h_frac)

        sx0 = max(0, sx0 - pw_pad)
        sy0 = max(0, sy0 - ph_pad)
        sx1 = min(pw, sx1 + pw_pad)
        sy1 = min(ph, sy1 + ph_pad)

        scale_back = 1.0 / scale
        ox0 = max(0, int(sx0 * scale_back))
        oy0 = max(0, int(sy0 * scale_back))
        ox1 = min(orig_w, int(sx1 * scale_back))
        oy1 = min(orig_h, int(sy1 * scale_back))
    else:
        status = "fallback"
        fb_frac = roi_cfg.get("fallback_center_crop_frac", 0.6)
        crop_w, crop_h = int(orig_w * fb_frac), int(orig_h * fb_frac)
        ox0 = (orig_w - crop_w) // 2
        oy0 = int(orig_h * 0.40)
        ox1 = min(orig_w, ox0 + crop_w)
        oy1 = min(orig_h, oy0 + crop_h)
        sx0, sy0, sx1, sy1 = 0, 0, pw, ph
        best_mask = np.full((ph, pw), 255, dtype=np.uint8)

    crop = img[oy0:oy1, ox0:ox1]
    sub_mask = best_mask[sy0:sy1, sx0:sx1] if status == "ok" else np.full((sy1 - sy0, sx1 - sx0), 255, dtype=np.uint8)
    crop_mask = cv2.resize(sub_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    roi_masked = crop.copy()
    roi_masked[crop_mask == 0] = 0

    return {
        "status": status,
        "bbox": (ox0, oy0, ox1, oy1),
        "roi": crop,
        "roi_masked": roi_masked,
        "mask": crop_mask,
        "original": img,
    }


def _save_debug_composite(result: dict, out_path: Path) -> None:
    """Save a 4-panel QA composite: original+bbox | mask | rect crop | masked crop."""
    orig = result["original"].copy()
    x0, y0, x1, y1 = result["bbox"]
    color = (0, 255, 0) if result["status"] == "ok" else (0, 0, 255)
    cv2.rectangle(orig, (x0, y0), (x1, y1), color, max(2, orig.shape[1] // 300))

    target_h = 300

    def _resize_h(im: np.ndarray, h: int) -> np.ndarray:
        if im is None or im.size == 0 or im.shape[0] == 0:
            return np.zeros((h, h, 3), dtype=np.uint8)
        scale = h / im.shape[0]
        return cv2.resize(im, (max(1, int(im.shape[1] * scale)), h))

    orig_r = _resize_h(orig, target_h)
    mask_bgr = cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR) if len(result["mask"].shape) == 2 else result["mask"]
    mask_r = _resize_h(mask_bgr, target_h)
    roi_r = _resize_h(result["roi"], target_h)
    roi_masked_r = _resize_h(result["roi_masked"], target_h)

    composite = np.hstack([orig_r, mask_r, roi_r, roi_masked_r])
    cv2.imwrite(str(out_path), composite)


def _process_one(task: dict) -> dict:
    """Worker function for parallel batch extraction."""
    result = extract_roi(task["image_path"], task["roi_cfg"])
    out = {"idx": task["idx"], "status": result["status"]}

    if result["status"] == "unreadable":
        out["roi_path"] = None
        out["roi_masked_path"] = None
        return out

    cv2.imwrite(task["out_path"], result["roi"])
    cv2.imwrite(task["masked_path"], result["roi_masked"])
    out["roi_path"] = task["out_path"]
    out["roi_masked_path"] = task["masked_path"]

    if task["debug_path"] is not None:
        _save_debug_composite(result, Path(task["debug_path"]))

    return out


def run_batch(config: dict, logger) -> pd.DataFrame:
    roi_cfg = config["roi_extraction"]
    metadata_csv = resolve_path(config["metadata_builder"]["output_csv"])

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {metadata_csv}. Run scripts/build_metadata.py first."
        )

    df = pd.read_csv(metadata_csv)
    logger.info(f"Loaded {len(df)} rows from {metadata_csv}")

    if len(df) == 0:
        logger.warning(
            "Metadata CSV has 0 rows — nothing to extract ROI for. This usually means "
            "Stage 1 found no images in the configured folders. Check `dataset.raw_images_dir` "
            "in the config, then re-run Stage 1 before Stage 2."
        )
        df["roi_image_path"] = pd.Series(dtype="object")
        df["roi_masked_path"] = pd.Series(dtype="object")
        df["roi_status"] = pd.Series(dtype="object")
        return df

    output_dir = resolve_path(roi_cfg["output_dir"])
    masked_dir = resolve_path(roi_cfg.get("masked_output_dir", str(output_dir) + "_masked"))
    debug_dir = resolve_path(roi_cfg["debug_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for label_text in df["label_text"].unique():
        (output_dir / label_text).mkdir(parents=True, exist_ok=True)
        (masked_dir / label_text).mkdir(parents=True, exist_ok=True)

    random.seed(config.get("random_seed", 42))
    debug_indices = set(random.sample(range(len(df)), min(roi_cfg.get("debug_sample_size", 20), len(df))))

    num_workers = roi_cfg.get("num_workers", max(1, (os.cpu_count() or 2) - 1))

    tasks = []
    for idx, row in df.iterrows():
        out_name = Path(row["image_path"]).name
        out_path = output_dir / row["label_text"] / out_name
        masked_path = masked_dir / row["label_text"] / out_name
        debug_path = debug_dir / f"debug_{out_name}" if idx in debug_indices else None
        tasks.append({
            "idx": idx,
            "image_path": row["image_path"],
            "out_path": str(out_path),
            "masked_path": str(masked_path),
            "debug_path": str(debug_path) if debug_path is not None else None,
            "roi_cfg": roi_cfg,
        })

    results_by_idx = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for out in tqdm(executor.map(_process_one, tasks), total=len(tasks), desc=f"Extracting ROI ({num_workers} workers)"):
            results_by_idx[out["idx"]] = out

    roi_paths, roi_masked_paths, statuses = [], [], []
    n_ok, n_fallback, n_unreadable = 0, 0, 0

    for idx, row in df.iterrows():
        out = results_by_idx[idx]
        roi_paths.append(out["roi_path"])
        roi_masked_paths.append(out["roi_masked_path"])
        statuses.append(out["status"])

        if out["status"] == "unreadable":
            n_unreadable += 1
            logger.warning(f"Could not read image: {row['image_path']}")
        elif out["status"] == "ok":
            n_ok += 1
        else:
            n_fallback += 1

    df["roi_image_path"] = roi_paths
    df["roi_masked_path"] = roi_masked_paths
    df["roi_status"] = statuses

    logger.info(f"ROI extraction complete: {n_ok} ok, {n_fallback} fallback, {n_unreadable} unreadable")
    if n_fallback > 0:
        fallback_rate = n_fallback / len(df)
        logger.warning(
            f"Fallback rate: {fallback_rate:.1%}. Check the composites saved in {debug_dir}"
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Extract conjunctiva ROI crops for every image in the metadata CSV.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(log_file="outputs/logs/roi_extraction.log")

    df = run_batch(config, logger)

    out_csv = ensure_parent_dir(config["roi_extraction"]["output_csv"])
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote metadata + ROI paths -> {out_csv}")


if __name__ == "__main__":
    main()