"""Extract conjunctiva tissue within the lower-lid region of a detected eye."""
from __future__ import annotations

import cv2
import numpy as np
from frame_selection import CaptureConfig, quality

VERSION = "lower_lid_chroma_v3"


def _search_region(eye, height, width):
    """Evaluate geometry in source coordinates at either segmentation scale."""
    source_h, source_w = eye["crop"].shape[:2]
    search = np.ones((height, width), bool)
    if eye.get("method") != "manual_region" and eye.get("oval"):
        oval, box = eye["oval"], eye["box"]
        yy, xx = np.mgrid[:height, :width]
        xx = (xx + .5) * source_w / width - .5 - (oval["cx"] - box[0])
        yy = (yy + .5) * source_h / height - .5 - (oval["cy"] - box[1])
        angle = np.deg2rad(oval["angle"])
        u = (xx*np.cos(angle) + yy*np.sin(angle))/max(oval["rx"], 1)
        v = (-xx*np.sin(angle) + yy*np.cos(angle))/max(oval["ry"], 1)
        search = (np.abs(u) <= .90) & (v >= -.15) & (v <= .65)
    if eye.get("roi_search_mask") is not None:
        search &= cv2.resize(eye["roi_search_mask"], (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return search


def choose_capture_roi(result, config=None):
    """Select and cache the best retained ROI using tissue quality and the original quality gate."""
    candidates = result["candidates"]
    pool = dict(candidates)
    for candidate in result.pop("_shortlist", []):
        if not any(c["idx"] == candidate["idx"] for c in pool.values()):
            pool[f"alternative_{candidate['idx']}"] = candidate
    original = result["best"]
    alternatives = sorted((name for name in pool if name != original),
                          key=lambda name: (pool[name]["ok"], pool[name]["score"]),
                          reverse=True)
    cache, checked = {}, {}
    selected, selected_roi, original_roi = original, None, None
    best_key = None
    cfg = CaptureConfig(**result.get("quality_config", {}))
    for name in [original, *alternatives]:
        candidate = pool[name]
        eye = candidate["eye"]
        if eye is None:
            checked[name] = dict(frame_idx=candidate["idx"], status="no_eye")
            continue
        key = (candidate["idx"], tuple(eye["box"]))
        if key not in cache:
            roi = extract_capture_roi(eye, config)
            tissue = tissue_quality(roi, cfg) if roi["status"] == "ok" else None
            cache[key] = roi, tissue
        roi, tissue = cache[key]
        checked[name] = dict(frame_idx=candidate["idx"], status=roi["status"])
        if name == original:
            original_roi = roi
        if tissue is not None:
            checked[name]["tissue_quality"] = tissue
            rank = (bool(candidate["ok"] and tissue["ok"]), tissue["score"], candidate["score"])
            if best_key is None or rank > best_key:
                selected, selected_roi, best_key = name, roi, rank
    if selected_roi is None:
        selected_roi = original_roi
    if selected_roi and selected_roi["status"] == "ok":
        candidate = dict(pool[selected])
        candidate["tissue_quality"] = checked[selected]["tissue_quality"]
        candidate["ok"] = bool(candidate["ok"] and candidate["tissue_quality"]["ok"])
        if not candidate["tissue_quality"]["ok"]:
            candidate["feedback"] = candidate["tissue_quality"]["feedback"]
        candidates[selected] = candidate
    result["best"] = selected
    result["_roi_result"] = selected_roi
    result["roi_selection"] = dict(quality_best=original, selected=selected,
                                    fallback_used=selected != original, checked=checked)
    return result


def tissue_quality(roi, cfg=None):
    """Check real tissue pixels; mask edges never contribute to focus."""
    image, mask = roi["roi"], roi["mask"]
    h, w = image.shape[:2]
    measured = quality(image, dict(box=(0, 0, w, h), roi_mask=mask), cfg or CaptureConfig())
    measured["native_pixels"] = int(np.count_nonzero(mask))
    measured["native_size"] = [w, h]
    if min(h, w) < 16 or measured["native_pixels"] < 256:
        measured.update(ok=False, feedback="Move closer: too few original tissue pixels")
    return measured


def manual_capture_roi(eye, points):
    """An operator-drawn tissue boundary includes pale pixels without color gates."""
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 200:
        raise ValueError("Mark between 3 and 200 points around the exposed tissue")
    if not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
        raise ValueError("Tissue outline must stay within the eye crop")
    image = eye["crop"]
    h, w = image.shape[:2]
    polygon = np.rint(polygon * [w-1, h-1]).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    ys, xs = np.where(mask > 0)
    if len(xs) < 256:
        raise ValueError("Tissue outline is too small; select more exposed tissue or retake closer")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    crop, mask = image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    return dict(status="ok", roi=crop, roi_masked=cv2.bitwise_and(crop, crop, mask=mask),
                mask=mask, bbox=(x0,y0,x1,y1), original=image, method="manual_polygon_v1",
                reason="", tissue_pixels=len(xs), boundary_refined=False,
                polygon_normalized=np.asarray(points).tolist())


def extract_capture_roi(eye, config=None):
    config = config or {}
    image = eye["crop"]
    h, w = image.shape[:2]
    failure = dict(status="failed", roi=None, roi_masked=None, mask=None,
                   bbox=None, original=image, method=VERSION,
                   reason="No reliable conjunctiva component in the lower-lid region")
    if min(h, w) < 12:
        return dict(failure, reason="Eye crop is too small for tissue segmentation")
    target = config.get("resize_long_side", 800) or max(h, w)
    scale = min(1., float(target) / max(h, w))
    sw, sh = max(1, round(w*scale)), max(1, round(h*scale))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    search = _search_region(eye, sh, sw)
    if search.sum() < 80:
        return dict(failure, reason="Lower-lid search region is too small")

    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    a, b = lab[..., 1], lab[..., 2]
    # Subtract Lab yellowness to distinguish pink tissue from warm skin.
    # Inclusive percentile thresholds preserve uniform tissue.
    red_limit = max(135., float(np.percentile(a[search], 60)))
    pink_limit = max(5., float(np.percentile((a-b)[search], 30)))
    saturation_limit = max(25., float(np.percentile(hsv[..., 1][search], 20)))
    glare = (hsv[..., 2] > 245) & (hsv[..., 1] < 35)
    # Require red hues to exclude iris pixels that pass the Lab test.
    red_family = (hsv[..., 0] <= 25) | (hsv[..., 0] >= 150)
    valid = search & (hsv[..., 2] >= 40) & ~glare & red_family
    tissue = (valid & (a >= red_limit) & ((a-b) >= pink_limit)
              & (hsv[..., 1] >= saturation_limit))
    _, sx = np.where(search)
    band_width = int(sx.max()-sx.min()+1)
    kw = max(3, round(band_width*.07) | 1)
    kh = max(3, round(band_width*.025) | 1)
    mask = cv2.morphologyEx(tissue.astype(np.uint8)*255, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask[~valid] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in range(1, count):
        _, _, bw, bh, area = stats[label]
        if area < max(24, .015*search.sum()) or bw < .20*band_width or bw < bh:
            continue
        density = area / (bw*bh)
        if density < .12:
            continue
        candidates.append((area*min(bw/max(bh, 1), 4), label))
    if not candidates:
        return failure
    chosen = max(candidates)[1]
    mask = (labels == chosen).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    mask[~valid] = 0
    full_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Refine the boundary at source resolution to avoid upsampling artifacts.
    radius = max(1, int(np.ceil(1 / scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    interior = cv2.erode(full_mask, kernel) > 0
    vicinity = cv2.dilate(full_mask, kernel) > 0
    native_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    native_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    native_valid = (_search_region(eye, h, w) & (native_hsv[..., 2] >= 40)
                    & ((native_hsv[..., 0] <= 25) | (native_hsv[..., 0] >= 150))
                    & ~((native_hsv[..., 2] > 245) & (native_hsv[..., 1] < 35)))
    edge_tissue = ((native_lab[..., 1] >= max(135., red_limit-3))
                   & ((native_lab[..., 1]-native_lab[..., 2]) >= max(5., pink_limit-2))
                   & (native_hsv[..., 1] >= max(25., saturation_limit-5)))
    full_mask = ((interior | (vicinity & edge_tissue)) & native_valid).astype(np.uint8)*255
    ys, xs = np.where(full_mask > 0)
    if len(xs) < 24:
        return failure
    pad_x = max(1, round((xs.max()-xs.min()+1)*.04))
    pad_y = max(1, round((ys.max()-ys.min()+1)*.08))
    x0, y0 = max(0, int(xs.min())-pad_x), max(0, int(ys.min())-pad_y)
    x1, y1 = min(w, int(xs.max())+1+pad_x), min(h, int(ys.max())+1+pad_y)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = full_mask[y0:y1, x0:x1]
    masked = cv2.bitwise_and(crop, crop, mask=crop_mask)
    return dict(status="ok", roi=crop, roi_masked=masked, mask=crop_mask,
                bbox=(x0, y0, x1, y1), original=image, method=VERSION,
                reason="", tissue_pixels=int(len(xs)), boundary_refined=True)
