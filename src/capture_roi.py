"""Conjunctiva segmentation for detected eyes, rather than whole close-up photos.

The bulk dataset extractor assumes an already close-up eye. Capture crops can
also contain brows, cheeks and fingers, so constrain segmentation to the lower
lid first. Failure never produces an all-white mask or a successful center crop.
"""
from __future__ import annotations

import cv2
import numpy as np

VERSION = "lower_lid_chroma_v1"


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
    search = np.ones((sh, sw), bool)
    if eye.get("method") != "manual_region" and eye.get("oval"):
        oval, box = eye["oval"], eye["box"]
        cx, cy = (oval["cx"]-box[0])*scale, (oval["cy"]-box[1])*scale
        rx, ry = oval["rx"]*scale, oval["ry"]*scale
        angle = np.deg2rad(oval["angle"])
        yy, xx = np.mgrid[:sh, :sw]
        u = ((xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle))/max(rx, 1)
        v = (-(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle))/max(ry, 1)
        # This band excludes the brow above and the pulling fingers below.
        search = (np.abs(u) <= .90) & (v >= -.15) & (v <= .65)
    if eye.get("roi_search_mask") is not None:
        search &= cv2.resize(eye["roi_search_mask"], (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    if search.sum() < 80:
        return dict(failure, reason="Lower-lid search region is too small")

    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    a, b = lab[..., 1], lab[..., 2]
    # Lab a alone also selects warm skin. Subtract b (yellow) to distinguish
    # pink mucosa from cheek/finger skin, and exclude the low-chroma sclera.
    red_limit = max(135., float(np.percentile(a[search], 60)) + 1.)
    pink_limit = max(5., float(np.percentile((a-b)[search], 30)))
    saturation_limit = max(25., float(np.percentile(hsv[..., 1][search], 20)))
    glare = (hsv[..., 2] > 245) & (hsv[..., 1] < 35)
    valid = search & (hsv[..., 2] >= 40) & ~glare
    tissue = (valid & (a >= red_limit) & ((a-b) >= pink_limit)
              & (hsv[..., 1] >= saturation_limit))
    # Scale morphology to the actual search band, not the surrounding face.
    sy, sx = np.where(search)
    band_width = int(sx.max()-sx.min()+1)
    kw = max(3, round(band_width*.07) | 1)
    kh = max(3, round(band_width*.025) | 1)
    mask = cv2.morphologyEx(tissue.astype(np.uint8)*255, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask[~valid] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        # Reject specks and vertical skin fragments. Do not merge independent
        # components just because they happen to be on the same image row.
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
                reason="", tissue_pixels=int(len(xs)))
