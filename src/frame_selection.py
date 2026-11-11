"""Bounded-memory, one-pass eye-frame selection and live focus guidance.

Detector geometry comes from Task_B. Scores are capture-quality heuristics,
not a diagnosis or proof that the lower eyelid has been correctly exposed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np

from eye_detection import detect_eyes, oval_mask

CRITERIA = ("overall", "sharpness", "exposure", "glare")


@dataclass(frozen=True)
class CaptureConfig:
    analysis_width: int = 640
    stride: int = 2
    detect_interval: int = 5
    min_sharpness: float = 35.0
    brightness_min: float = 55.0
    brightness_max: float = 205.0
    max_glare: float = 0.04
    max_clipped: float = 0.15

    def __post_init__(self):
        if self.analysis_width < 160 or self.stride < 1 or self.detect_interval < 1:
            raise ValueError("analysis_width >= 160, stride >= 1 and detect_interval >= 1 required")
        values = (self.min_sharpness, self.brightness_min, self.brightness_max,
                  self.max_glare, self.max_clipped)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Quality thresholds must be finite")
        if self.min_sharpness <= 0 or not 0 <= self.brightness_min < self.brightness_max <= 255:
            raise ValueError("Invalid sharpness or brightness limits")
        if not 0 <= self.max_glare <= 1 or not 0 <= self.max_clipped <= 1:
            raise ValueError("Glare and clipping limits must be fractions")


def analysis_frame(frame, width):
    h, w = frame.shape[:2]
    scale = min(1.0, width / max(w, h))
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else frame


def quality(frame, eye, cfg):
    """Measure only the lower-lid band, at a fixed comparison resolution.

    Red coverage is reported for review but never used to gate/rank frames.
    Blur is measured inside an eroded mask to exclude artificial mask edges.
    """
    x0, y0, x1, y1 = map(int, eye["box"])
    patch = frame[y0:y1, x0:x1]
    mask = eye["roi_mask"][y0:y1, x0:x1]
    zero = dict(score=0., coverage=0., sharp=0., brightness=0., glare=1.,
                clipped=1., ok=False, valid=False, feedback="Move closer: eye region is too small")
    if patch.size == 0 or min(patch.shape[:2]) < 12 or cv2.countNonZero(mask) < 100:
        return zero
    # Normalize large eye regions; never magnify tiny regions to manufacture detail.
    scale = min(1., 160 / patch.shape[1])
    size = (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))
    patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    inside = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    if inside.sum() < 64:
        return zero
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    # Slight denoising avoids rewarding high-frequency sensor noise as focus.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0.6)
    sharp = float(cv2.Laplacian(denoised, cv2.CV_32F)[inside].var())
    brightness = float(gray[inside].mean())
    glare = float(((hsv[..., 2] > 240) & (hsv[..., 1] < 40))[inside].mean())
    clipped = float(((gray < 10) | (gray > 245))[inside].mean())
    coverage = float(((lab[..., 1] > 145) & (lab[..., 0] > 40))[inside].mean())
    target = (cfg.brightness_min + cfg.brightness_max) / 2
    focus = sharp / (sharp + cfg.min_sharpness)
    exposure = max(0., 1 - abs(brightness - target) / 128)
    score = float(focus * exposure * (1 - glare) ** 3 * (1 - clipped) ** 2)
    issues = []
    if sharp < cfg.min_sharpness:
        issues.append("Hold steady; wait for camera focus")
    if brightness < cfg.brightness_min:
        issues.append("Increase diffuse light")
    if brightness > cfg.brightness_max:
        issues.append("Reduce light intensity")
    if glare > cfg.max_glare:
        issues.append("Change angle to reduce glare")
    if clipped > cfg.max_clipped:
        issues.append("Avoid deep shadows and clipped highlights")
    return dict(score=score, coverage=coverage, sharp=sharp, brightness=brightness,
                glare=glare, clipped=clipped, valid=True, ok=not issues,
                feedback="; ".join(issues) or "Focus looks good; check lower-lid visibility")


def _distance(a, b):
    return np.hypot(a["oval"]["cx"] - b["oval"]["cx"],
                    a["oval"]["cy"] - b["oval"]["cy"])


class RegionDetector:
    """Operator-defined fixed lower-lid region for extreme close-up captures.

    Coordinates are normalized x, y, width, height in the source image. The
    operator holds the tissue within this region; this is not an eye detector.
    """
    fixed_region = True

    def __init__(self, region):
        if len(region) != 4 or not all(np.isfinite(v) for v in region):
            raise ValueError("eye-region needs four finite values: x,y,width,height")
        x, y, w, h = map(float, region)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("eye-region must lie within normalized image coordinates 0..1")
        self.region = x, y, w, h

    def __call__(self, frame, allow_guess=False):
        h, w = frame.shape[:2]
        x, y, rw, rh = self.region
        x0, y0 = round(x*w), round(y*h)
        x1, y1 = min(w, round((x+rw)*w)), min(h, round((y+rh)*h))
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 255
        return [dict(box=(x0, y0, x1, y1), roi_mask=mask, ref_mask=None,
                     oval=dict(cx=(x0+x1)/2, cy=(y0+y1)/2, rx=(x1-x0)/2, ry=(y1-y0)/2, angle=0.),
                     side="selected", method="manual_region")]


class EyeTracker:
    """Periodic detection with validated forward/backward optical flow between.

    Keep the same eye by proximity rather than swapping when redness changes.
    Failed flow immediately triggers detection; stale geometry is never scored.
    """
    def __init__(self, cfg=None, detector=None):
        self.cfg = cfg or CaptureConfig()
        self.detector = detector or detect_eyes
        self.reset()

    def reset(self):
        self.previous = None
        self.eye = None
        self.points = None
        self.age = 0
        self.detection_calls = 0
        self.flow_frames = 0

    def _flow(self, gray):
        if self.points is None or len(self.points) < 6 or self.previous.shape != gray.shape:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None)
        if nxt is None:
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None)
        if back is None:
            return None
        good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                & (np.linalg.norm(back - self.points, axis=2).ravel() < 1.5))
        if good.sum() < 6 or good.mean() < .6:
            return None
        displacement = (nxt - self.points).reshape(-1, 2)[good]
        dx, dy = np.median(displacement, axis=0)
        if np.median(np.linalg.norm(displacement - (dx, dy), axis=1)) > 2.5:
            return None
        if np.hypot(dx, dy) > self.eye["oval"]["rx"] * .6:
            return None
        h, w = gray.shape
        x0, y0, x1, y1 = self.eye["box"]
        box = tuple(int(round(v)) for v in (x0 + dx, y0 + dy, x1 + dx, y1 + dy))
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
            return None
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        eye = dict(self.eye, box=box, oval=dict(self.eye["oval"]))
        eye["oval"]["cx"] += float(dx)
        eye["oval"]["cy"] += float(dy)
        for key in ("roi_mask", "ref_mask"):
            if eye.get(key) is not None:
                eye[key] = cv2.warpAffine(eye[key], transform, (w, h), flags=cv2.INTER_NEAREST)
        self.points = nxt[good].reshape(-1, 1, 2)
        return eye

    def update(self, frame):
        small = analysis_frame(frame, self.cfg.analysis_width)
        if getattr(self.detector, "fixed_region", False):
            eye = self.detector(small)[0]
            self.eye = eye
            return small, eye, quality(small, eye, self.cfg)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        eye = None
        if self.eye is not None and self.age < self.cfg.detect_interval - 1:
            eye = self._flow(gray)
        if eye is None:
            self.detection_calls += 1
            eyes = self.detector(small, allow_guess=False)
            if eyes:
                measured = [(e, quality(small, e, self.cfg)) for e in eyes]
                valid = [(e, q) for e, q in measured if q["valid"]]
                pool = valid or measured
                closest = min(pool, key=lambda eq: _distance(eq[0], self.eye)) if self.eye else None
                if closest and _distance(closest[0], self.eye) < self.eye["oval"]["rx"] * 1.5:
                    eye = closest[0]
                else:
                    eye = max(pool, key=lambda eq: (eq[1]["ok"], eq[1]["score"]))[0]
                feature_mask = np.zeros_like(gray)
                x0, y0, x1, y1 = eye["box"]
                feature_mask[y0:y1, x0:x1] = 255
                self.points = cv2.goodFeaturesToTrack(gray, 60, .01, 5, mask=feature_mask)
            else:
                self.points = None
            self.age = 0
        else:
            self.flow_frames += 1
            self.age += 1
        self.eye, self.previous = eye, gray
        return small, eye, quality(small, eye, self.cfg) if eye else None

    def live(self, frame):
        small, eye, q = self.update(frame)
        if eye is None:
            return dict(found=False, feedback="Searching for an eye; adjust distance and angle")
        return dict(found=True, w=small.shape[1], h=small.shape[0], **eye["oval"],
                    ok=q["ok"], coverage=q["coverage"], side=eye["side"],
                    method=eye["method"], feedback=q["feedback"])


def _key(name, q, cfg):
    # Every criterion prioritizes passing quality gates, not just its own metric.
    target = (cfg.brightness_min + cfg.brightness_max) / 2
    value = {"overall": q["score"], "sharpness": q["sharp"],
             "exposure": -abs(q["brightness"] - target),
             "glare": -round(q["glare"], 3)}[name]
    return q["ok"], value, q["score"]


def _candidate(frame, small, eye, q, idx):
    result = dict(frame=frame.copy(), idx=idx, sharpness=q["sharp"],
                  brightness=q["brightness"], glare=q["glare"], coverage=q["coverage"],
                  score=q["score"], clipped=q["clipped"], feedback=q["feedback"], ok=q["ok"], eye=None)
    if eye:
        sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
        box = tuple(int(round(v * (sx if i % 2 == 0 else sy))) for i, v in enumerate(eye["box"]))
        x0, y0, x1, y1 = box
        oval = dict(eye["oval"])
        for key in ("cx", "rx"):
            oval[key] *= sx
        for key in ("cy", "ry"):
            oval[key] *= sy
        result["eye"] = dict(side=eye["side"], method=eye["method"], box=box,
                             oval=oval, conj=q, crop=result["frame"][y0:y1, x0:x1])
        full_mask = cv2.resize(eye["roi_mask"], (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        result["eye"]["roi_search_mask"] = full_mask[y0:y1, x0:x1]
    return result


def select_best_frame(video_path, cfg=None, detector=None):
    """Decode once, retain at most four full-resolution candidate frames.

    Skipped frames use grab() without conversion. No frame list or second decode
    pass is needed. A separate tracker prevents live-preview state leaking in.
    """
    cfg = cfg or CaptureConfig()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    tracker = EyeTracker(cfg, detector)
    winners, keys = {}, {}
    n = analyzed = 0
    has_eye = False
    started = perf_counter()
    try:
        while cap.grab():
            n += 1
            if (n - 1) % cfg.stride:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            analyzed += 1
            small, eye, q = tracker.update(frame)
            usable = eye is not None and q["valid"]
            if usable and not has_eye:
                winners.clear()
                keys.clear()
                has_eye = True
            if not usable:
                if has_eye:
                    continue
                eye = None
                h, w = small.shape[:2]
                q = quality(small, dict(box=(0, 0, w, h), roi_mask=oval_mask(h, w)), cfg)
                q["ok"] = False
            candidate = None
            for name in CRITERIA:
                key = _key(name, q, cfg)
                if name not in keys or key > keys[name]:
                    if candidate is None:
                        candidate = _candidate(frame, small, eye, q, n)
                    winners[name], keys[name] = candidate, key
    finally:
        cap.release()
    if not winners:
        raise RuntimeError("No frames could be decoded from the recording")
    return dict(n_frames=n, analyzed_frames=analyzed, candidates=winners, best="overall",
                eye_found=has_eye, elapsed_seconds=perf_counter() - started,
                detection_calls=tracker.detection_calls, tracked_frames=tracker.flow_frames,
                quality_config=asdict(cfg),
                retained_frames=len({id(c["frame"]) for c in winners.values()}))
