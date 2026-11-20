"""Eye detectors adapted from Subhankar191/Task_B.
Upstream commit b11e8467ebc2295ad50e6cfab0f6c31ed9c9d99a; see docs/CAPTURE.md.
"""
import cv2
import numpy as np
try:
    import mediapipe as mp
    _HAS_MP = hasattr(mp, "solutions")
except Exception:
    mp = None
    _HAS_MP = False

OVAL_RX, OVAL_RY = 0.30, 0.25

CONJ_BAND_FRAC = 0.50
CROP_PAD = 0.30
EYE_OVAL_SCALE = 1.15
OVAL_BAND_FRAC = 0.35

# MediaPipe FaceMesh landmark indices (image-left eye = subject's right eye)
EYES_MP = {
    "image_left": dict(lower=[33, 7, 163, 144, 145, 153, 154, 155, 133],
                       upper=[246, 161, 160, 159, 158, 157, 173], corners=(33, 133)),
    "image_right": dict(lower=[362, 382, 381, 380, 374, 373, 390, 249, 263],
                        upper=[466, 388, 387, 386, 385, 384, 398], corners=(362, 263)),
}
_face_mesh = None
_haar = None
_mp_warned = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_OK = (118, 230, 0)
COL_WARN = (40, 202, 255)


def oval_mask(h, w):
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(OVAL_RX * w), int(OVAL_RY * h)), 0, 0, 360, 255, -1)
    return mask


def _padded_box(x0, y0, x1, y1, w, h, pad=CROP_PAD):
    bw, bh = x1 - x0, y1 - y0
    return (int(max(0, x0 - pad * bw)), int(max(0, y0 - pad * bh)),
            int(min(w, x1 + pad * bw)), int(min(h, y1 + pad * bh)))


def _oval_from_points(pts, angle_deg=0.0, scale=EYE_OVAL_SCALE):
    """
    Oval (cx, cy, rx, ry, angle in degrees) enclosing the points, aligned with the eye axis.
    Same convention as cv2.ellipse / canvas ellipse (image coordinates, y down).
    """
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    a = np.deg2rad(angle_deg)
    u = np.array([np.cos(a), np.sin(a)], np.float32)
    v = np.array([-np.sin(a), np.cos(a)], np.float32)
    pu, pv = pts @ u, pts @ v
    u0, u1, v0, v1 = float(pu.min()), float(pu.max()), float(pv.min()), float(pv.max())
    centre = ((u0 + u1) / 2) * u + ((v0 + v1) / 2) * v
    return dict(cx=float(centre[0]), cy=float(centre[1]),
                rx=float(max(4.0, (u1 - u0) / 2 * scale)),
                ry=float(max(4.0, (v1 - v0) / 2 * scale)),
                angle=float(angle_deg))


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.3)
    return _face_mesh


def eyes_mediapipe(frame):
    """Precise eyelid landmarks. Needs most of a face in view. ROI follows head tilt."""
    h, w = frame.shape[:2]
    res = _get_face_mesh().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return []
    lm = res.multi_face_landmarks[0].landmark

    def pts(idx):
        return np.array([[lm[i].x * w, lm[i].y * h] for i in idx], np.float32)

    eyes = []
    for side, cfg in EYES_MP.items():
        lower, upper = pts(cfg["lower"]), pts(cfg["upper"])
        c = pts(cfg["corners"])
        axis = c[1] - c[0]
        eye_w = float(np.linalg.norm(axis))
        if eye_w < 1:
            continue
        axis /= eye_w
        normal = np.array([-axis[1], axis[0]], np.float32)
        if normal[1] < 0:
            normal = -normal
        off = normal * (CONJ_BAND_FRAC * eye_w)
        # Sample the strip below the lower lid.
        poly = np.vstack([lower, (lower + off)[::-1]])
        allp = np.vstack([lower, upper, poly])
        box = _padded_box(allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max(), w, h)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        roi = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi, [poly.astype(np.int32)], 255)
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        oval_pts = np.vstack([lower, upper, lower + off * OVAL_BAND_FRAC])
        eyes.append(dict(side=side, method="mediapipe", roi_mask=roi, box=box,
                         oval=_oval_from_points(oval_pts, angle)))
    return eyes


def eyes_haar(frame):
    """Fallback: OpenCV Haar eye detector; ROI = region just below each detected eye."""
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    h, w = frame.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    found = _haar.detectMultiScale(gray, 1.1, 6, minSize=(max(12, w // 20), max(12, w // 20)))
    if len(found) == 0:
        return []
    found = sorted(found, key=lambda r: r[2] * r[3], reverse=True)[:2]
    found = sorted(found, key=lambda r: r[0])
    eyes = []
    for i, (x, y, bw, bh) in enumerate(found):
        roi = np.zeros((h, w), np.uint8)
        y0, y1 = int(y + 0.55 * bh), int(min(h, y + 1.5 * bh))
        cv2.rectangle(roi, (x, y0), (x + bw, y1), 255, -1)
        box = _padded_box(x, y, x + bw, y1, w, h)
        side = ("image_left" if i == 0 else "image_right") if len(found) == 2 else "single"
        oval = dict(cx=float(x + bw / 2), cy=float(y + 0.60 * bh),
                    rx=float(0.58 * bw), ry=float(0.42 * bw), angle=0.0)
        eyes.append(dict(side=side, method="haar", roi_mask=roi, box=box, oval=oval))
    return eyes


def detect_eyes(frame):
    global _mp_warned
    if _HAS_MP:
        try:
            eyes = eyes_mediapipe(frame)
            if eyes:
                return eyes
        except Exception as e:
            if not _mp_warned:
                print(f"[warn] mediapipe failed: {e}")
                _mp_warned = True
    return eyes_haar(frame)
