"""Decode and encode images without passing Unicode paths to OpenCV on Windows."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags) if data else None


def write_image(path, image):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())
