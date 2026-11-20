"""Identify the exact pixels supplied to capture inference."""
import hashlib

import cv2
import numpy as np


def pixel_hash(image):
    pixels = np.ascontiguousarray(image)
    header = f"{pixels.dtype}:{pixels.shape}:BGR:".encode("ascii")
    return hashlib.sha256(header + pixels.tobytes()).hexdigest()


def prediction_input_path(folder, prediction):
    name = {"rectangular": "roi.png", "masked": "roi_masked.png"}.get(prediction.get("input_format"))
    if name is None:
        raise ValueError("Unknown prediction input format")
    return folder / name


def read_prediction_input(folder, prediction, *, require_file_hash=True):
    """Check the saved PNG and return the exact pixels used for the prediction."""
    path = prediction_input_path(folder, prediction)
    data = path.read_bytes() if path.is_file() else b""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None or pixel_hash(image) != prediction.get("input_pixel_sha256"):
        raise ValueError("Saved image does not match the prediction; rerun inference on the current image")
    digest = hashlib.sha256(data).hexdigest()
    expected = prediction.get("input_file_sha256") if require_file_hash else prediction.get("input_file_sha256", digest)
    if digest != expected:
        raise ValueError("Saved image file hash does not match the prediction; rerun inference on the current image")
    return path, image, digest
