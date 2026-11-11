"""Task_B capture integrated with CP-AnemiC. See docs/CAPTURE.md."""
from __future__ import annotations

import argparse
import base64
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capture_storage import next_patient_id, prepare_manifest, safe_component, save_capture
from capture_roi import extract_capture_roi
from capture_ui import HTML, _jpeg_b64
from eye_detection import OVAL_RX, OVAL_RY
from frame_selection import CRITERIA, CaptureConfig, EyeTracker, RegionDetector, select_best_frame
from utils import load_config, resolve_path


def validate_details(details):
    """Optional operator metadata; never infer a physical distance or lighting."""
    cleaned = {}
    for key, fallback in (("device", "webcam"), ("distance", "unspecified"),
                          ("lighting", "unspecified"), ("patient_id", "")):
        value = str(details.get(key, "")).strip() or fallback
        if value:
            try:
                safe_component(value)
            except ValueError as e:
                raise ValueError(f"{key.replace('_', ' ').title()}: use letters, numbers, hyphens or underscores") from e
        cleaned[key] = value
    return cleaned


def open_output_folder(output):
    if sys.platform == "win32":
        os.startfile(str(output))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(output)])
    else:
        subprocess.Popen(["xdg-open", str(output)])


def main():
    parser = argparse.ArgumentParser(description="Guided eye capture or best-frame selection from a video")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="webcam", help="Optional camera label; editable in the app")
    parser.add_argument("--distance", default="unspecified", help="Optional distance label; editable in the app")
    parser.add_argument("--lighting", default="unspecified", help="Optional lighting label; editable in the app")
    parser.add_argument("--patient-id", default="", help="Optional existing patient ID; default: automatic sequential ID")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--out", help="Capture folder, relative to project root")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--analysis-width", type=int)
    parser.add_argument("--video", help="Process an existing video without opening a camera or browser")
    parser.add_argument("--fake-camera", action="store_true")
    parser.add_argument("--eye-region", help="Fixed conjunctiva region: normalized x,y,width,height (0..1)")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config.get("capture", {})
    try:
        initial_details = validate_details(vars(args))
        if not 1 <= args.duration <= 60:
            raise ValueError("duration must be between 1 and 60 seconds")
        options = {f.name: settings[f.name] for f in fields(CaptureConfig) if f.name in settings}
        for key in ("stride", "analysis_width"):
            if getattr(args, key) is not None:
                options[key] = getattr(args, key)
        cfg = CaptureConfig(**options)
        detector = RegionDetector([float(v) for v in args.eye_region.split(",")]) if args.eye_region else None
    except ValueError as e:
        parser.error(str(e))
    output = resolve_path(args.out or settings.get("output_dir", "outputs/captures"))
    try:
        manifest = prepare_manifest(output)
        initial_details["next_patient_id"] = next_patient_id(output)
    except (ValueError, OSError) as e:
        parser.exit(1, f"Cannot prepare capture CSV: {e}\n")
    print(f"Capture folder: {output}\nCapture CSV: {manifest}")

    def save(result, video, details=None):
        details = initial_details if details is None else details
        return save_capture(result, video, output, details["device"], details["distance"], details["lighting"],
                            details["patient_id"], config.get("roi_extraction"))

    if args.video:
        try:
            video = resolve_path(args.video)
            result = select_best_frame(video, cfg, detector)
            row = save(result, video)
        except (ValueError, RuntimeError, OSError) as e:
            parser.exit(1, f"Capture failed: {e}\n")
        print(json.dumps(dict(capture=row, analyzed_frames=result["analyzed_frames"],
                              detection_calls=result["detection_calls"],
                              elapsed_seconds=result["elapsed_seconds"]), indent=2))
        return
    # Offline selection and training never require the browser dependency.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        parser.exit(1, "Install requirements-capture.txt and run: python -m playwright install chromium\n")

    tracker = EyeTracker(cfg, detector)
    pending = {}
    with tempfile.TemporaryDirectory(prefix="conj_") as temp:
        video = Path(temp) / "recording.webm"

        def track(b64):
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            return tracker.live(frame) if frame is not None else dict(found=False)

        def set_region(region):
            nonlocal detector, tracker
            try:
                detector = RegionDetector(region) if region is not None else None
                tracker = EyeTracker(cfg, detector)
                return dict(ok=True)
            except ValueError as e:
                return dict(error=str(e))

        def process(b64, details=None):
            pending.clear()
            try:
                details = validate_details(initial_details if details is None else details)
                video.write_bytes(base64.b64decode(b64, validate=True))
                result = select_best_frame(video, cfg, detector)
                pending["result"] = result
                pending["details"] = details
                bc = result["candidates"][result["best"]]
                eye = bc["eye"]
                response = dict(image=_jpeg_b64(bc["frame"], width=1280),
                                frames=result["n_frames"], best=result["best"],
                                sharpness=bc["sharpness"], brightness=bc["brightness"], glare=bc["glare"],
                                feedback=bc["feedback"], crop=None, roi=None, roi_status="no_eye",
                                crop_ok=False, cands=[])
                for name in CRITERIA:
                    c = result["candidates"][name]
                    response["cands"].append(dict(name=name, best=name == result["best"],
                        image=_jpeg_b64(c["frame"], width=300, quality=80), sharpness=c["sharpness"],
                        brightness=c["brightness"], glare=c["glare"], coverage=c["coverage"]))
                if eye:
                    roi = extract_capture_roi(eye, config.get("roi_extraction"))
                    result["_roi_result"] = roi
                    response.update(crop=_jpeg_b64(eye["crop"], width=400),
                                    roi=_jpeg_b64(roi["roi_masked"], width=400) if roi["status"] == "ok" else None,
                                    roi_status=roi["status"], roi_reason=roi["reason"],
                                    crop_ok=bc["ok"] and roi["status"] == "ok",
                                    eye_side=eye["side"], eye_method=eye["method"],
                                    conj_coverage=bc["coverage"], conj_score=bc["score"])
                return response
            except Exception as e:
                pending.clear()
                return dict(error=str(e))

        def keep():
            if "result" not in pending:
                return dict(error="No capture pending")
            try:
                row = save(pending["result"], video, pending["details"])
            except Exception as e:
                return dict(error=str(e))
            pending.clear()
            tracker.reset()
            return row

        def discard():
            pending.clear()
            video.unlink(missing_ok=True)
            tracker.reset()
            return True

        def open_folder():
            try:
                open_output_folder(output)
                return dict(ok=True)
            except OSError as e:
                return dict(error=str(e))

        chrome_args = ["--use-fake-ui-for-media-stream", "--start-maximized"]
        if args.fake_camera:
            chrome_args.append("--use-fake-device-for-media-stream")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=chrome_args)
            context = browser.new_context(permissions=["camera"], no_viewport=True)
            page = context.new_page()
            for name, callback in (("pyTrack", track), ("pyProcess", process),
                                   ("pyKeep", keep), ("pyDiscard", discard), ("pyRegion", set_region),
                                   ("pyOpenFolder", open_folder)):
                page.expose_function(name, callback)
            html = HTML.replace("%DURATION%", str(args.duration)).replace("%OVAL_RX%", str(OVAL_RX)).replace("%OVAL_RY%", str(OVAL_RY))
            # JSON encoding plus escaping '<' keeps custom paths out of HTML markup.
            html = html.replace("%OUTPUT_JSON%", json.dumps(output.as_posix()).replace("<", "\\u003c"))
            html = html.replace("%DETAILS_JSON%", json.dumps(initial_details).replace("<", "\\u003c"))
            page.route("https://capture.local/", lambda r: r.fulfill(body=html, content_type="text/html"))
            page.goto("https://capture.local/")
            print("Close the capture browser to finish.")
            try:
                page.wait_for_event("close", timeout=0)
            finally:
                browser.close()


if __name__ == "__main__":
    main()
