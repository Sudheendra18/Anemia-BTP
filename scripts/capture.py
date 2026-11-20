"""Task_B capture integrated with CP-AnemiC. See docs/CAPTURE.md."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_dependency_error = None
try:
    import cv2
    import numpy as np

    if not hasattr(cv2, "FONT_HERSHEY_SIMPLEX") or not hasattr(cv2, "IMREAD_COLOR"):
        raise ImportError(f"Incomplete OpenCV module at {getattr(cv2, '__file__', None)}")
    from capture_storage import next_patient_id, prepare_manifest, safe_component, save_capture, available_patient_ids
    from capture_roi import choose_capture_roi, manual_capture_roi, tissue_quality
    from capture_prediction import CapturePredictor, SCORED_STATUSES
    from capture_ui import HTML, _jpeg_b64, _png_b64
    from eye_detection import OVAL_RX, OVAL_RY, load_haar_detector
    from frame_selection import CRITERIA, CaptureConfig, EyeTracker, RegionDetector, select_best_frame
    from utils import load_config, resolve_path
except ImportError as error:
    _dependency_error = error


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
    cleaned["patient_mode"] = details.get("patient_mode", "")
    if cleaned["patient_mode"] not in ("", "new", "existing"):
        raise ValueError("Invalid patient selection")
    if cleaned["patient_mode"] == "existing" and not cleaned["patient_id"]:
        raise ValueError("Select an existing patient ID")
    return cleaned


def camera_metadata(settings):
    """Persist observed browser settings without browser-specific device identifiers."""
    if not isinstance(settings, dict):
        raise ValueError("Camera settings must be an object")
    output = {"source": "browser", "camera_name": str(settings.get("camera_name", ""))[:200]}
    for key in ("width", "height", "frame_rate", "requested_duration_seconds", "video_bits_per_second", "recording_bytes"):
        value = settings.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid camera setting: {key}")
            output[key] = value
    output["mime_type"] = str(settings.get("mime_type", ""))[:100]
    return output


def open_output_folder(output):
    if sys.platform == "win32":
        os.startfile(str(output))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(output)])
    else:
        subprocess.Popen(["xdg-open", str(output)])


@contextmanager
def capture_chromium(parser):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        parser.exit(1, "Install requirements-capture.txt. See First-time setup in README.md.\n")
    with sync_playwright() as playwright:
        chromium = playwright.chromium
        if not Path(chromium.executable_path).is_file():
            parser.exit(1, "Chromium is missing. Run python -m playwright install chromium in the project environment.\n")
        yield chromium


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--video", help="Process an existing video without opening a camera or browser")
    mode.add_argument("--check-setup", action="store_true", help="Check the detector, Chromium and model without opening the camera")
    parser.add_argument("--fake-camera", action="store_true")
    parser.add_argument("--eye-region", help="Fixed conjunctiva region: normalized x,y,width,height (0..1)")
    args = parser.parse_args()
    if _dependency_error is not None:
        parser.exit(1, f"Capture dependencies are unavailable: {_dependency_error}. See First-time setup in README.md.\n")
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
    if not args.video or detector is None:
        try:
            load_haar_detector()
        except RuntimeError as error:
            parser.exit(1, f"Capture setup failed: {error}\n")

    predictor = CapturePredictor(settings.get("inference"))
    if args.check_setup:
        with capture_chromium(parser):
            pass
        if predictor.enabled:
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            result = predictor.predict_roi({"roi": pixels, "roi_masked": pixels})
            if result["status"] not in SCORED_STATUSES:
                parser.exit(1, f"Prediction setup failed: {result['reason']}\n")
        inference = "all five model checkpoints scored on CPU" if predictor.enabled else "inference disabled"
        print(f"Capture setup passed. OpenCV eye detector and Chromium found. Model: {inference}.")
        return

    if not args.video and predictor.enabled:
        try:
            predictor.checkpoint_paths()
        except OSError as error:
            initial_details["setup_warning"] = f"Prediction unavailable: {error}"
            print(initial_details["setup_warning"], file=sys.stderr)

    output = resolve_path(args.out or settings.get("output_dir", "outputs/captures"))
    try:
        manifest = prepare_manifest(output)
        initial_details["next_patient_id"] = next_patient_id(output)
        initial_details["patient_ids"] = available_patient_ids(output)
    except (ValueError, OSError) as e:
        parser.exit(1, f"Cannot prepare capture CSV: {e}\n")
    print(f"Capture folder: {output}\nCapture CSV: {manifest}")
    recovery_path = output / "recovery_report.json"
    if recovery_path.exists():
        print(f"Interrupted-save recovery report: {recovery_path}")

    def save(result, details=None):
        if "prediction" not in result:
            choose_capture_roi(result, config.get("roi_extraction"))
            result.setdefault("roi_confirmed", False)
            result["prediction"] = predictor.predict(result)
        details = initial_details if details is None else details
        return save_capture(result, output, details["device"], details["distance"], details["lighting"],
                            details["patient_id"], config.get("roi_extraction"), details.get("patient_mode", ""))

    if args.video:
        try:
            video = resolve_path(args.video)
            result = select_best_frame(video, cfg, detector)
            reader = cv2.VideoCapture(str(video))
            try:
                result["camera_settings"] = dict(source="imported_video", width=reader.get(cv2.CAP_PROP_FRAME_WIDTH),
                    height=reader.get(cv2.CAP_PROP_FRAME_HEIGHT), frame_rate=reader.get(cv2.CAP_PROP_FPS),
                    frame_count=reader.get(cv2.CAP_PROP_FRAME_COUNT), recording_bytes=video.stat().st_size)
            finally:
                reader.release()
            row = save(result)
        except (ValueError, RuntimeError, OSError) as e:
            parser.exit(1, f"Capture failed: {e}\n")
        print(json.dumps(dict(capture=row, analyzed_frames=result["analyzed_frames"],
                              detection_calls=result["detection_calls"],
                              elapsed_seconds=result["elapsed_seconds"]), indent=2))
        return
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

        def process(b64, details=None, camera_settings=None):
            pending.clear()
            try:
                details = validate_details(initial_details if details is None else details)
                video.write_bytes(base64.b64decode(b64, validate=True))
                result = select_best_frame(video, cfg, detector)
                result["camera_settings"] = camera_metadata(camera_settings) if camera_settings is not None else {"source": "unknown"}
                choose_capture_roi(result, config.get("roi_extraction"))
                result["roi_confirmed"] = False
                result["prediction"] = predictor.preview(result.get("_roi_result"))
                pending["result"] = result
                pending["details"] = details
                bc = result["candidates"][result["best"]]
                eye = bc["eye"]
                response = dict(prediction=result["prediction"], image=_jpeg_b64(bc["frame"], width=1280),
                                sharpness=bc["sharpness"], brightness=bc["brightness"], glare=bc["glare"],
                                feedback=bc["feedback"], crop=None, roi=None, roi_status="no_eye",
                                crop_ok=False, cands=[], thumbnails=[])
                names = list(CRITERIA)
                if result["best"] not in CRITERIA:
                    names.append(result["best"])
                thumbnail_indices = {}
                for name in names:
                    c = result["candidates"][name]
                    frame_key = id(c["frame"])
                    if frame_key not in thumbnail_indices:
                        thumbnail_indices[frame_key] = len(response["thumbnails"])
                        response["thumbnails"].append(_jpeg_b64(c["frame"], width=300, quality=80))
                    response["cands"].append(dict(name=name, best=name == result["best"],
                        thumbnail=thumbnail_indices[frame_key], sharpness=c["sharpness"],
                        brightness=c["brightness"], glare=c["glare"]))
                if eye:
                    roi = result["_roi_result"]
                    response.update(crop=_png_b64(eye["crop"]),
                                    roi=_png_b64(roi["roi_masked"]) if roi["status"] == "ok" else None,
                                    crop_size=list(eye["crop"].shape[1::-1]),
                                    roi_size=list(roi["roi"].shape[1::-1]) if roi["status"] == "ok" else None,
                                    roi_status=roi["status"], roi_reason=roi["reason"],
                                    crop_ok=bc["ok"] and roi["status"] == "ok",
                                    eye_method=eye["method"])
                return response
            except Exception as e:
                pending.clear()
                return dict(error=str(e))

        def refine(points):
            try:
                result = pending["result"]
                candidate = result["candidates"][result["best"]]
                if candidate["eye"] is None:
                    raise ValueError("Retake and select the exposed tissue in the camera view")
                roi = manual_capture_roi(candidate["eye"], points)
                measured = tissue_quality(roi, cfg)
                preview = _png_b64(roi["roi_masked"])
                candidate["tissue_quality"] = measured
                candidate["ok"] = bool(candidate["eye"]["conj"]["ok"] and measured["ok"])
                candidate["feedback"] = measured["feedback"]
                result["_roi_result"] = roi
                result["roi_confirmed"] = False
                result["prediction"] = predictor.preview(roi, corrected=True)
                return dict(roi=preview, roi_size=list(roi["roi"].shape[1::-1]),
                            crop_ok=candidate["ok"], feedback=candidate["feedback"],
                            prediction=result["prediction"])
            except (KeyError, ValueError, TypeError) as error:
                return dict(error=str(error))

        def confirm_roi():
            if "result" not in pending:
                return dict(error="No capture pending")
            result = pending["result"]
            if not result.get("_roi_result") or result["_roi_result"]["status"] != "ok":
                return dict(error="Correct the tissue mask before confirming")
            result["roi_confirmed"] = True
            prediction = predictor.predict(result, allow_review=True)
            result["prediction"] = prediction
            response = dict(prediction=prediction, quality_passed=bool(result["candidates"][result["best"]]["ok"]))
            if prediction["status"] in SCORED_STATUSES and prediction["input_format"] == "rectangular":
                # Masked predictions can reuse the lossless ROI preview already in the browser.
                response["model_input"] = _png_b64(result["_roi_result"]["roi"])
            return response

        def keep():
            if "result" not in pending:
                return dict(error="No capture pending")
            try:
                row = save(pending["result"], pending["details"])
            except Exception as e:
                return dict(error=str(e))
            pending.clear()
            try:
                video.unlink(missing_ok=True)
            except OSError:
                pass  # TemporaryDirectory retries cleanup when the app closes.
            tracker.reset()
            return row

        def discard():
            try:
                video.unlink(missing_ok=True)
            except OSError as e:
                return dict(error=str(e))
            pending.clear()
            tracker.reset()
            return True

        def open_folder():
            try:
                open_output_folder(output)
                return dict(ok=True)
            except OSError as e:
                return dict(error=str(e))

        def refresh_patient_ids():
            try:
                return dict(ids=available_patient_ids(output))
            except (OSError, ValueError) as error:
                return dict(error=str(error))

        chrome_args = ["--use-fake-ui-for-media-stream", "--start-maximized"]
        if args.fake_camera:
            chrome_args.append("--use-fake-device-for-media-stream")
        with capture_chromium(parser) as chromium:
            from playwright.sync_api import Error as PlaywrightError
            try:
                browser = chromium.launch(headless=False, args=chrome_args)
            except PlaywrightError as error:
                parser.exit(1, f"Cannot launch Chromium. See First-time setup in README.md.\n{error}\n")
            context = browser.new_context(permissions=["camera"], no_viewport=True)
            page = context.new_page()
            for name, callback in (("pyTrack", track), ("pyProcess", process),
                                   ("pyKeep", keep), ("pyDiscard", discard), ("pyRegion", set_region),
                                   ("pyRefine", refine), ("pyConfirmRoi", confirm_roi),
                                   ("pyOpenFolder", open_folder), ("pyPatientIds", refresh_patient_ids)):
                page.expose_function(name, callback)
            html = HTML.replace("%DURATION%", str(args.duration)).replace("%OVAL_RX%", str(OVAL_RX)).replace("%OVAL_RY%", str(OVAL_RY))
            # Escape '<' so values cannot close the script tag.
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
