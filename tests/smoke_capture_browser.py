"""Optional integration test: actual fake-camera record/analyze/discard/keep flow.

Run explicitly: python tests/smoke_capture_browser.py
Requires Playwright and its Chromium binary. Never accesses a real webcam.
"""
from pathlib import Path
import csv
import json
import sys
import tempfile
from unittest.mock import patch

from playwright.sync_api import BrowserType, Page

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture

launch = BrowserType.launch
errors = []


def headless(self, **kwargs):
    kwargs["headless"] = True
    return launch(self, **kwargs)


def exercise(page, event, **kwargs):
    assert event == "close"
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.locator("#go").wait_for(state="visible", timeout=15000)
    assert page.locator("#deviceLabel").input_value() == "webcam"
    assert page.locator("#distanceLabel").input_value() == "unspecified"
    assert page.locator("#lightingLabel").input_value() == "unspecified"
    assert page.locator("#cameraSelect option").count() >= 1
    assert "1920" in page.locator("#cameraQuality").inner_text()
    camera = page.evaluate("stream.getVideoTracks()[0].getSettings()")
    assert (camera["width"], camera["height"]) == (1920, 1080), camera
    assert camera["frameRate"] <= 30
    # Display the whole source frame and keep selection coordinates correct when letterboxed.
    assert page.locator(".appbar").count() == 0
    assert page.locator("#cam").evaluate("v => getComputedStyle(v).objectFit") == "contain"
    stage_box = page.locator("#stage").bounding_box()
    viewer_box = page.locator(".viewer").bounding_box()
    assert abs(stage_box["width"] - viewer_box["width"]) < 1, "Camera must fill its allocated width"
    assert abs(stage_box["width"] / stage_box["height"] - camera["width"] / camera["height"]) < .01
    page.evaluate("""() => {
        document.getElementById('stage').style.aspectRatio='4/3';
        window.fitRegion=null;window.fitOriginalRegion=window.pyRegion;
        window.pyRegion=async region=>{window.fitRegion=region;return window.fitOriginalRegion(region);};
    }""")
    guide_box = page.locator("#guide").bounding_box()
    fit = page.evaluate("frameGeometry()")
    page.mouse.move(guide_box["x"]+fit["ox"]+fit["w"]*fit["s"]*.25,
                    guide_box["y"]+fit["oy"]+fit["h"]*fit["s"]*.25)
    page.mouse.down()
    page.mouse.move(guide_box["x"]+fit["ox"]+fit["w"]*fit["s"]*.75,
                    guide_box["y"]+fit["oy"]+fit["h"]*fit["s"]*.75, steps=4)
    page.mouse.up()
    page.wait_for_function("Array.isArray(window.fitRegion) && !regionBusy")
    assert all(abs(a-b)<.01 for a,b in zip(page.evaluate("window.fitRegion"), [.25,.25,.5,.5]))
    page.evaluate("""() => {
        window.pyRegion=window.fitOriginalRegion;
        document.getElementById('stage').style.removeProperty('aspect-ratio');
    }""")
    page.locator("#auto").click()
    page.wait_for_function("!regionBusy")
    assert page.locator("#patientId").input_value() == "P000001"
    assert not page.locator("#patientId").is_editable()
    page.locator("#captureOptions summary").click()
    device = page.locator("#deviceLabel")
    device.fill("bad label")
    assert device.evaluate("input => input.validity.patternMismatch")
    device.fill("phone-test_1")
    assert device.evaluate("input => input.validity.valid")
    device.fill("webcam")
    page.locator("#captureOptions summary").click()
    page.evaluate("""() => {
        window.processCalls=0;const process=window.pyProcess;
        window.pyProcess=(...args)=>{window.processCalls++;return process(...args);};
    }""")
    page.locator("#duration").select_option("3")
    page.locator("#go").click()
    page.locator("#cancel").click()
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Capture cancelled')")
    assert page.locator("#go").is_visible()
    assert page.locator("#count").inner_text() == ""
    assert not page.locator("#patientId").is_disabled()
    assert page.evaluate("window.processCalls") == 0
    # A phone can pause without ending its live track.
    page.locator("#go").click()
    page.locator("#cancel").wait_for(state="visible")
    page.evaluate("stream.getVideoTracks()[0].dispatchEvent(new Event('mute'))")
    page.wait_for_function("document.getElementById('msg').textContent.includes('Camera paused during recording')")
    assert not page.locator("#go").is_visible()
    assert page.evaluate("window.processCalls") == 0
    page.evaluate("stream.getVideoTracks()[0].dispatchEvent(new Event('unmute'))")
    page.locator("#go").wait_for(state="visible")
    page.locator("#go").click()
    page.locator("#cancel").wait_for(state="visible")
    page.evaluate("""() => {
        const track=stream.getVideoTracks()[0];track.stop();track.dispatchEvent(new Event('ended'));
    }""")
    page.wait_for_function("document.getElementById('msg').textContent.includes('Camera disconnected during recording')")
    assert not page.locator("#go").is_visible()
    assert not page.locator("#cancel").is_visible()
    assert page.evaluate("window.processCalls") == 0
    # Re-pairing can change the device ID.
    page.evaluate("""() => {
        const select=document.getElementById('cameraSelect');
        select.add(new Option('Disconnected phone','missing-phone-id'));
        select.value='missing-phone-id';
    }""")
    page.locator("#retryCamera").click()
    page.locator("#go").wait_for(state="visible")
    assert "Previous camera unavailable" in page.locator("#msg").inner_text()
    assert page.locator("#cameraSelect").input_value() != "missing-phone-id"
    page.evaluate("""() => {
        window.NativeMediaRecorder=window.MediaRecorder;
        window.MediaRecorder=class extends window.NativeMediaRecorder {
            start(...args){super.start(...args);setTimeout(()=>this.dispatchEvent(new Event('error')),30);}
        };
    }""")
    page.locator("#go").click()
    page.wait_for_function("document.getElementById('msg').textContent.includes('Camera recording failed')")
    page.evaluate("() => {window.MediaRecorder=window.NativeMediaRecorder;}")
    assert page.evaluate("window.processCalls") == 0
    assert page.locator("#go").is_visible()
    page.locator("#duration").select_option("1")
    with patch.object(capture, "open_output_folder") as open_folder:
        page.locator("#openFolder").click()
        page.wait_for_timeout(100)
        open_folder.assert_called_once()
    page.locator("#go").click()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    assert page.locator(".cand").count() == 4
    assert "no eye" in page.locator("#msg").inner_text().lower()
    assert "Retake" in page.locator("#prediction").inner_text()
    page.evaluate("() => {window.originalDiscard=window.pyDiscard;}")
    with patch.object(Path, "unlink", side_effect=PermissionError("Recording is temporarily locked")):
        page.locator("#disc").click()
        page.wait_for_function("document.getElementById('msg').textContent.startsWith('Could not discard:')")
    assert page.locator("#keep").is_enabled()
    assert page.locator("#rev").is_visible()
    # A cleanup failure must preserve the pending backend capture.
    with patch.object(capture, "save_capture", side_effect=OSError("Test save failure")) as save_attempt:
        page.locator("#keep").click()
        page.wait_for_function("document.getElementById('msg').textContent.startsWith('Save failed:')")
        save_attempt.assert_called_once()
    page.locator("#disc").click()
    page.locator("#go").wait_for(state="visible")
    page.locator("#go").click()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    page.evaluate("""() => {
        window.originalKeep=window.pyKeep;window.saveCalls=0;window.discardCalls=0;
        window.pyKeep=async()=>{
            window.saveCalls++;await new Promise(resolve=>window.releaseSave=resolve);
            return window.originalKeep();
        };
        window.pyDiscard=()=>{window.discardCalls++;return window.originalDiscard();};
    }""")
    with patch.object(capture, "save_capture", side_effect=OSError("CSV is open in another application")):
        page.locator("#keep").click()
        assert page.locator("#keep").is_disabled()
        assert page.locator("#disc").is_disabled()
        page.evaluate("""() => {
            document.getElementById('keep').onclick();document.getElementById('disc').onclick();
            window.releaseSave();
        }""")
        page.wait_for_function("document.getElementById('msg').textContent.startsWith('Save failed:')")
    assert page.evaluate("window.saveCalls") == 1
    assert page.evaluate("window.discardCalls") == 0
    assert page.locator("#rev").is_visible()
    assert page.locator("#patientId").input_value() == "P000001"
    assert page.locator("#keep").is_enabled() and page.locator("#disc").is_enabled()
    page.evaluate("() => {window.pyKeep=window.originalKeep;window.pyDiscard=window.originalDiscard;}")
    page.locator("#keep").click()
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Saved:')")
    assert "CAP_000001" in page.locator("#msg").inner_text()
    assert "Saved for review" in page.locator("#msg").inner_text()
    assert "dataset.csv" in page.locator("#csvPath").text_content()
    assert "3 best frames saved" in page.locator("#lastSave").inner_text()
    assert page.locator("#patientId").input_value() == "P000002"
    page.locator("#captureOptions summary").click()
    page.locator("#deviceLabel").fill("phone_test")
    page.locator("#distanceLabel").select_option("15cm")
    page.locator("#lightingLabel").select_option("daylight")
    page.locator("#captureOptions summary").click()
    page.locator("#stage").scroll_into_view_if_needed()
    box = page.locator("#guide").bounding_box()
    page.mouse.move(box["x"] + box["width"]*.25, box["y"] + box["height"]*.25)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"]*.75, box["y"] + box["height"]*.75, steps=5)
    page.evaluate("""() => {
        window.originalRegion=window.pyRegion;
        window.pyRegion=async(...args)=>{
            await new Promise(resolve=>window.releaseRegion=resolve);
            return window.originalRegion(...args);
        };
    }""")
    page.mouse.up()
    page.wait_for_function("typeof window.releaseRegion === 'function'")
    assert page.locator("#go").is_disabled()
    assert page.locator("#cameraSelect").is_disabled()
    page.evaluate("() => {window.releaseRegion();}")
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Selected region:')")
    page.evaluate("() => {window.pyRegion=window.originalRegion;}")
    page.locator("#go").click()
    assert page.locator("#patientId").is_disabled()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    assert "Selected area" in page.locator("#info").inner_text()
    assert "ROI extraction failed" in page.locator("#msg").inner_text()
    assert page.locator("#crop").get_attribute("src").startswith("data:image/png;base64,")
    assert "original pixels" in page.locator("#cropSize").inner_text()
    page.locator("#inspect").click()
    page.wait_for_function("document.getElementById('detailImage').complete && document.getElementById('detailImage').naturalWidth > 0")
    assert page.locator("#detail").is_visible()
    assert "1:1 view" in page.locator("#detailTitle").inner_text()
    assert page.locator("#detailRoi").is_disabled()
    assert page.locator("#detailImage").evaluate("image => image.getBoundingClientRect().width === image.naturalWidth")
    page.locator("#detailClose").click()
    page.locator("#candidateReview summary").click()
    candidates = page.locator("#cands").bounding_box()
    crop = page.locator("#cropbox").bounding_box()
    assert candidates["y"] >= crop["y"] + crop["height"], "Candidate frames must sit below ROI review"
    page.set_viewport_size({"width":390,"height":844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile layout overflows"
    page.locator("#keep").click()
    page.wait_for_function("document.getElementById('msg').textContent.includes('CAP_000002')")
    assert "cropped eye | ROI extraction failed" in page.locator("#lastSave").inner_text()
    assert page.locator("#patientId").input_value() == "P000003"
    page.locator("#patientMode").select_option("same")
    assert page.locator("#patientId").input_value() == "P000002"
    page.locator("#go").click()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    page.evaluate("""() => {
        const track=stream.getVideoTracks()[0];
        track.dispatchEvent(new Event('mute'));
        track.dispatchEvent(new Event('unmute'));
    }""")
    assert page.locator("#rev").is_visible()
    assert not page.locator("#go").is_visible(), "Reconnect must preserve the current review"
    assert page.locator("#keep").is_visible()
    page.evaluate("""() => {
        const track=stream.getVideoTracks()[0];track.stop();track.dispatchEvent(new Event('ended'));
    }""")
    assert page.locator("#rev").is_visible()
    page.locator("#keep").click()
    page.wait_for_function("document.getElementById('msg').textContent.includes('CAP_000003')")
    assert "Patient P000002" in page.locator("#msg").inner_text()
    assert not page.locator("#go").is_visible()
    page.locator("#retryCamera").click()
    page.locator("#go").wait_for(state="visible")
    assert page.locator("#patientMode").input_value() == "new"
    page.locator("#patientMode").select_option("existing")
    page.locator("#patientId").fill("P999999")
    page.locator("#go").click()
    page.wait_for_function("document.getElementById('patientId').validity.customError")
    assert page.locator("#go").is_visible()
    assert not page.locator("#rev").is_visible()
    assert not page.locator("#cancel").is_visible()
    page.locator("#patientId").fill("P000002")
    # Use a passing fixture to exercise confirmation and persistence with synthetic frames.
    from test_capture import tissue_frame, eye_detector
    from frame_selection import _candidate, quality, CaptureConfig, CRITERIA
    frame = tissue_frame()
    eye = eye_detector(frame)[0]
    q = quality(frame, eye, CaptureConfig())
    q["ok"] = True
    candidate = _candidate(frame, frame, eye, q, 1)
    selected = dict(best="overall", eye_found=True, n_frames=1, candidates={name: candidate for name in CRITERIA})
    from capture_provenance import pixel_hash
    def predicted(selection, *, allow_review=False):
        assert allow_review
        if inference.call_count == 1:
            return dict(status="unavailable", reason="Checkpoint unavailable")
        return dict(status="uncertain", label="Anemic", anemia_score=.8, threshold=.5,
                    input_format="rectangular", input_pixel_sha256=pixel_hash(selection["_roi_result"]["roi"]))
    with patch.object(capture, "select_best_frame", return_value=selected), \
            patch.object(capture.CapturePredictor, "predict", side_effect=predicted) as inference:
        page.locator("#go").click()
        page.locator("#keep").wait_for(state="visible", timeout=30000)
        assert "confirm" in page.locator("#prediction").inner_text().lower()
        inference.assert_not_called()
        page.locator("#editRoi").click()
        page.wait_for_function("document.getElementById('outlineCanvas').width > 100")
        page.locator("#outlineApply").click()
        page.wait_for_function("document.getElementById('outlineStatus').textContent.includes('3 and 200')")
        assert page.locator("#outline").is_visible()
        canvas=page.locator("#outlineCanvas").bounding_box()
        for x,y in ((.10,.55),(.90,.55),(.90,.90),(.10,.90)):
            page.mouse.click(canvas["x"]+x*canvas["width"],canvas["y"]+y*canvas["height"])
        page.locator("#outlineApply").click()
        page.locator("#outline").wait_for(state="hidden")
        inference.assert_not_called()
        assert "corrected mask" in page.locator("#prediction").inner_text()
        page.locator("#confirmRoi").click()
        page.wait_for_function("document.getElementById('msg').textContent.includes('No prediction was generated')")
        assert "Checkpoint unavailable" in page.locator("#prediction").inner_text()
        assert page.locator("#confirmRoi").inner_text() == "Retry prediction"
        assert page.locator("#viewModelInput").count() == 0
        page.locator("#confirmRoi").click()
        page.wait_for_function("document.getElementById('prediction').textContent.includes('80.0%')")
        assert page.locator("#prediction .result-label").inner_text() == "Anemic"
        assert "80.0%" in page.locator("#prediction").inner_text()
        assert "Research result only" in page.locator("#prediction").inner_text()
        page.locator("#viewModelInput").click()
        assert page.locator("#detail").is_visible()
        assert page.locator("#detailTitle").inner_text().startswith("Model input")
        assert page.locator("#detailImage").get_attribute("src") == (
            "data:image/png;base64," + capture._png_b64(selected["_roi_result"]["roi"])
        )
        page.locator("#detailClose").click()
        # The full review must stay usable on desktop, tablet, and phone widths.
        for width, height in ((1440, 900), (1024, 768), (768, 900), (390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Horizontal overflow at {width}px"
            assert page.locator("#crop").is_visible()
            assert page.locator("#prediction .result-label").is_visible()
            assert page.locator("#keep").is_visible()
        page.set_viewport_size({"width": 1280, "height": 900})
        page.locator("#keep").click()
        page.wait_for_function("document.getElementById('msg').textContent.includes('CAP_000004')")
        assert "fold models disagree" in page.locator("#lastSave").inner_text()
        assert inference.call_count == 2
    page.close()


with tempfile.TemporaryDirectory() as temp:
    args = ["capture.py", "--duration", "1", "--fake-camera", "--out", temp]
    with patch.object(sys, "argv", args), patch.object(BrowserType, "launch", headless), \
            patch.object(Page, "wait_for_event", exercise):
        capture.main()
    with (Path(temp)/"dataset.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4, "Retake must not consume a capture or patient ID"
    assert [r["serial_id"] for r in rows] == ["1", "2", "3", "4"]
    assert [r["patient_id"] for r in rows] == ["P000001", "P000002", "P000002", "P000002"]
    metrics = json.loads((Path(temp) / rows[3]["capture_dir"] / "metrics.json").read_text())
    assert rows[3]["prediction_label"] == "Anemic"
    assert rows[3]["prediction_status"] == "uncertain"
    assert rows[3]["prediction_input_path"] == "CAP_000004/roi.png"
    assert rows[3]["anemia_score"] == "0.8"
    assert metrics["prediction"]["anemia_score"] == .8
    assert metrics["roi_confirmed"] is True
    assert metrics["capture_status"] == "approved_for_testing"
    assert metrics["prediction"]["input_file_sha256"]
    assert metrics["camera_settings"]["width"] == 1920
    assert metrics["camera_settings"]["height"] == 1080
    assert metrics["camera_settings"]["mime_type"].startswith("video/webm")
    assert metrics["camera_settings"]["recording_bytes"] > 0
    assert metrics["roi_extraction"]["method"] == "manual_polygon_v1"
    assert len(metrics["roi_extraction"]["polygon_normalized"]) == 4
    assert rows[3]["roi_image_path"].endswith("roi_masked.png")
    assert (rows[0]["device"],rows[0]["distance"],rows[0]["lighting"]) == ("webcam","unspecified","unspecified")
    assert (rows[1]["device"],rows[1]["distance"],rows[1]["lighting"],rows[1]["patient_id"]) == ("phone_test","15cm","daylight","P000002")
    assert rows[0]["roi_status"] == "no_eye"
    assert rows[0]["crop_ok"] == "False"
    assert (Path(temp) / rows[0]["capture_dir"] / "frame.png").is_file()
    for row in rows:
        for name in ("sharpness", "exposure", "glare"):
            assert (Path(temp) / row["capture_dir"] / f"candidate_{name}_review.png").is_file()
    assert (Path(temp) / rows[1]["capture_dir"] / "eye.png").is_file()
    assert rows[1]["roi_image_path"] == ""
    assert rows[1]["roi_status"] == "failed"
    assert not errors, errors
print("PASS: cancel, camera recovery, recorder errors, save retry/action locking, review preservation, candidates, patient IDs and manifest")
