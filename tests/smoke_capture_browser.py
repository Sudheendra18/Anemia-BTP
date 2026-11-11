"""Optional integration test: actual fake-camera record/analyze/discard/keep flow.

Run explicitly: python tests/smoke_capture_browser.py
Requires Playwright and its Chromium binary. Never accesses a real webcam.
"""
from pathlib import Path
import csv
import sys
import tempfile
from unittest.mock import patch

from playwright.sync_api import BrowserType, Page

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp").mkdir(exist_ok=True)
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
    assert page.locator("#patientId").input_value() == "P000001"
    assert not page.locator("#patientId").is_editable()
    with patch.object(capture, "open_output_folder") as open_folder:
        page.locator("#openFolder").click()
        page.wait_for_timeout(100)
        open_folder.assert_called_once()
    page.locator("#go").click()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    assert page.locator(".cand").count() == 4
    assert "no eye" in page.locator("#msg").inner_text().lower()
    page.locator("#disc").click()
    page.locator("#go").wait_for(state="visible")
    page.locator("#go").click()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    page.locator("#keep").click()
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Saved:')")
    assert "CAP_000001" in page.locator("#msg").inner_text()
    assert "dataset.csv" in page.locator("#csvPath").inner_text()
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
    page.mouse.up()
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Selected region:')")
    page.locator("#go").click()
    assert page.locator("#patientId").is_disabled()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    assert "Selected area" in page.locator("#info").inner_text()
    assert "ROI extraction failed" in page.locator("#msg").inner_text()
    candidates = page.locator("#cands").bounding_box()
    crop = page.locator("#cropbox").bounding_box()
    assert candidates["x"] + candidates["width"] <= crop["x"], "Crop must not cover candidate frames"
    page.screenshot(path=str(ROOT / "tmp" / "capture-smoke.png"), full_page=True)
    page.set_viewport_size({"width":390,"height":844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile layout overflows"
    page.screenshot(path=str(ROOT / "tmp" / "capture-mobile.png"), full_page=True)
    page.locator("#keep").click()
    page.wait_for_function("document.getElementById('msg').textContent.includes('CAP_000002')")
    assert "cropped eye | ROI extraction failed" in page.locator("#lastSave").inner_text()
    assert page.locator("#patientId").input_value() == "P000003"
    page.locator("#samePatient").check()
    assert page.locator("#patientId").input_value() == "P000002"
    page.locator("#go").click()
    page.locator("#keep").wait_for(state="visible", timeout=30000)
    page.locator("#keep").click()
    page.wait_for_function("document.getElementById('msg').textContent.includes('CAP_000003')")
    assert "Patient P000002" in page.locator("#msg").inner_text()
    page.locator("#go").wait_for(state="visible")
    page.close()


with tempfile.TemporaryDirectory() as temp:
    args = ["capture.py", "--duration", "1", "--fake-camera", "--out", temp]
    with patch.object(sys, "argv", args), patch.object(BrowserType, "launch", headless), \
            patch.object(Page, "wait_for_event", exercise):
        capture.main()
    with (Path(temp)/"dataset.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3, "Retake must not consume a capture or patient ID"
    assert [r["serial_id"] for r in rows] == ["1", "2", "3"]
    assert [r["patient_id"] for r in rows] == ["P000001", "P000002", "P000002"]
    assert (rows[0]["device"],rows[0]["distance"],rows[0]["lighting"]) == ("webcam","unspecified","unspecified")
    assert (rows[1]["device"],rows[1]["distance"],rows[1]["lighting"],rows[1]["patient_id"]) == ("phone_test","15cm","daylight","P000002")
    assert rows[0]["roi_status"] == "no_eye"
    assert rows[0]["crop_ok"] == "False"
    assert Path(rows[0]["image_path"]).is_file()
    for row in rows:
        for key in ("sharpness_path", "exposure_path", "glare_path"):
            assert Path(row[key]).is_file(), key
    assert Path(rows[1]["eye_image_path"]).is_file()
    assert rows[1]["roi_image_path"] == ""
    assert rows[1]["roi_status"] == "failed"
    assert not errors, errors
print("PASS: fake camera, recording, four candidates, discard, keep, manual target, clean image and manifest")
