# Eye capture and best-frame selection

The Task_B capture workflow includes lower-lid segmentation and uses the project's
training data format. Capture is optional; existing pipeline/training commands
continue to work.

## Start capturing

From the project root, using the project Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-capture.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts/capture.py
```

Or double-click **start_capture.cmd** in the project folder on Windows. No
device, distance, lighting or patient flags are required. Patient IDs are assigned
automatically on Save: `P000001`, `P000002`, etc. The read-only field previews
the next ID. Numbering continues from saved patient IDs across restarts, including
older IDs such as `P001`. Retake and failed saves do not consume a patient ID.
For another saved image of the same person, check **Same patient as the last save**;
uncheck it to automatically assign the next patient. No typing is needed.
Choose a recording length in the app. Expand **Camera & optional
details** to switch cameras or label your device, distance and lighting. Blank
device labels use `webcam`; omitted distance and lighting use `unspecified`
(shown as **Not recorded**). Neither quantity is estimated automatically.
These choices remain in the form for the session. Details are locked during recording and review to keep each
saved row associated with the settings used for that recording.

The **Open save folder** button opens the output folder in your file manager.
CSV and capture paths remain visible beneath it. **Retake** discards the pending
recording; **Save capture** writes the images and CSV row. Camera failures offer
a **Retry camera** action.

Place the eye in view and expose the lower eyelid. The guide follows the eye,
reports focus/exposure/glare problems, and requests continuous camera autofocus
when the camera and browser expose that capability. It cannot force hardware
autofocus on unsupported devices. Click **Start capture**, hold still, review the
four candidates and segmented ROI preview, then **Save capture** or **Retake**. The save folder,
CSV location and last saved capture folder are shown below the buttons. Green indicates that
image-quality checks passed; visually check that the exposed lower conjunctiva
is actually present. Quality metrics alone cannot establish that anatomy.

Use **Same patient as the last save** for repeat captures of one person.
An existing ID can still be supplied with `--patient-id` for a returning patient.
Device, distance, lighting and patient labels accept letters, digits, `_`, `-`.
No real patient name is needed. Close the browser to end the session.

For extreme close-ups where Haar locks onto the iris/eyelashes or misses the eye,
**drag a box over the exposed lower conjunctiva** in the live preview. This selects
a fixed region: hold the tissue within the guide during recording. No eye detection
is run for that mode; the selected region is scored directly and marked
`crop_method=manual_region`. Use **Automatic eye detection** to clear it. Manual
selection is operator input, not an algorithmic confirmation of anatomy.

For a prerecorded close-up, use `--eye-region 0.15,0.45,0.70,0.30` (example only):
normalized x, y, width, height, measured from the full image's upper-left corner.
Choose coordinates for the actual clip; they remain fixed throughout it.

```powershell
# Test without a real webcam; no eye/crop is expected from the test pattern.
.\.venv\Scripts\python.exe scripts/capture.py --fake-camera

# Select and save from an existing video, without opening a camera/browser.
.\.venv\Scripts\python.exe scripts/capture.py --video path/to/eye.mp4
```

`--video` saves directly, including failed quality results for review. Browser
capture saves only on Save capture. Neither path automatically admits a capture to training.
`--out`, `--config`, `--stride`, and `--analysis-width` override defaults. Relative
paths resolve against the project root, matching the other pipeline commands.
The original `--device`, `--distance`, `--lighting`, `--patient-id` and `--duration`
flags still work as optional initial values, including for offline video processing.

## What changed from Task_B

* Detection runs at a maximum dimension of 640 pixels by default. Tracking uses
  forward/backward optical flow between detections (every fifth analyzed frame),
  with immediate redetection when flow fails. Eye proximity keeps the guide from
  switching solely because the other eye becomes redder.
* Selection decodes the video once. Skipped frames use `grab()` without image
  conversion, and at most four full-resolution candidate frames are retained.
  Runtime still grows with video length; full source frames are decoded when scored.
* Four candidates: overall quality, sharpness, exposure, lowest glare. The final
  choice is the overall winner across all analyzed valid eye frames, rather than
  only the three individual-metric winners. Each criterion first prefers frames
  passing every quality gate. Ties keep the earliest frame.
* Focus is measured on the lower-lid band after mild denoising, with eroded mask
  edges. Large patches are normalized in size; small patches are not enlarged.
* Red coverage is a review diagnostic only. It does not filter or rank captures,
  avoiding a built-in preference for red tissue in an anemia dataset. This does
  not remove color dependence from the existing conjunctiva ROI segmentation.
* A missed eye never becomes a guessed valid crop. If every frame fails detection,
  whole-guide candidates remain available for review with `crop_ok=False` and
  `roi_status=no_eye`. If detected eyes fail quality checks, the best available
  frame remains visibly flagged. There is no automatic successful-capture claim.

Thresholds live under `capture:` in `configs/config.yaml`. These are starting
heuristics, not calibrated accuracy guarantees. Use recorded clips from the target
device to tune focus, glare and brightness limits. `--stride 1` examines every
frame at a higher processing cost. Rapid motion can fall between sampled frames.

## Saved artifacts and pipeline integration

The default index is `outputs/captures/dataset.csv`. Every Save adds one row with
an automatic `serial_id` (1, 2, 3, ...). New captures use IDs `CAP_000001`,
`CAP_000002`, etc. Numbering continues across restarts and devices sharing that
output folder. Retake does not consume an ID. IDs are independent of patient IDs.

Each CSV row contains these image links:

| CSV column | Saved image |
| --- | --- |
| `sharpness_path` | Best-sharpness candidate |
| `exposure_path` | Best-exposure candidate |
| `glare_path` | Lowest-glare candidate |
| `eye_image_path` | Cropped eye from the selected overall frame |
| `roi_image_path` | Extracted conjunctiva ROI |
| `image_path` | Clean full-resolution overall winner |

The CSV stores absolute file paths; the actual PNG images are saved beside one
another in the capture folder shown in the app. A single frame may win multiple quality criteria;
all three candidate files and their CSV paths are still saved.

`outputs/captures/<device>/<distance>/<lighting>/CAP_000001/` contains:

| File | Purpose |
| --- | --- |
| `frame.png` | Clean full-resolution overall winner; `image_path` |
| `eye.png` | Clean detected eye crop; `eye_image_path` |
| `roi.png` | Tight rectangular crop around segmented lower-lid tissue |
| `roi_masked.png`, `roi_mask.png` | Tissue-only pixels on black, and the binary segmentation mask |
| `roi_debug.png` | Four panels: eye with bounding box, mask, rectangular ROI, masked ROI |
| `eye_search_mask.png` | Detector's lower-lid search region, when available |
| `review.png`, `candidate_*.png` | Annotated review images; never training inputs |
| `recording.*` | Original recording |
| `metrics.json` | Candidate scores, frame indices, timing and capture metadata |

`outputs/captures/dataset.csv` also records `patient_id`, `roi_status` and `crop_ok`.
The CSV has 17 columns: the two IDs, patient/device/distance/lighting details,
the six image paths above, two quality-status fields, and sharpness/brightness/glare
scores. Timestamp, folder/video/masked/review/metrics paths, frame index, internal
conjunctiva scores and crop method stay in each capture's `metrics.json` instead.
Earlier 27-column CSVs are trimmed automatically with a `dataset.before_cleanup_*.csv`
backup. Images and per-capture metadata remain intact.
No diagnosis/label is inferred from capture quality. Capture segmentation uses
eye geometry to restrict the search to the lower lid, then separates pink
tissue from yellowish skin using Lab chroma and connected-component geometry.
The bulk dataset extractor remains unchanged. Capture ROI status `failed` means
segmentation found no acceptable component: no placeholder ROI is saved,
the ROI paths stay blank, and `crop_ok=False`. The UI shows that failure before
Save. These color/geometry heuristics are not anatomical ground truth, especially
for small, blurred webcam eyes; check the mask and move closer when needed.
Artifacts are staged before publication, and the CSV is replaced atomically.
A save lock prevents simultaneous sessions from assigning the same serial.
If Excel blocks the CSV, close it and retry Save; the failed save is rolled back.
Existing exports from the earlier integration are automatically upgraded with
serial IDs and the three missing candidate paths; a `dataset.before_serial_ids_*.csv`
backup is retained. Existing image IDs and image files are preserved. Unknown
CSV schemas fail with an actionable message. With no detected/selected eye,
the crop and ROI columns stay empty and `roi_status=no_eye` is recorded.

To regenerate a previous capture's ROI with the new extractor:

```powershell
.\.venv\Scripts\python.exe scripts/reextract_capture_roi.py --capture-dir outputs/captures/<device>/<distance>/<lighting>/<capture_id>
```

This updates its ROI images, QA preview and existing CSV row, preserving the
old outputs and metadata in a `roi_backup_*` folder. For older captures without
saved eye geometry, it replays the recording and requires the same selected
frame and exact eye pixels before proceeding. Original frames are not modified.

To use reviewed captures for training, create a labels CSV with explicit ground
truth (plus optional clinical columns such as `hb_level`):

```csv
image_id,patient_id,label_text
CAP_000001,P001,Anemic
CAP_000002,P002,Non-anemic
```

Replace example IDs with actual capture IDs. List only reviewed captures. Then:

```powershell
.\.venv\Scripts\python.exe scripts/import_captures.py --labels path/to/capture_labels.csv
```

This writes `outputs/metadata/captures_with_roi.csv`. It rejects duplicate IDs,
unknown captures, missing files, patient-ID conflicts, missing labels, failed
quality gates and fallback ROIs. Unlabelled captures are excluded. It preserves
patient IDs for group-based cross-validation; frames from one patient must never
be assigned different IDs merely to fill folds.

Copy `configs/config.yaml` to a separate capture-training config, set
`roi_extraction.output_csv` to `outputs/metadata/captures_with_roi.csv`, retain
`training.image_source: with_roi`, and use a separate `training.output_dir`.
Run `scripts/train_baseline.py --config <that-config>` once there are enough
labelled patients in both classes for the configured folds. Do not run the
metadata/ROI rebuild stages against this capture manifest; they target the
original folder-and-spreadsheet dataset.

For existing trained-model inference on an individual saved capture:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --input outputs/captures/<device>/<distance>/<lighting>/<capture_id>/roi.png
```

Review `roi_status` first. Do not pass the entire capture folder to prediction:
it also contains annotated candidate images.

## Verification and limitations

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_capture_browser.py
```

Unit tests cover sharp/blurred ranking, dark/clipped frames, pale tissue eligibility,
flow recovery, bounded retained frames, original-resolution coordinates, clean
saved pixels, ROI fallback, invalid inputs and patient-label joins. The browser
smoke test uses Chromium's fake camera and exercises recording, all four review
candidates, Retake, Save capture and CSV output without using a physical webcam.

Haar is available with the project's OpenCV install. Optional MediaPipe is used
only when its installed package exposes `solutions.face_mesh`; otherwise Haar is
used. MediaPipe is not installed or version-pinned by this integration. Haar can
miss very close, tilted or occluded eyes; the 13 existing close-up photographs
produced 8 detections in a local check, but visual inspection found several
boxes localized to an iris/eyelash fragment rather than the complete eye. Use
manual region selection on such close-ups. Detection counts are not accuracy.
Real moving-eye clips and physical camera autofocus still need device testing.

Local timing check (2026-11-11, single run, CPU): a 90-frame MJPG clip at 30 fps,
1280-pixel width, made from `data/uncropped/20200203_190841.jpg`, with Gaussian
blur applied to the first 30 frames. Both versions used stride 2 and Haar.

| Implementation | Selection time | Full detector calls | Selected frame |
| --- | ---: | ---: | ---: |
| Task_B source at the commit below | 8.91 s | 45 | 31 |
| Integrated selector | 2.03 s | 9 | 31 |

This is approximately 4.4x faster on that synthetic stationary clip. It is a
processing check, not a moving-eye accuracy benchmark or a general speed guarantee.
The integrated run tracked 36 analyzed frames between detections and retained
two distinct full-resolution candidate frames. Per-capture JSON records timing
and detector counts so real-device clips can be compared next.

## Source attribution

Eye geometry, annotation and browser UI were adapted from
[Subhankar191/Task_B](https://github.com/Subhankar191/Task_B), inspected at commit
`b11e8467ebc2295ad50e6cfab0f6c31ed9c9d99a` (2026-11-10).
The source supplied no license file alongside the code; this integration retains
its attribution and does not add a licensing grant for that source.
Frame scoring/selection, optical-flow tracking, storage and CP-AnemiC integration
are implemented in separate local modules. Upstream datasets were not imported.
