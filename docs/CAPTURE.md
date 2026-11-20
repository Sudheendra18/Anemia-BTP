# Eye capture and best-frame selection

The Task_B capture workflow includes lower-lid segmentation and uses the project's
training data format. Capture is optional; existing pipeline/training commands
continue to work.

## Start capturing

Follow [First-time setup](../README.md#first-time-setup) to install the
dependencies and Chromium, extract the model checkpoints and run the setup
check. Then launch from the project root using the project Python environment:

```powershell
.\.venv\Scripts\python.exe scripts/capture.py
```

Or double-click **start_capture.cmd** in the project folder on Windows. No
device, distance, lighting or patient flags are required. Patient IDs are assigned
automatically on Save: `P000001`, `P000002`, etc. The read-only field previews
the next ID. Numbering continues from saved patient IDs across restarts, including
older IDs such as `P001`. Retake and failed saves do not consume a patient ID.
Choose **New patient**, **Same patient as last save**, or **Existing patient**.
Existing patient IDs can be selected from the saved IDs in this capture folder.
After a successful save, the selection resets to **New patient** to avoid accidentally
grouping different people. Retakes and failed saves preserve the current selection.
Choose a camera and recording length in the settings sidebar. Expand **Optional
capture details** to label your device, distance and lighting. Blank
device labels use `webcam`; omitted distance and lighting use `unspecified`
(shown as **Not recorded**). Neither quantity is estimated automatically.
These choices remain in the form for the session. Details are locked during recording and review to keep each
saved row associated with the settings used for that recording.

The camera requests 1920 × 1080 at up to 30 fps, falling back to a supported
resolution. The actual resolution is shown above the preview. Recording bitrate
scales with the camera resolution (8–24 Mbps requested); the browser/encoder may
use a different actual bitrate. Continuous-focus requests preserve the resolution
constraints instead of resetting the camera to its defaults.

Review crops use original-size lossless PNG previews. **View original pixels**
opens a scrollable 1:1 view with separate eye-crop and tissue-ROI buttons. Crop
dimensions are shown so a tiny crop is not mistaken for a high-detail image.
The pipeline preserves source colors and pixels; it does not fabricate detail
through enlargement or sharpen an already blurred recording.

The **Open folder** button opens the output folder in your file manager.
Expand **Storage location** for CSV and output paths. The last save shows its
capture folder and saved prediction. **Retake** discards the pending
recording; **Save capture** writes the images and CSV row. **Cancel capture** stops
an active recording immediately without saving or using a patient ID. Camera
failures offer a **Retry camera** action, including if the camera disconnects
mid-recording. A disconnect after recording leaves the review available to save.
Save and Retake are both locked while either action is in progress. If a save
fails (for example, the CSV is open in Excel), the preview stays available so you
can fix the problem and retry Save without recording again.

Place the eye in view and expose the lower eyelid. The guide follows the eye,
reports focus/exposure/glare problems, and requests continuous camera autofocus
when the camera and browser expose that capability. It cannot force hardware
autofocus on unsupported devices. Click **Start capture**, hold still, review the
category candidates and segmented ROI preview. Correct the outline if needed,
then **Confirm ROI & predict anemia** before saving, or choose **Retake**. The save folder,
CSV location and last saved capture folder are shown below the buttons. Green indicates that
image-quality checks passed; visually check that the exposed lower conjunctiva
is actually present. Quality metrics alone cannot establish that anatomy.

Use **Same patient as the last save** for repeat captures of one person.
Use **Existing patient** for a returning patient. An ID can also be supplied with
`--patient-id` when launching the app, including an externally assigned ID.
Device, distance, lighting and patient labels accept letters, digits, `_`, `-`.
No real patient name is needed. Close the browser to end the session.

## Save status, provenance, and recovery

**Save for review** keeps a capture even when tissue has not been confirmed or
quality checks failed. Such captures have `crop_ok=false` and
`capture_status=saved_for_review` in `metrics.json`. The save message explicitly
says they are not approved for testing. Confirmed captures that pass quality
checks have `capture_status=approved_for_testing`; this refers to image review,
not clinical approval or a guarantee that model inference succeeded.

Each new capture's `metrics.json` records the camera name, actual browser-reported
resolution and frame rate, requested duration, recorder-reported bitrate and
codec, and recording byte count. Imported videos record their available video
properties instead; unavailable camera details are not invented.

Predictions include a hash of the original input pixels. On save, this is checked
against the PNG and supplemented by a file hash. A mismatch marks the prediction
stale. Ensemble evaluation verifies both hashes and requires the evaluation CSV's
patient ID (and image ID, if provided) to match the saved capture metadata. Older
predictions without these hashes cannot pass the new verification; they are not
automatically relabeled or treated as verified.

At startup, complete published capture folders missing from `dataset.csv` are
reindexed only when their IDs, image inventory, hashes, and paths validate. The
previous CSV is backed up before recovery. Partial staging folders, damaged
captures, and older folders without an image inventory are preserved for manual
review. Recovery findings are recorded in `recovery_report.json`; no capture
images are deleted by recovery.

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
  conversion, and at most twelve full-resolution candidate frames are retained.
  Runtime still grows with video length; full source frames are decoded when scored.
* Four candidates: overall quality, sharpness, exposure, lowest glare. Each
  criterion first prefers frames passing every quality gate. Ties keep the
  earliest frame. Up to eight quarter-second-window alternatives supplement the
  category winners. Before review/save, the shortlist is segmented and ranked by
  passing tissue quality, tissue score, then original quality. If none succeeds,
  the original winner remains flagged as failed. A successful segmentation does
  not turn failed focus/exposure checks into a passing capture.
* ROI comparison checks at most twelve retained candidates without decoding the
  recording again. Shared candidates are segmented once, and the preview's
  result is reused on Save. `metrics.json` records the original quality winner,
  checked frames, and final selection under `roi_selection`.
* Live detection reuses each eye's measurements. For recorded high-resolution
  frames, selection scores the original eye crop instead of the reduced detection
  preview. Only the local mask is mapped to source resolution, avoiding a
  full-frame mask allocation. `metrics.json` records `scoring_source`.
* Focus is measured on the lower-lid band after mild denoising, with eroded mask
  edges. Large patches are normalized in size; small patches are not enlarged.
* Red coverage is a review diagnostic only. It does not filter or rank captures,
  avoiding a built-in preference for red tissue in an anemia dataset. This does
  not remove color dependence from the existing conjunctiva ROI segmentation.
* ROI extraction refines the coarse contour using source-resolution tissue color,
  excludes small native-resolution glare spots, and reapplies the eye search
  bounds. Saved tissue pixels remain unchanged. An inclusive color threshold
  also handles manually selected areas of uniform tissue without rejecting
  every pixel for being equal to its own threshold. These remain heuristics.
* A missed eye never becomes a guessed valid crop. If every frame fails detection,
  whole-guide candidates remain available for review with `crop_ok=False` and
  `roi_status=no_eye`. If detected eyes fail quality checks, the best available
  frame remains visibly flagged. There is no automatic successful-capture claim.

Thresholds live under `capture:` in `configs/config.yaml`. These are starting
heuristics, not calibrated accuracy guarantees. Use recorded clips from the target
device to tune focus, glare and brightness limits. `--stride 1` examines every
frame at a higher processing cost. Rapid motion can fall between sampled frames.

## Saved artifacts and pipeline integration

Camera choices refresh when devices change. Retry refreshes device IDs before
opening a camera, so a re-paired phone's stale ID does not trap the app in repeated
failures. The current camera name is displayed with its resolution. Reconnecting
or switching cameras resets the manual eye region; select it again for the new
view. Capture and camera controls are locked while a region change is applying,
and tracking responses from the previous region are ignored. An invalid inference
threshold produces an unavailable model result instead of preventing app startup.

After selection and ROI extraction, review the masked tissue. Expand **Compare
selected frames** to inspect the category winners. Use **Edit
tissue outline** to click around the exposed lower-lid boundary if automatic
extraction is wrong, including pale areas. Undo points or cancel without changing
the saved capture. Apply the outline, inspect the preview, then choose **Confirm
ROI & predict anemia**. Manual outlines retain original colors and bypass redness
thresholds, but still require passing tissue focus/exposure checks. Outline points
are stored in metrics so regeneration preserves your correction. If no eye crop
is found, retake after drawing a region in the live camera view.

Confirmed extracted ROIs run through the five saved
fold checkpoints of `efficientnet_b0_lr1e-4`, the highest-accuracy experiment in
the local leaderboard. Their softmax anemia scores are averaged; threshold 0.5
is a default decision rule, not a clinically validated cutoff. This new ensemble
has not itself been independently validated on phone-camera captures.

For the existing model, inference uses **rectangular `roi.png`**, matching that experiment's training
input, RGB conversion, 224×224 validation resize and ImageNet normalization.
The CSV's `prediction_input_path` links to `roi.png` for this model, while
`roi_image_path` links to `roi_masked.png`. Changing the inference input to
masked tissue requires matching training/validation. New training runs write a
checkpoint contract with ROI format, image size, normalization version, label map
and weight hash. Capture verifies all five contracts and selects rectangular or
masked input accordingly; mixed formats are rejected. Legacy weights use the
explicit `capture.inference.legacy_input_format` setting.

Choose **View model input** after scoring to inspect the exact ROI pixels sent
to the predictor before resizing. The original-pixels dialog lets you compare
that image with the tissue mask and the eye crop.

The app displays the model
prediction as a research result, not a diagnosis. Failed ROI extraction skips
inference. If capture quality fails but you confirm an extracted ROI, the
prediction is displayed and saved as `review_only`. Missing weights/dependencies show an unavailable
message and do not prevent saving.

The mean score determines the displayed **Anemic** or **Non-anemic** prediction.
When folds fall on opposite sides of the threshold, the app flags disagreement
and records the prediction with status `uncertain`. Saving without confirming the ROI is allowed for
inspection, but `crop_ok=False` keeps that capture out of label import/training.
Offline video processing cannot confirm anatomy and does not run inference by
default.

Frame selection retains four category winners plus at most eight alternatives
from quarter-second time windows. ROI extraction evaluates distinct candidates,
then ranks passing tissue quality above failed tissue quality. Focus is measured
inside an eroded tissue mask, excluding the artificial mask boundary, with minimum
native tissue dimensions/pixel count. These additional gates need device-specific
validation; a successful mask is not proof of correct anatomy.

Phone mute events and missing video-frame callbacks stop an in-progress capture
instead of treating a paused connection as a working camera. The existing review
is preserved if the camera pauses after processing.

Automatic patient numbering checks published and staged captures missing from
the CSV, reserving their patient IDs after an interrupted save. Malformed orphan
metadata blocks automatic numbering rather than guessing. Orphan captures are
preserved for recovery; they are not silently inserted into the CSV or deleted.

Weights are loaded once per session from local checkpoints. Saving a capture
records `prediction_status`, `prediction_label`, and `anemia_score` in the CSV.
The label and score are saved for all three scored outcomes:

- `ok` means the fold models agreed on the label at the configured threshold.
- `uncertain` means at least one fold model landed on each side of the threshold.
  The saved label is based on the average score.
- `review_only` means a score was generated, but the capture failed the quality
  check or the ROI was not confirmed. This takes precedence over fold disagreement,
  which is recorded separately in `metrics.json`.

The label and score are blank for unscored statuses. `needs_review` means the ROI
is awaiting confirmation. `skipped` means the input could not be scored.
`disabled` means inference was turned off. `unavailable` means inference failed.
`stale` means a previously scored ROI changed or failed verification.
The threshold, fold scores and checkpoint hashes are saved in `metrics.json`.
Training still requires independently supplied labels.
Settings are under `capture.inference`. Restart the capture app after updates.
ROI regeneration invalidates any old saved prediction.

Training now verifies settings, exact image bytes, labels, patient splits and
preprocessing code before reusing completed folds or resuming. Unverifiable old
runs require a **new output directory** or explicit `--fresh`; existing trained
weights remain usable for inference. A rejected experiment does not overwrite its
effective configuration. `--fresh` replaces training artifacts, so use a new output
directory to retain your current best model.

## Measuring the remaining ROI and model limitations

`scripts/validate_capture.py` writes a new JSON report without modifying captures,
model weights, decision thresholds or existing reports. Paths in input manifests
are resolved relative to the manifest. Parent output directories must exist.

To check whether saved results can be reproduced, start the capture app once to
upgrade an older capture index, then run:

```powershell
.\.venv\Scripts\python.exe scripts/validate_capture.py replay --manifest outputs/captures/dataset.csv --output capture_replay.json
```

Replay verifies the saved image and checkpoint hashes, runs the same capture
predictor, and checks the label and score against the CSV. It reports saved
captures with no score and exits with an error if a scored row fails verification.
Replay also checks the individual fold scores and their disagreement when recorded.

For ROI validation, supply a CSV with `image_id,eye_image_path,reference_mask_path,group`.
Each reference is a manually labeled binary mask at exactly the eye-crop dimensions.
Optional groups such as `pale`, `pink` or camera names allow comparison. Failures
count as zero overlap; the report includes Dice, IoU and tissue recall by group.
This measures segmentation on supplied crops, not whole-frame eye detection.

```powershell
.\.venv\Scripts\python.exe scripts/validate_capture.py roi --manifest roi_labels.csv --output roi_validation.json
```

For ensemble validation, supply `patient_id,split,label,metrics_path` where split
is `calibration` or `test`, and label is an independently confirmed `0` or `1`.
Use stable patient identities shared with the training metadata; automatically
generated capture IDs do not establish that two datasets contain different people.
Include every training patient in `--training-metadata`. The tool rejects overlap,
duplicate captures, conflicting labels, unconfirmed ROIs and mixed checkpoints.
Multiple captures are averaged per patient. Calibration and threshold selection
use only calibration patients; untouched test patients are reported separately.
The raw test report uses the threshold recorded with the saved predictions,
falling back to 0.5 for older metadata without a threshold.
Each split needs both classes (at least two patients per class to run; this minimum
is not sufficient evidence of clinical reliability).

```powershell
.\.venv\Scripts\python.exe scripts/validate_capture.py ensemble --manifest validation_labels.csv --training-metadata training_patients.csv --output ensemble_validation.json
```

For a real-device trial, use one manifest row per attempt, including failures.
Keep `metrics_path` blank for discarded or unsaved attempts and pass
`--include-unscored`. The report includes quality pass rates and scored capture
and patient counts per split alongside accuracy metrics for eligible captures.
Try the phone and webcam under the same lighting, then repeat with changes in
lighting and movement. Confirm the tissue preview before scoring.

The report compares raw and calibrated scores and saves calibration coefficients
for review. It deliberately does not install them into the app: this is a
patient-mean evaluation, whereas live capture predicts individual captures.
Labeled reference masks and an independent patient cohort are still needed to
measure real improvements; software tests alone cannot establish accuracy.

The default index is `outputs/captures/dataset.csv`. Every Save adds one row with
an automatic `serial_id` (1, 2, 3, ...). New captures use IDs `CAP_000001`,
`CAP_000002`, etc. Numbering continues across restarts and devices sharing that
output folder. Retake does not consume an ID. IDs are independent of patient IDs.

The CSV paths are relative to the directory containing `dataset.csv`:

| CSV column | Example |
| --- | --- |
| `capture_dir` | `CAP_000001` |
| `roi_image_path` | `CAP_000001/roi_masked.png` (blank when extraction fails) |
| `prediction_input_path` | `CAP_000001/roi.png` (blank without a verified score) |

The index upgrade fills the new column only when the saved prediction and both
image hashes verify. It keeps the original index as a backup.

`outputs/captures/CAP_000001/` contains:

| File | Purpose |
| --- | --- |
| `frame.png` | Clean full-resolution final selected frame |
| `eye.png` | Clean detected eye crop |
| `roi.png` | Tight rectangular crop around segmented lower-lid tissue |
| `roi_masked.png`, `roi_mask.png` | Tissue-only pixels on black, and the binary segmentation mask |
| `roi_debug.png` | Four panels: eye with bounding box, mask, rectangular ROI, masked ROI |
| `eye_search_mask.png` | Detector's lower-lid search region, when available |
| `review.png`, `candidate_*_review.png` | Annotated review images; never training inputs |
| `metrics.json` | Candidate scores, frame indices, timing and capture metadata |

`outputs/captures/dataset.csv` also records `patient_id`, `roi_status` and `crop_ok`.
Device, distance, lighting and quality scores remain in the index.
When the app opens an older index, it backs up the CSV and moves each indexed
capture to a single folder under the output directory, preserving its ID and files.
New saves keep annotated candidate review images. `frame.png` and `eye.png` remain
clean source images. All four ROI
outputs (`roi.png`, `roi_masked.png`, `roi_mask.png`, `roi_debug.png`) remain.
The CSV's `roi_image_path` points to the masked image, so labelled imports train
on masked tissue.
To remove recordings and duplicate candidates from existing indexed captures:

```powershell
.\.venv\Scripts\python.exe scripts/compact_captures.py
```

Cleanup deletes a candidate original only if its matching review exists. Older
annotated candidates without a separate review are retained. Saved eye geometry
supports ROI regeneration without a recording; very old exports lacking geometry
must be repaired before their recording is removed.
Compaction stages affected files until the CSV is written. If Excel blocks
the CSV update, it restores the original capture files. Close Excel before
retrying.
After an interruption, startup restores the originals or finishes cleanup
according to the CSV state. It records the result in `recovery_report.json`.
Unjournaled staging folders from earlier versions are retained for review.
Copied capture folders can be reindexed when their local images match the
saved hashes.
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
CSV schemas, missing/extra columns and duplicate capture IDs fail before a save
can rewrite the index. UTF-8 CSVs saved with Excel's byte-order mark are accepted.
With no detected/selected eye,
the crop and ROI columns stay empty and `roi_status=no_eye` is recorded.

To regenerate a previous capture's ROI with the new extractor:

```powershell
.\.venv\Scripts\python.exe scripts/reextract_capture_roi.py --capture-dir outputs/captures/<capture_id>
```

This updates its ROI images, QA preview and existing CSV row, preserving the
old outputs and metadata in a `roi_backup_*` folder. For older captures without
saved eye geometry, it replays the recording and requires the same selected
frame and exact eye pixels before proceeding. Original frames are not modified.
A successful repair recomputes `crop_ok` using the original recorded quality
gate. It can clear a segmentation-only failure, but cannot turn a blurred or
poorly exposed original into a passing capture.

Patient and image IDs remain exact strings through import, batch ROI extraction
and training, including leading zeros and literal IDs such as `NA`. Training
rejects empty patient IDs before making patient-level splits.

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
The importer checks the frame and ROI against their hashes in `metrics.json`.
Captures with missing inventories or changed images must be recovered from the
original verified files or captured again before import.

Copy `configs/config.yaml` to a separate capture-training config, set
`roi_extraction.output_csv` to `outputs/metadata/captures_with_roi.csv`, retain
`training.image_source: with_roi`, and use a separate `training.output_dir`.
Run `scripts/train_baseline.py --config <that-config>` once there are enough
labelled patients in both classes for the configured folds. Do not run the
metadata/ROI rebuild stages against this capture manifest; they target the
original folder-and-spreadsheet dataset.

For existing trained-model inference on an individual saved capture:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --input outputs/captures/<capture_id>/roi.png
```

Review `roi_status` first. Do not pass the entire capture folder to prediction:
it also contains annotated candidate images.
Offline prediction uses the configured capture model and threshold by default
and requires all five checkpoints. Use `--fold` for a single fold or
`--threshold` to explicitly override the configured threshold.

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
It also verifies actual 1080p fake-camera output, lossless crop previews and the
1:1 detail dialog. Unit tests check source-pixel scoring, candidate review exports,
native-resolution highlight removal and unchanged tissue pixels after cropping.

Haar is available with the project's OpenCV install. Optional MediaPipe is used
only when its installed package exposes `solutions.face_mesh`; otherwise Haar is
used. MediaPipe is not installed or version-pinned by this integration. Haar can
miss very close, tilted or occluded eyes; the 13 existing close-up photographs
produced 8 detections in a local check, but visual inspection found several
boxes localized to an iris/eyelash fragment rather than the complete eye. Use
manual region selection on such close-ups. Detection counts are not accuracy.
Real moving-eye clips and physical camera autofocus still need device testing.

The November 14 close-up audit found blue iris false positives. Capture ROI v3
adds a red/pink hue-family gate at segmentation and native boundary refinement.
This is an additional rejection heuristic, not an anatomical classifier. Reference
masks are still required to measure overlap/accuracy across pale tissue and skin
tones. Use manual region selection and inspect the tissue preview when automatic
eye localization misses the lower lid.

Live guidance now receives native-size JPEG frames (quality 95) and uses the same
source-crop scorer as recorded-frame selection. Detection remains at 640 pixels;
one request runs at a time with a 150 ms pause. JPEG/video compression can still
cause small score differences. Cleanup preserves recordings for captures missing
saved eye geometry and reports their paths instead of deleting their recovery input.

Historical timing check before original-crop scoring was added (2026-11-11,
single run, CPU): a 90-frame MJPG clip at 30 fps,
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
