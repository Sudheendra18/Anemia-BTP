# CP-AnemiC research pipeline

This project prepares conjunctiva images, trains and benchmarks ResNet-50 and
EfficientNet-B0, and runs the selected EfficientNet-B0 experiment in an eye
capture app. The app produces research predictions. It has not been validated
for clinical use.

## Run the capture app

From the project root in Windows PowerShell, run:

```powershell
.\.venv\Scripts\python.exe scripts/capture.py
```

You can also open [start_capture.cmd](start_capture.cmd) on Windows.

## First-time setup

Clone `main` and install the dependencies. These commands use
Windows PowerShell and Python 3.12.

```powershell
git clone --single-branch --branch main https://github.com/Sudheendra18/Anemia-BTP.git
cd Anemia-BTP
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-capture.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

On a fresh clone, extract the five EfficientNet-B0 checkpoints from the
[included model archive](outputs/experiments/efficientnet_b0_lr1e-4.zip). Only
the checkpoint files are extracted.

```powershell
$model = 'efficientnet_b0_lr1e-4'
$weights = 1..5 | ForEach-Object { "$model/fold_$_/best_model.pt" }
tar -xf "outputs/experiments/$model.zip" -C outputs/experiments $weights
.\.venv\Scripts\python.exe scripts/capture.py --check-setup
.\.venv\Scripts\python.exe scripts/capture.py
```

On macOS or Linux, use `python3` to create `.venv` and `.venv/bin/python` to
install dependencies and run the app. Extract the checkpoints with:

```sh
unzip -n outputs/experiments/efficientnet_b0_lr1e-4.zip 'efficientnet_b0_lr1e-4/fold_*/best_model.pt' -d outputs/experiments
```

Capture and inference run on a laptop with a webcam and use the saved model on
the CPU. The raw training dataset and retraining are unnecessary for the demo.
`--check-setup` checks the OpenCV eye detector and Chromium installation, then
runs a CPU prediction on a synthetic image when inference is enabled. It does
not open the camera or create a capture folder.
Paths and settings are defined in [configs/config.yaml](configs/config.yaml).
Relative configuration paths are resolved from the project root.

### Fix a missing OpenCV eye detector

If capture fails with `module 'cv2' has no attribute 'CascadeClassifier'`,
close the app and check which OpenCV module the project environment loads.
Run this from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print(cv2.__file__); print(hasattr(cv2, 'CascadeClassifier'))"
```

The path should point inside `.venv` and the last line should print `True`.
If the path points to a local `cv2.py` or `cv2` folder outside `.venv`, rename
that file or folder so Python can load OpenCV from the environment.

If the check prints `False` for the installed package or the import fails,
reinstall the standard OpenCV package. The variants below share the `cv2`
module, so keep only one in the project environment.

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
.\.venv\Scripts\python.exe -m pip install --no-deps opencv-python==4.11.0.86
```

Repeat the check and restart the capture app once it prints `True`.

## Capture and score an eye image

Choose your camera, record an eye clip and review the selected
conjunctiva crop. Use **Confirm ROI & predict anemia** to score the crop.
**View model input** shows the exact image used by the predictor.

Saved captures are indexed in `outputs/captures/dataset.csv`. The
`prediction_input_path` column identifies the verified image used for inference.
See [the capture guide](docs/CAPTURE.md) for controls, status codes, saved files
and replay validation.

## Prepare data and train

Place the raw images in `data/raw/Anemic/` and `data/raw/Non-anemic/`.
The default metadata source is `data/Anemia_Data_Collection_Sheet.xlsx`.
See the [data contract](docs/DATA_CONTRACT.md) for the expected schema and
[ROI tuning](docs/DATA_CONTRACT.md#tuning-roi-extraction) for crop review.

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py
.\.venv\Scripts\python.exe scripts/train_baseline.py
```

The pipeline writes metadata, reconciliation issues and ROI paths under
`outputs/metadata/`. Review the issues CSV and the sample crops before training.
The CNN takes image pixels as input. Hemoglobin values in the metadata are
not passed to the model.

| Task | Entry point |
| --- | --- |
| Compare experiment configurations | [scripts/run_experiments.py](scripts/run_experiments.py) |
| Compare training with and without augmentation | [scripts/train_no_augmentation.py](scripts/train_no_augmentation.py) |
| Analyze decision thresholds from saved folds | [scripts/analyze_thresholds.py](scripts/analyze_thresholds.py) |
| Predict on a file or folder | [scripts/predict.py](scripts/predict.py) |
| Review ROI extraction on a folder | [scripts/test_roi_on_folder.py](scripts/test_roi_on_folder.py) |
| Evaluate or replay saved captures | [scripts/validate_capture.py](scripts/validate_capture.py) |

Use a new output directory when starting a different training run.
See [the training guide](docs/TRAINING.md) for configuration, resume behavior,
threshold analysis and the augmentation comparison.

## Results

The [experiment leaderboard](outputs/experiments/leaderboard.csv) records
cross-validation metrics for the saved experiments. EfficientNet-B0 at
`1e-4` had the highest mean accuracy in that table (86.62%). That number
comes from cross-validation and does not establish accuracy on new devices
or a separate patient cohort.

See the [augmentation ablation report](outputs/no_augmentation/ablation_report.md)
for the comparison with augmentation disabled.

See the [developer test commands](docs/CAPTURE.md#verification-and-limitations)
to run the unit suite or the simulated-camera check.
