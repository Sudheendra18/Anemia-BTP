# CP-AnemiC: Non-Invasive Anemia Data Pipeline & Deep Learning Benchmarking

A clinical-grade, end-to-end machine learning pipeline for non-invasive anemia screening from smartphone-captured palpebral conjunctiva images.

This repository covers the complete research workflow: from raw patient image validation and automated conjunctival region-of-interest (ROI) extraction, through patient-stratified deep learning cross-validation, post-hoc clinical threshold tuning, and direct clinical inference.

---

## Table of Contents

1. [Clinical Background & Why Conjunctiva?](#clinical-background--why-conjunctiva)
2. [End-to-End Pipeline Architecture](#end-to-end-pipeline-architecture)
3. [Repository Structure](#repository-structure)
4. [Dependencies & Import Requirements](#dependencies--import-requirements)
5. [Environment Setup & Installation](#environment-setup--installation)
6. [Data Preparation & Data Contract](#data-preparation--data-contract)
7. [Comprehensive File-by-File Guide](#comprehensive-file-by-file-guide)
   - [Core Engine (`src/`)](#core-engine-src)
   - [CLI Entry Points (`scripts/`)](#cli-entry-points-scripts)
8. [Configuration Reference (`configs/config.yaml`)](#configuration-reference-configsconfigyaml)
9. [Step-by-Step Usage Guide](#step-by-step-usage-guide)
   - [1. Data Pipeline (Stages 1 & 2)](#1-data-pipeline-stages-1--2)
   - [2. Model Training (Stage 3)](#2-model-training-stage-3)
   - [3. Post-Hoc Threshold Optimization](#3-post-hoc-threshold-optimization)
   - [4. Systematic Architecture Benchmarking](#4-systematic-architecture-benchmarking)
   - [5. Inference on New Images](#5-inference-on-new-images)
10. [Generated Outputs & Artifacts](#generated-outputs--artifacts)
11. [Clinical Modeling Best Practices & FAQ](#clinical-modeling-best-practices--faq)

---

## Clinical Background & Why Conjunctiva?

Anemia affects over 1.6 billion people globally. The gold standard for diagnosis requires an invasive venous blood draw analyzed via complete blood count (CBC) or a Hemocue photometer. In low-resource, rural, or point-of-care settings, frequent needle sticks carry infection risks, require skilled phlebotomists, and face consumable supply chain bottlenecks.

The **palpebral conjunctiva** (the vascular mucous membrane lining the inside of the lower eyelid) serves as an optimal anatomical window for non-invasive visual screening:
- **Rich Microvasculature**: The capillary bed is located just beneath a thin, transparent epithelial layer.
- **Absence of Melanin**: Unlike external epidermal skin, the conjunctiva has minimal melanocytes, minimizing racial and skin-tone pigmentation confounds.
- **Direct Pallor Correlation**: Decreased hemoglobin concentration reduces oxyhemoglobin absorption, causing visible pallor (whitening/loss of redness).

However, smartphone photos captured in the wild suffer from flash glare, variable illumination, eyelid positioning shifts, and eyelash shadows. This repository establishes a reproducible, automated processing pipeline designed to handle these real-world challenges reliably.

---

## End-to-End Pipeline Architecture

```
+-----------------------------------------------------------------------------------------+
|                                    INPUT DATA                                           |
|  +-------------------------------------+      +--------------------------------------+  |
|  | Raw Photos: data/raw/               |      | Clinical Metadata:                   |  |
|  |   - Anemic/ (Image_001.jpg, ...)    |      |   data/Anemia_Data_Collection_Sheet  |  |
|  |   - Non-anemic/ (Image_002.jpg, ...) |      |   (.xlsx containing Hb, age, sex)    |  |
|  +-------------------------------------+      +--------------------------------------+  |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| STAGE 1: Dataset Loader & Audit Trail (src/dataset_loader.py)                           |
|   * Case- and separator-insensitive fuzzy ID normalization (Image_001 == image-001)     |
|   * Reconciles on-disk image location against Excel clinical labels                     |
|   * Flags label conflicts, missing rows, and uncaptured entries to cp_anemic_issues.csv  |
|   * Emits validated metadata -> outputs/metadata/cp_anemic_metadata.csv                 |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| STAGE 2: Conjunctiva ROI Extraction (src/roi_extraction.py)                             |
|   * Multi-space Chromatic Analysis: Lab (a*, b*), YCrCb (Cr), HSV (Hue/Sat), Excess Red |
|   * Composite Mucosal Redness Index (MRI) suppresses skin yellowness & flash glare      |
|   * Adaptive ocular thresholding + Horizontal palpebral crescent morphological bridging |
|   * Connected component scoring based on anatomical position and aspect ratio           |
|   * Automatic fallback center-crop failsafe (prevents pipeline crashes on tough photos) |
|   * Emits tight crops -> outputs/roi/ and debug QA composites -> outputs/debug/         |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| STAGE 3: Transfer-Learning CNN & Evaluation (src/train.py)                              |
|   * StratifiedGroupKFold on patient_id (guarantees zero patient data leakage across CV)  |
|   * ResNet-50 and EfficientNet-B0 architectures with pre-trained ImageNet backbones     |
|   * Clinical class weighting to balance 424 Anemic vs 286 Non-anemic distribution       |
|   * Domain-safe augmentations (rotation, jitter, horizontal flip - no vertical flips)   |
|   * Early stopping and checkpoint selection driven by F2 Score (weights recall)         |
|   * Crash-resilient state checkpointing (safe to interrupt and resume instantly)        |
+--------------------------------------------+--------------------------------------------+
                      |                                             |
                      v                                             v
+-------------------------------------------+ +-------------------------------------------+
| POST-HOC THRESHOLD TUNING                 | | SYSTEMATIC EXPERIMENT BENCHMARKING        |
| (src/threshold_analysis.py)               | | (src/experiment_runner.py)                |
| * Pools out-of-fold validation probas     | | * Automated multi-architecture grid search|
| * Sweeps cutoffs from 0.05 to 0.95        | | * Compares ResNet-50 vs EfficientNet-B0   |
| * Identifies Youden's J & Sensitivity     | | * Ranks runs by primary metric into       |
|   floor (e.g. >=90% recall) thresholds    | |   outputs/experiments/leaderboard.csv     |
+-------------------------------------------+ +-------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| DIRECT CLINICAL INFERENCE (scripts/predict.py)                                          |
| * Direct prediction on single images or entire unorganized folders                      |
| * Optional on-the-fly ROI extraction for raw smartphone photos                          |
| * Multi-model ensemble averaging across all trained fold checkpoints                    |
| * Produces per-image probabilities, classifications, and CSV summary reports            |
+-----------------------------------------------------------------------------------------+
```

---

## Repository Structure

```
anemia-data-pipeline/
|
├── configs/
|   └── config.yaml             # Single source of truth for all paths, thresholds, and hyperparameters
|
├── data/
|   ├── Anemia_Data_Collection_Sheet.xlsx   # Official clinical patient metadata
|   ├── raw/                                # Raw source images (Anemic/ and Non-anemic/ subfolders)
|   └── uncropped/                          # Optional folder for unlabelled evaluation images
|
├── docs/
|   ├── DATA_CONTRACT.md        # Exact schema definition for generated metadata CSVs
|   ├── ROI_TUNING.md           # Visual troubleshooting guide for color segmentation thresholds
|   └── TRAINING.md             # In-depth training documentation, GPU advice, and metric logs
|
├── outputs/                    # Generated automatically (git-ignored)
|   ├── metadata/               # Clean metadata, issues log, and ROI-enriched metadata
|   ├── roi/                    # Extracted conjunctiva crops (Anemic/ and Non-anemic/)
|   ├── debug/                  # 4-panel QA composites (original, mask, crop, masked crop)
|   ├── training/               # Baseline CV fold checkpoints, metrics tables, and evaluation plots
|   ├── experiments/            # Multi-run benchmark results and comparison leaderboards
|   ├── predictions/            # Inference output CSVs from scripts/predict.py
|   └── logs/                   # Dedicated log file for each script execution
|
├── scripts/                    # Command-line entry points
|   ├── run_pipeline.py         # Runs Stage 1 and Stage 2 end-to-end
|   ├── train_baseline.py       # Runs Stage 3 k-fold cross-validation
|   ├── analyze_thresholds.py   # Runs post-hoc threshold optimization on trained folds
|   ├── run_experiments.py      # Runs systematic architecture & learning rate comparison
|   ├── predict.py              # Runs inference on unorganized images or folders
|   └── test_roi_on_folder.py   # Standalone utility to test ROI extraction on any folder
|
├── src/                        # Modular pipeline implementation
|   ├── dataset.py              # PyTorch Dataset class and clinical augmentation transforms
|   ├── dataset_loader.py       # Stage 1: folder scanner and Excel reconciler
|   ├── experiment_runner.py    # Multi-run experiment orchestrator and leaderboard compiler
|   ├── metrics.py              # Clinical metrics (F1, F2, AUC, sensitivity, specificity) & plots
|   ├── model.py                # Model factory for ResNet-50 and EfficientNet-B0
|   ├── roi_extraction.py       # Stage 2: multi-space chromaticity ROI segmentation
|   ├── threshold_analysis.py   # Out-of-fold threshold sweeping and recommendation engine
|   ├── train.py                # K-fold training loop, AMP, class weighting, and early stopping
|   └── utils.py                # Shared configuration loader, path resolver, and logging setup
|
├── requirements.txt            # Locked minimum versions of project dependencies
└── README.md                   # Comprehensive project documentation
```

---

## Dependencies & Import Requirements

All project dependencies are declared in [requirements.txt](requirements.txt). Here is why each specific library is required:

| Package | Version Requirement | Purpose in This Pipeline |
| :--- | :--- | :--- |
| **`torch`** | `>=2.2` | Core deep learning framework. Powers neural network operations, tensor computations, automatic differentiation (`autograd`), GPU acceleration, and Automatic Mixed Precision (`torch.amp`). |
| **`torchvision`** | `>=0.17` | Computer vision library providing pre-trained CNN backbones (`ResNet50_Weights`, `EfficientNet_B0_Weights`) and image transformations (`Resize`, `ColorJitter`, `RandomRotation`, `Normalize`). |
| **`opencv-python`** | `>=4.8` | High-performance computer vision engine. Used for reading images, converting across color spaces (`BGR` $\leftrightarrow$ `LAB`, `HSV`, `YCrCb`), morphological operations (`cv2.morphologyEx`), and connected component analysis. |
| **`pandas`** | `>=2.0` | High-performance tabular data manipulation. Manages dataset records, joins disk filenames with clinical Excel records, pools out-of-fold predictions, and exports summary tables. |
| **`openpyxl`** | `>=3.1` | Excel reader engine. Enables `pandas.read_excel` to parse clinical metadata spreadsheets (`.xlsx`) without needing Microsoft Excel installed. |
| **`PyYAML`** | `>=6.0` | Safe YAML parser. Loads and validates [configs/config.yaml](configs/config.yaml), ensuring no parameters are hard-coded in python scripts. |
| **`scikit-learn`** | `>=1.2` | Machine learning evaluation suite. Provides `StratifiedGroupKFold` for patient-isolated cross-validation, plus metric routines (`roc_auc_score`, `confusion_matrix`, `fbeta_score`, `precision_recall_curve`). |
| **`matplotlib`** | `>=3.7` | Plotting library configured in headless mode (`matplotlib.use("Agg")`). Renders publication-grade ROC curves, PR curves, confusion matrices, and training curves directly to PNG files. |
| **`Pillow`** | `>=10.0` | Python Imaging Library. Serves as the standard bridge between OpenCV numpy arrays and torchvision's PIL-based image transforms. |
| **`tqdm`** | `>=4.65` | Terminal progress bars for batch processing during multiprocessing ROI extraction and model training. |
| **`numpy`** | `>=1.24` | Foundational array computing. Handles numerical masks, percentiles for adaptive thresholding, and vector arithmetic in chromatic calculations. |

---

## Environment Setup & Installation

### Prerequisites
- **Python**: Version `3.10`, `3.11`, or `3.12` recommended.
- **Hardware**: Compatible with CPU, Apple Silicon (MPS), and NVIDIA GPUs (CUDA).

### 1. Clone the Repository
```bash
git clone https://github.com/Sudheendra18/Anemia-BTP.git
cd Anemia-BTP
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. GPU Acceleration Check (Optional but Recommended)
PyTorch will automatically detect and leverage CUDA if available. Verify your GPU configuration:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Note for NVIDIA GPU users**: If `torch.cuda.is_available()` returns `False`, install the CUDA-enabled PyTorch wheel directly from [pytorch.org](https://pytorch.org/get-started/locally/) for your installed driver version (e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

---

## Data Preparation & Data Contract

### Directory Layout
The data pipeline expects raw patient photographs organized by clinical class. By default, paths are configured relative to the project root, but absolute paths anywhere on your filesystem can be specified in [configs/config.yaml](configs/config.yaml).

```
data/raw/
├── Anemic/
│   ├── Image_001.jpg
│   ├── Image_002.jpg
│   └── ...
└── Non-anemic/
    ├── Image_003.jpg
    ├── Image_004.jpg
    └── ...
```

### Clinical Spreadsheet
Place your clinical metadata Excel sheet at `data/Anemia_Data_Collection_Sheet.xlsx`.
The Excel file must contain the clinical columns defined in `configs/config.yaml`:
- `IMAGE_ID`: Patient/image identifier (e.g. `Image_001`).
- `HB_LEVEL`: Blood hemoglobin level in g/dL.
- `Severity`: Qualitative severity (`Non-Anemic`, `Mild`, `Moderate`, `Severe`).
- `Age(Months)`: Age in months.
- `GENDER`: Patient sex.
- `REMARK`: Clinical diagnosis (`Anemic` or `Non-anemic`).
- `HOSPITAL`, `CITY/TOWN`, `MUNICIPALITY/DISTRICT`, `REGION`, `COUNTRY`: Geographical metadata.

### Fuzzy Identifier Matching
Filenames are matched against the Excel `IMAGE_ID` column in a case- and separator-insensitive manner:
- `Image_001.jpg`, `image001.PNG`, `IMAGE-001.jpeg`, and `image 001.bmp` all resolve identically to the record `Image_001`.
- You do **not** need to manually rename image files to match Excel capitalization or hyphens.

### Issues Tracking (`outputs/metadata/cp_anemic_issues.csv`)
If any image or metadata entry is inconsistent, it is logged to `outputs/metadata/cp_anemic_issues.csv` rather than silently failing:
- `folder_excel_label_mismatch`: The image is in the `Anemic/` folder, but Excel says `Non-anemic` (or vice-versa).
- `image_without_excel_row`: An image exists on disk with no matching Excel row.
- `excel_row_without_image`: An Excel row exists with no corresponding photo on disk.

---

## Comprehensive File-by-File Guide

### Core Engine (`src/`)

#### [`src/utils.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/utils.py) — Pipeline Utilities & Infrastructure
Provides shared plumbing used across all pipeline stages so that scripts behave consistently regardless of the working directory they are invoked from.
- **`PROJECT_ROOT`**: Automatically anchors to the root directory containing `src/`.
- **`load_config(config_path)`**: Loads `config.yaml` using PyYAML and resolves relative paths against the project root.
- **`resolve_path(path_str)`**: Converts relative config paths into verified absolute `Path` objects.
- **`setup_logging(log_file, level)`**: Initializes unified dual console and file logging with standardized timestamps (`HH:MM:SS | LEVEL | message`).
- **`ensure_parent_dir(path)`**: Resolves a path and ensures all parent directories exist before writing.

#### [`src/model.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/model.py) — Neural Network Architecture Factory
Initializes transfer-learning CNN backbones tailored for binary clinical classification.
- **`build_model(architecture, num_classes, pretrained, freeze_backbone)`**:
  - `resnet50`: Loads ResNet-50 (`IMAGENET1K_V2` weights) and replaces `model.fc` with a new `nn.Linear(2048, num_classes)` head.
  - `efficientnet_b0`: Loads EfficientNet-B0 (`IMAGENET1K_V1` weights) and replaces `model.classifier[1]` with `nn.Linear(1280, num_classes)`.
  - Supports backbone freezing (`freeze_backbone=True`) to train only the classifier head, drastically reducing memory footprint and training time.
- **`count_trainable_params(model)`**: Computes trainable vs. total parameter counts to verify freezing behavior.

#### [`src/dataset.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/dataset.py) — PyTorch Dataset & Clinical Transforms
Interfaces between the pandas metadata table and PyTorch `DataLoader`.
- **`ConjunctivaDataset(Dataset)`**: Reads image paths directly from the metadata DataFrame (`image_path` for raw photos or `roi_image_path` for crops). Loads via OpenCV, converts BGR $\rightarrow$ RGB, creates PIL images, applies transforms, and returns `(tensor, label)`.
- **`build_transforms(image_size, train, augmentation)`**:
  - **Validation/Inference**: Deterministic `Resize((image_size, image_size))`, `ToTensor()`, and standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - **Training**: Clinically appropriate data augmentation:
    - `RandomHorizontalFlip(p=0.5)` (eyes have bilateral symmetry).
    - `RandomRotation(degrees=15)` (accounts for head tilt).
    - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` (accounts for camera exposure differences).
    - **No vertical flipping**: Unlike generic scene photos, conjunctiva pull-down photos have a fixed anatomical up/down orientation; vertical flipping breaks domain realism.
    - Optional `RandomErasing`: Regularizer that masks random patches to prevent reliance on single localized artifacts.

#### [`src/dataset_loader.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/dataset_loader.py) — Stage 1 Metadata Builder
Scans image directories, loads the Excel sheet, cross-references IDs, and creates the primary data contract.
- **`_normalize_id(raw)`**: Strips non-alphanumeric characters and lowercases strings for fuzzy matching.
- **`scan_image_folder(folder, extensions, folder_label)`**: Recursively scans class folders and records valid image files.
- **`build_metadata(config, logger)`**: Reconciles disk files with the Excel sheet, audits label agreement, builds `outputs/metadata/cp_anemic_metadata.csv`, and writes anomalies to `outputs/metadata/cp_anemic_issues.csv`.

#### [`src/roi_extraction.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/roi_extraction.py) — Stage 2 ROI Extraction Engine
Extracts the palpebral conjunctiva using robust multi-space color segmentation and anatomical priors.
1. **Multi-Space Color Decomposition**: Analyzes CIELAB ($a^*$ redness, $b^*$ yellowness), YCrCb ($Cr$ red chroma), and HSV (Hue and Saturation).
2. **Mucosal Redness Index (MRI)**: Computes a composite redness index:
   $$\text{MRI} = (a^* - 128) + (Cr - 128) + 80 \cdot (R_{\text{norm}} - G_{\text{norm}}) + 0.15 \cdot S - 0.25 \cdot (b^* - 128)$$
   This accentuates vascular mucosal redness while actively subtracting epidermal skin yellowness and rejecting white flash glare.
3. **Adaptive Thresholding**: Dynamically computes threshold cutoffs based on the 95th and 68th percentiles of redness within the lower-central ocular zone.
4. **Horizontal Morphological Bridging**: Applies an elliptical horizontal kernel matching the anatomical shape of the palpebral conjunctival arch.
5. **Component Scoring**: Evaluates candidate blobs using area, vertical positioning prior (favoring the lower eyelid), aspect ratio, and mean redness density.
6. **Failsafe Center-Crop Fallback**: If an image is heavily underexposed, blurred, or occluded, the extractor falls back to an anatomical lower-central crop, marking the status as `fallback` so the batch run never crashes.
7. **Debug Composites**: Exports 4-panel visual QA composites (`original + bbox | mask | rect crop | masked crop`).

#### [`src/metrics.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/metrics.py) — Clinical Evaluation & Diagnostics
Calculates performance metrics and generates publication-grade visualizations.
- **`compute_metrics(y_true, y_pred, y_prob)`**: Computes Accuracy, Sensitivity (Recall on Anemic), Specificity (True Negative Rate), F1 score, F2 score, and ROC-AUC.
- **Diagnostic Plotting Functions**:
  - `plot_confusion_matrix()`: Heatmap with exact count overlays for TN, FP, FN, TP.
  - `plot_roc_curve()`: True Positive Rate vs False Positive Rate with AUC score.
  - `plot_pr_curve()`: Precision vs Recall with Average Precision (AP) score.
  - `plot_training_curves()`: Side-by-side loss and validation metric trajectories across epochs.
  - `plot_threshold_sweep()`: Sensitivity, Specificity, and F2 curves across candidate cutoffs.
- **`sweep_thresholds(y_true, y_prob)`**: Evaluates complete confusion matrix and metrics for thresholds from 0.05 to 0.95 in 0.01 increments.
- **`recommend_thresholds(sweep_df, sensitivity_floor)`**:
  - `max_f2`: Optimizes F2 score.
  - `max_youden_j`: Maximizes Youden's $J = \text{Sensitivity} + \text{Specificity} - 1$ (best diagnostic balance).
  - `max_specificity_at_sensitivity_floor`: Maximizes specificity while guaranteeing sensitivity remains above a clinical threshold (e.g. $\ge 0.90$).
- **`aggregate_fold_metrics(fold_metrics)`**: Aggregates metrics across folds into a summary table with mean and standard deviation rows.

#### [`src/train.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/train.py) — Stage 3 Cross-Validation Training Engine
Orchestrates patient-level stratified cross-validation with early stopping and crash resilience.
- **`load_metadata_and_splits(config, logger)`**: Loads metadata and uses `StratifiedGroupKFold` on `patient_id` to strictly prevent patient data leakage across folds.
- **`run_single_fold()`**: Trains a single fold:
  - Supports Adam or SGD optimizers with weight decay.
  - Cosine annealing or ReduceLROnPlateau learning rate schedulers.
  - Automatic mixed precision (AMP fp16) on CUDA devices.
  - Inverse frequency class weighting to handle label imbalance.
  - Early stopping monitoring validation F2 score.
  - **Crash-resilient checkpointing**: Saves `resume_state.pt` after every epoch; if interrupted, it resumes mid-fold seamlessly without losing progress.
- **`run_kfold_cv(config, logger, fresh)`**: Manages all $k$ folds, skips already-completed folds, and writes `baseline_results_table.csv`.

#### [`src/threshold_analysis.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/threshold_analysis.py) — Zero-Retraining Threshold Optimizer
Tunes the decision cutoff using already-trained model checkpoints without retraining.
- Pools out-of-fold (OOF) validation predictions across all $k$ folds.
- Evaluates decision boundaries against clinical criteria.
- Emits `oof_predictions.csv`, `threshold_sweep.csv`, and updated confusion matrix plots.

#### [`src/experiment_runner.py`](file:///c:/Users/sudhe/anemia-data-pipeline/src/experiment_runner.py) — Systematic Benchmark Orchestrator
Automates multi-configuration benchmarking across architectures and hyperparameter combinations.
- Runs experiments defined under `experiments.runs` in `config.yaml`.
- Isolates outputs per experiment in `outputs/experiments/<name>/`.
- Produces an aggregated `leaderboard.csv` ranked by validation performance.

---

### CLI Entry Points (`scripts/`)

- **[`scripts/run_pipeline.py`](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/run_pipeline.py)**:
  Main entry point for Stages 1 & 2. Accepts `--stage all`, `--stage metadata`, or `--stage roi`, and optional `--config`.
- **[`scripts/train_baseline.py`](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/train_baseline.py)**:
  CLI runner for Stage 3 cross-validation training. Supports `--fresh` to force retraining from scratch.
- **[`scripts/analyze_thresholds.py`](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/analyze_thresholds.py)**:
  Runs threshold optimization on existing checkpoints. Accepts `--sensitivity-floor 0.90`.
- **[`scripts/run_experiments.py`](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/run_experiments.py)**:
  Executes the full experiment grid defined in `config.yaml` and outputs the leaderboard.
- **[`scripts/predict.py`](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/predict.py)**:
  Direct inference engine. Evaluates individual photos or entire folders of raw or cropped images with multi-fold ensemble averaging.
- **[`scripts/test_roi_on_folder.py`](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/test_roi_on_folder.py)**:
  Diagnostic testing utility to run ROI extraction on any arbitrary directory and inspect debug composites.

---

## Configuration Reference (`configs/config.yaml`)

All parameters are centrally managed in [configs/config.yaml](configs/config.yaml). The table below summarizes the key options:

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **`dataset`** | `raw_images_dir` | `"data/raw"` | Root folder containing class sub-folders (`Anemic/`, `Non-anemic/`). |
| | `metadata_excel_path` | `"data/Anemia_Data_Collection_Sheet.xlsx"` | Path to clinical Excel sheet. |
| | `image_extensions` | `[".jpg", ".jpeg", ".png", ".bmp"]` | Permitted image file extensions. |
| **`metadata_builder`**| `output_csv` | `"outputs/metadata/cp_anemic_metadata.csv"` | Clean metadata output path. |
| | `issues_csv` | `"outputs/metadata/cp_anemic_issues.csv"` | Anomaly and mismatch log output path. |
| | `drop_conflicts` | `false` | If true, drops label mismatches from the training metadata. |
| **`roi_extraction`** | `output_dir` | `"outputs/roi"` | Directory where cropped conjunctiva images are saved. |
| | `output_csv` | `"outputs/metadata/cp_anemic_metadata_with_roi.csv"` | Metadata enriched with crop paths and extraction status. |
| | `debug_dir` | `"outputs/debug"` | Folder for 4-panel visual QA composites. |
| | `debug_sample_size` | `20` | Number of random debug images to generate per run. |
| | `resize_long_side` | `800` | Working resolution for ROI segmentation. |
| | `fallback_center_crop_frac` | `0.6` | Box scale used if color segmentation fails. |
| **`training`** | `image_source` | `"with_roi"` | `"with_roi"` (crops) or `"raw"` (full images). |
| | `architecture` | `"resnet50"` | `"resnet50"` or `"efficientnet_b0"`. |
| | `freeze_backbone` | `false` | `false` = full fine-tuning; `true` = train classifier head only. |
| | `image_size` | `224` | Input resolution fed to the CNN. |
| | `batch_size` | `8` | Mini-batch size. |
| | `num_epochs` | `40` | Maximum training epochs per fold. |
| | `learning_rate` | `0.0001` | Initial optimizer learning rate. |
| | `weight_decay` | `0.0001` | L2 weight regularization penalty. |
| | `lr_scheduler` | `"cosine"` | `"cosine"`, `"plateau"`, or `"none"`. |
| | `use_class_weights`| `true` | Applies inverse-frequency class weights to cross-entropy loss. |
| | `early_stopping_patience` | `10` | Epochs without validation improvement before stopping. |
| | `mixed_precision` | `true` | Enables fp16 AMP on CUDA GPUs. |
| | `k_folds` | `5` | Number of cross-validation folds. |
| | `primary_metric` | `"f2"` | Metric that drives early stopping and checkpoint selection. |
| **`experiments`** | `output_dir` | `"outputs/experiments"` | Root output path for multi-run benchmarks. |

---

## Step-by-Step Usage Guide

### 1. Data Pipeline (Stages 1 & 2)
Run metadata reconciliation and conjunctiva ROI extraction:
```bash
# Run both Stage 1 and Stage 2 end-to-end
python scripts/run_pipeline.py

# Or run stages individually:
python scripts/run_pipeline.py --stage metadata
python scripts/run_pipeline.py --stage roi
```

**What to check after running:**
1. Check terminal output for the match count and any label mismatches.
2. Inspect `outputs/metadata/cp_anemic_issues.csv` to resolve any discrepancies.
3. Review sample composites in `outputs/debug/` to visually verify crop quality.

---

### 2. Model Training (Stage 3)
Train the baseline transfer-learning CNN using 5-fold cross-validation:
```bash
# Standard training (safe to interrupt and resume)
python scripts/train_baseline.py

# Force retrain all folds from scratch
python scripts/train_baseline.py --fresh
```

**Training features:**
- Uses GPU automatically if available.
- Saves the best checkpoint per fold immediately upon discovery (`best_model.pt`).
- Interrupted runs resume automatically from the last completed epoch.

---

### 3. Post-Hoc Threshold Optimization
Evaluate out-of-fold predictions to find the best clinical decision cutoff without retraining:
```bash
# Default threshold sweep with 90% sensitivity floor
python scripts/analyze_thresholds.py

# Custom sensitivity requirement (e.g. 85% or 95%)
python scripts/analyze_thresholds.py --sensitivity-floor 0.85
```

**Output:** Logs recommended thresholds for maximum F2, maximum Youden's J, and maximum specificity at your specified sensitivity floor.

---

### 4. Systematic Architecture Benchmarking
Run an automated multi-architecture comparison across ResNet-50 and EfficientNet-B0:
```bash
python scripts/run_experiments.py
```
View the final comparison in `outputs/experiments/leaderboard.csv`.

To tune decision thresholds against the winning experiment:
```bash
python scripts/analyze_thresholds.py --config outputs/experiments/<winning_experiment_name>/effective_config.yaml
```

---

### 5. Inference on New Images
Run predictions on new, unlabelled images using [scripts/predict.py](file:///c:/Users/sudhe/anemia-data-pipeline/scripts/predict.py):

```bash
# Predict on a folder of cropped conjunctiva images:
python scripts/predict.py --input path/to/cropped_images

# Predict on raw, uncropped eye photos (automatically crops the conjunctiva first):
python scripts/predict.py --input path/to/raw_photos --extract-roi

# Predict on a single image file:
python scripts/predict.py --input path/to/patient_eye.jpg --extract-roi

# Use a custom decision threshold and output destination:
python scripts/predict.py --input path/to/images --threshold 0.58 --output outputs/my_predictions.csv
```

---

## Generated Outputs & Artifacts

All outputs are saved to structured subdirectories inside `outputs/`:

```
outputs/
├── metadata/
│   ├── cp_anemic_metadata.csv           # Stage 1: Clean, validated metadata
│   ├── cp_anemic_issues.csv             # Stage 1: Logged discrepancies and missing files
│   └── cp_anemic_metadata_with_roi.csv  # Stage 2: Metadata enriched with ROI crop paths
|
├── roi/
│   ├── Anemic/                          # Stage 2: Cropped conjunctiva images
│   └── Non-anemic/
|
├── debug/                               # Stage 2: 4-panel visual QA composites
│   ├── debug_Image_001.jpg
│   └── ...
|
├── training/                            # Stage 3: Baseline model training artifacts
│   ├── baseline_results_table.csv       # Summary table across all folds (mean + std)
│   ├── oof_predictions.csv              # Pooled out-of-fold validation probabilities
│   ├── threshold_sweep.csv              # Metrics evaluated at cutoffs from 0.05 to 0.95
│   ├── threshold_sweep.png              # Sensitivity / Specificity tradeoff curve
│   └── fold_1/ ... fold_5/
│       ├── best_model.pt                # PyTorch checkpoint with highest validation F2
│       ├── metrics.json                 # Complete metric scores for the best checkpoint
│       ├── confusion_matrix.png         # Confusion matrix heatmap
│       ├── roc_curve.png                # Receiver Operating Characteristic plot
│       ├── pr_curve.png                 # Precision-Recall curve
│       └── training_history.png         # Loss and F2 learning curves
|
├── experiments/                         # Systematic benchmarking outputs
│   ├── leaderboard.csv                  # Ranked leaderboard of all experimental runs
│   └── <experiment_name>/               # Per-experiment checkpoints and tables
|
└── logs/                                # Execution logs for auditability
    ├── build_metadata.log
    ├── roi_extraction.log
    ├── train_baseline.log
    ├── analyze_thresholds.log
    └── run_experiments.log
```

---

## Clinical Modeling Best Practices & FAQ

### 1. Why Patient-Level (Grouped) Cross-Validation?
In clinical machine learning, standard random $k$-fold cross-validation is dangerous if a single patient contributes multiple photos. A random split could place images of the same patient into both training and validation sets. The neural network could memorize individual patient features (skin tone, lighting, eye shape) rather than generalizable anemia biomarkers, artificially inflating validation accuracy.
- This pipeline uses `StratifiedGroupKFold` grouped by `patient_id`.
- Before training each fold, the engine asserts that `set(train_patients) & set(val_patients) == empty`, strictly enforcing zero patient data leakage.

### 2. Why F2 Score Instead of Accuracy or F1?
In diagnostic screening, the cost of a **false negative** (telling an anemic patient they are healthy, delaying essential medical treatment) is far higher than a **false positive** (recommending a healthy patient for a confirmatory blood test).
- The $F_\beta$ score formula is:
  $$F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$
- With $\beta = 2$, recall (sensitivity) receives **four times the weight** of precision.
- Early stopping and model selection prioritize checkpoints that detect genuine anemic cases reliably.

### 3. What If ROI Extraction Shows a High Fallback Rate?
If more than 5–10% of images fall back to center-cropping:
1. Inspect the composite images in `outputs/debug/`.
2. Check if the images suffer from severe underexposure, blur, or off-center eye positioning.
3. Consult [docs/ROI_TUNING.md](docs/ROI_TUNING.md) to inspect and adjust the HSV / Lab thresholds in `configs/config.yaml`.

### 4. How Does Mixed Precision (AMP) Help?
On NVIDIA GPUs, setting `training.mixed_precision: true` uses PyTorch `torch.amp.autocast(device_type="cuda")` with `GradScaler`. This computes convolution operations in 16-bit floating point (`fp16`) while maintaining numerical stability in 32-bit float, cutting GPU VRAM consumption by ~50% and doubling batch throughput with no loss in model accuracy.

---

## License & Citation
Developed for the CP-AnemiC Data Pipeline & Benchmarking Project.
If you use this pipeline or its preprocessing methodologies in your research, please reference this repository.
