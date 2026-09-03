# Data Contract — `cp_anemic_metadata.csv`

This is the interface between this track (data pipeline / benchmarking) and
anything downstream (training harness, the capture app's exports, etc.). If
this schema changes, everyone consuming it needs to know.

## `outputs/metadata/cp_anemic_metadata.csv`

One row per image that was successfully matched between the folders and the
Excel sheet.

| Column                          | Type   | Notes                                                                 |
| -------------------------------- | ------ | ---------------------------------------------------------------------- |
| `patient_id`                    | str    | Currently == `image_id`. CP-AnemiC's public release doesn't expose a subject identifier distinct from the image ID, so **this assumes one image = one subject**. If you later get access to true patient IDs (e.g. multiple images per subject), swap this column's source and re-run — everything downstream (k-fold splitting in particular) depends on this being correct to avoid patient leakage between folds. |
| `image_id`                      | str    | e.g. `Image_001`, matches the Excel `IMAGE_ID` column.                |
| `image_path`                    | str    | Absolute path to the source image on disk.                            |
| `label_text`                    | str    | `"Anemic"` / `"Non-anemic"` — taken from which folder the image is in. |
| `label`                         | int    | `0` = Non-anemic, `1` = Anemic (see `dataset.label_map` in config).    |
| `label_source_agrees_with_excel`| bool   | `False` means the folder location and the Excel `REMARK` column disagreed for this image — see `cp_anemic_issues.csv` for the detail. Worth resolving before training, since it means the label is ambiguous. |
| `hb_level`                      | float  | Haemoglobin, g/dL, from Excel `HB_LEVEL`.                              |
| `severity`                      | str    | `Mild` / `Moderate` / `Severe` / `Non-Anemic`, from Excel `Severity`.  |
| `age_months`                    | int    | From Excel `Age(Months)`.                                              |
| `gender`                        | str    | `Male` / `Female`.                                                     |
| `hospital`                      | str    | Source hospital — needed for hospital-stratified k-fold.               |
| `city_town`, `district`, `region`, `country` | str | Geographic metadata, kept for completeness / stratification options. |

## `outputs/metadata/cp_anemic_issues.csv`

Every row that needed a human look, instead of being silently trusted or
dropped. `issue_type` is one of:

- `folder_excel_label_mismatch` — image's folder and its Excel `REMARK` disagree.
- `image_without_excel_row` — an image file with no corresponding Excel row (typo'd filename, or genuinely not in the sheet).
- `excel_row_without_image` — an Excel row with no matching image file on disk (not yet captured, or filename mismatch).

## `outputs/metadata/cp_anemic_metadata_with_roi.csv`

Everything in `cp_anemic_metadata.csv`, plus:

| Column          | Type | Notes                                                                 |
| ---------------- | ---- | ---------------------------------------------------------------------- |
| `roi_image_path` | str  | Path to the cropped conjunctiva region, mirrors `image_path`'s label subfolder under `outputs/roi/`. |
| `roi_status`     | str  | `ok` (color segmentation found a valid region), `fallback` (center-crop used instead — worth a manual look), or `unreadable` (OpenCV couldn't open the file). |

## Known open question (flag for the supervisor)

Whether `patient_id` really is 1:1 with `image_id` in CP-AnemiC — i.e.
whether any subject contributed more than one image — determines whether
patient-level k-fold splitting is already satisfied by a plain image-level
split, or needs an explicit grouping key. Worth confirming before the k-fold
harness is built, since it changes which `sklearn` splitter is correct
(`StratifiedKFold` vs `StratifiedGroupKFold`).
