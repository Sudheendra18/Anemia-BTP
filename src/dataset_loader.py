"""
Stage 1 — Dataset loader / metadata builder for CP-AnemiC.

What this does
---------------
1. Scans the two class folders (Anemic / Non-anemic) on disk for image files.
2. Loads the clinical metadata Excel sheet.
3. Matches every image to its Excel row via IMAGE_ID (tolerant to case /
   separator differences in filenames).
4. Cross-checks the folder the image lives in against the Excel REMARK
   column. They *should* always agree — when they don't, or when an image
   or Excel row has no counterpart, the row is written to an issues CSV
   instead of being silently trusted or silently dropped.
5. Writes a single clean metadata CSV: patient_id, image_id, image_path,
   hb_level, severity, age_months, gender, label, hospital, region, ...

No duplicate / hash-based validation happens here by design for this pass —
this stage assumes every image file is a unique, valid sample.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utils import load_config, resolve_path, setup_logging, ensure_parent_dir


def _normalize_id(raw: str) -> str:
    """Strip separators and case so 'Image_001', 'image001', 'IMAGE-001'
    all normalize to the same key for matching filenames to Excel rows."""
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def scan_image_folder(folder: Path, extensions: list[str], folder_label: str) -> list[dict]:
    """Return one record per image file found directly inside `folder`."""
    records = []
    if not folder.exists():
        return records

    ext_set = {e.lower() for e in extensions}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in ext_set:
            records.append(
                {
                    "filename": path.name,
                    "norm_id": _normalize_id(path.stem),
                    "image_path": str(path),
                    "folder_label": folder_label,
                }
            )
    return records


def build_metadata(config: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds_cfg = config["dataset"]
    cols = ds_cfg["excel_columns"]

    raw_dir = resolve_path(ds_cfg["raw_images_dir"])
    anemic_dir = raw_dir / ds_cfg["class_folders"]["anemic"]
    non_anemic_dir = raw_dir / ds_cfg["class_folders"]["non_anemic"]

    logger.info(f"Scanning image folders under: {raw_dir}")
    logger.info(f"  Anemic folder:     {anemic_dir}")
    logger.info(f"  Non-anemic folder: {non_anemic_dir}")

    image_records = scan_image_folder(anemic_dir, ds_cfg["image_extensions"], "Anemic")
    image_records += scan_image_folder(non_anemic_dir, ds_cfg["image_extensions"], "Non-anemic")

    if not image_records:
        logger.warning(
            "No image files found in either folder. Check `dataset.raw_images_dir` "
            "and `dataset.class_folders` in the config — they currently point to:\n"
            f"  {anemic_dir}\n  {non_anemic_dir}"
        )
    else:
        n_anemic = sum(1 for r in image_records if r["folder_label"] == "Anemic")
        n_non = sum(1 for r in image_records if r["folder_label"] == "Non-anemic")
        logger.info(f"Found {len(image_records)} images on disk ({n_anemic} Anemic, {n_non} Non-anemic)")

    # ---- Load Excel metadata ----------------------------------------------
    excel_path = resolve_path(ds_cfg["metadata_excel_path"])
    logger.info(f"Loading Excel metadata: {excel_path}")
    excel_df = pd.read_excel(excel_path, sheet_name=ds_cfg["metadata_sheet_name"])
    logger.info(f"Excel sheet has {len(excel_df)} rows")

    excel_df["_norm_id"] = excel_df[cols["image_id"]].astype(str).map(_normalize_id)
    excel_lookup = excel_df.set_index("_norm_id").to_dict(orient="index")

    # ---- Match images -> excel rows ----------------------------------------
    matched_rows = []
    issues = []
    matched_norm_ids = set()

    for rec in image_records:
        norm_id = rec["norm_id"]
        excel_row = excel_lookup.get(norm_id)

        if excel_row is None:
            issues.append(
                {
                    "issue_type": "image_without_excel_row",
                    "image_id": rec["filename"],
                    "folder_label": rec["folder_label"],
                    "excel_remark": None,
                    "image_path": rec["image_path"],
                    "detail": "No Excel row matched this filename's normalized ID.",
                }
            )
            continue

        matched_norm_ids.add(norm_id)
        excel_remark = str(excel_row[cols["remark"]]).strip()
        folder_label = rec["folder_label"]

        label_agrees = excel_remark.lower() == folder_label.lower()
        if not label_agrees:
            issues.append(
                {
                    "issue_type": "folder_excel_label_mismatch",
                    "image_id": excel_row[cols["image_id"]],
                    "folder_label": folder_label,
                    "excel_remark": excel_remark,
                    "image_path": rec["image_path"],
                    "detail": f"Image is in '{folder_label}' folder but Excel REMARK says '{excel_remark}'.",
                }
            )
            if config["metadata_builder"]["drop_conflicts"]:
                continue

        matched_rows.append(
            {
                "patient_id": excel_row[cols["image_id"]],  # see docs/README: 1 image = 1 subject assumption
                "image_id": excel_row[cols["image_id"]],
                "image_path": rec["image_path"],
                "label_text": folder_label,  # folder is treated as ground truth per current instructions
                "label": ds_cfg["label_map"][folder_label],
                "label_source_agrees_with_excel": label_agrees,
                "hb_level": excel_row[cols["hb_level"]],
                "severity": excel_row[cols["severity"]],
                "age_months": excel_row[cols["age_months"]],
                "gender": excel_row[cols["gender"]],
                "hospital": excel_row[cols["hospital"]],
                "city_town": excel_row[cols["city_town"]],
                "district": excel_row[cols["district"]],
                "region": excel_row[cols["region"]],
                "country": excel_row[cols["country"]],
            }
        )

    # Excel rows that never got matched to any image on disk
    unmatched_excel = excel_df[~excel_df["_norm_id"].isin(matched_norm_ids)]
    for _, row in unmatched_excel.iterrows():
        issues.append(
            {
                "issue_type": "excel_row_without_image",
                "image_id": row[cols["image_id"]],
                "folder_label": None,
                "excel_remark": row[cols["remark"]],
                "image_path": None,
                "detail": "Excel row has no matching image file in either folder.",
            }
        )

    # Fixed column list so an empty result still writes a valid, header-only
    # CSV that downstream stages can open (pd.DataFrame([]) would instead
    # produce a CSV with zero columns, which pandas can't read back in).
    metadata_columns = [
        "patient_id", "image_id", "image_path", "label_text", "label",
        "label_source_agrees_with_excel", "hb_level", "severity", "age_months",
        "gender", "hospital", "city_town", "district", "region", "country",
    ]
    issues_columns = ["issue_type", "image_id", "folder_label", "excel_remark", "image_path", "detail"]

    metadata_df = pd.DataFrame(matched_rows, columns=metadata_columns)
    issues_df = pd.DataFrame(issues, columns=issues_columns)

    logger.info(f"Matched {len(metadata_df)} images to Excel metadata")
    if len(issues_df) > 0:
        by_type = issues_df["issue_type"].value_counts().to_dict()
        logger.warning(f"{len(issues_df)} issues found (see issues CSV): {by_type}")
    else:
        logger.info("No issues found — every image matched an Excel row and labels agree.")

    return metadata_df, issues_df


def main():
    parser = argparse.ArgumentParser(description="Build the CP-AnemiC metadata CSV from image folders + Excel sheet.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(log_file="outputs/logs/build_metadata.log")

    metadata_df, issues_df = build_metadata(config, logger)

    out_csv = ensure_parent_dir(config["metadata_builder"]["output_csv"])
    metadata_df.to_csv(out_csv, index=False)
    logger.info(f"Wrote metadata CSV -> {out_csv}  ({len(metadata_df)} rows)")

    issues_csv = ensure_parent_dir(config["metadata_builder"]["issues_csv"])
    issues_df.to_csv(issues_csv, index=False)
    logger.info(f"Wrote issues CSV   -> {issues_csv}  ({len(issues_df)} rows)")

    if len(metadata_df) > 0:
        logger.info("Class balance in final metadata CSV:")
        logger.info("\n" + metadata_df["label_text"].value_counts().to_string())


if __name__ == "__main__":
    main()
