"""Save ROI crops and debug composites for a directory of images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2                                                    # noqa: E402
from roi_extraction import extract_roi, _save_debug_composite  # noqa: E402
from utils import load_config, resolve_path, setup_logging     # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Test ROI extraction on any folder of images.")
    parser.add_argument("--input", required=True, help="Folder containing the images to test (any path — doesn't need to be inside this project).")
    parser.add_argument("--output", default="outputs/roi_test", help="Where to write debug composites and crops. Default: outputs/roi_test")
    parser.add_argument("--config", default="configs/config.yaml", help="Config to read ROI settings from (same thresholds as the real pipeline).")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(log_file="outputs/logs/roi_test.log")

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input folder not found: {input_dir}")
        return

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {e.lower() for e in config["dataset"]["image_extensions"]}
    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)

    if not image_paths:
        logger.error(f"No images found in {input_dir} (looked for extensions: {sorted(extensions)})")
        return

    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    roi_cfg = config["roi_extraction"]
    n_ok, n_fallback, n_unreadable = 0, 0, 0

    for path in image_paths:
        result = extract_roi(str(path), roi_cfg)

        if result["status"] == "unreadable":
            logger.warning(f"{path.name}: could not read image (corrupt file or unsupported format)")
            n_unreadable += 1
            continue

        composite_path = output_dir / f"debug_{path.stem}.jpg"
        _save_debug_composite(result, composite_path)

        crop_path = output_dir / f"crop_{path.name}"
        cv2.imwrite(str(crop_path), result["roi"])

        status = result["status"]
        logger.info(f"{path.name}: {status:8s}  bbox={result['bbox']}")
        if status == "ok":
            n_ok += 1
        else:
            n_fallback += 1

    logger.info("=" * 60)
    logger.info(f"Done: {n_ok} ok, {n_fallback} fallback, {n_unreadable} unreadable  (of {len(image_paths)} total)")
    logger.info(f"Results written to -> {output_dir}")
    logger.info("Debug panels show the detected box, tissue mask, rectangular crop, and masked crop.")


if __name__ == "__main__":
    main()
