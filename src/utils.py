"""
Shared utilities: config loading, logging, small path helpers.

Nothing in this file is dataset-specific — it's plumbing used by both
dataset_loader.py and roi_extraction.py so the two stages behave consistently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of this file's parent (src/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML config and resolve all relative paths against the
    project root, so scripts behave the same regardless of the working
    directory they're invoked from."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Pass the path explicitly with --config, e.g.\n"
            f"  python scripts/build_metadata.py --config configs/config.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(path_str: str) -> Path:
    """Resolve a path from the config against the project root if relative."""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Console + optional file logging. Every stage script calls this once."""
    logger = logging.getLogger("cp_anemic_pipeline")
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate handlers if called twice in one process

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = resolve_path(str(log_file))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def ensure_parent_dir(path: str | Path) -> Path:
    """Resolve a path and make sure its parent directory exists."""
    resolved = resolve_path(str(path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
