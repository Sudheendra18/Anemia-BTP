"""
Run multiple training configurations back to back and rank them by
primary_metric. Reuses the exact same crash-resilient training/resume
machinery as scripts/train_baseline.py, once per experiment.

    python scripts/run_experiments.py
    python scripts/run_experiments.py --fresh
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_runner import main  # noqa: E402

if __name__ == "__main__":
    main()
