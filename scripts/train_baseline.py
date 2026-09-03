"""
Run Stage 3 (baseline model training + k-fold evaluation) from one config file.

    python scripts/train_baseline.py
    python scripts/train_baseline.py --config configs/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
