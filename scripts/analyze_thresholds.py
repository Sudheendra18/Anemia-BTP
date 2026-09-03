"""
Find a better decision threshold using already-trained fold checkpoints.
Does NOT retrain anything — just runs inference once per fold.

    python scripts/analyze_thresholds.py
    python scripts/analyze_thresholds.py --sensitivity-floor 0.85
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from threshold_analysis import main  # noqa: E402

if __name__ == "__main__":
    main()
