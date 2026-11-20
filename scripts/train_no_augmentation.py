"""Run the data augmentation ablation study."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_no_augmentation import main  # noqa: E402

if __name__ == "__main__":
    main()
