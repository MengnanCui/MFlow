#!/usr/bin/env python3
"""Kept for old scripts: the tool now lives in the package — use `mflow tool izo-formation ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mflow.extras.izo_formation import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
