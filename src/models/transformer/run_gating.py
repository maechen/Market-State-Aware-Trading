"""Variant wrapper: run transformer training with default gating behavior."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path = [path_entry for path_entry in sys.path if Path(path_entry or ".").resolve() != THIS_DIR]

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.models.transformer.run import main


if __name__ == "__main__":
    main(argv=[*sys.argv[1:], "--variant", "gating"], default_variant="gating")
