"""Variant wrapper: run transformer training with sentiment channels zeroed."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
sys.path = [path_entry for path_entry in sys.path if Path(path_entry or ".").resolve() != THIS_DIR]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.models.transformer.run import main

DEFAULT_OUTPUT_DIR = "data/transformer_npy/no_sentiment"


def _has_arg(argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _with_default_arg(argv: list[str], flag: str, value: str) -> list[str]:
    if _has_arg(argv, flag):
        return list(argv)
    return [*argv, flag, value]


if __name__ == "__main__":
    base_argv = [*sys.argv[1:]]
    base_argv = _with_default_arg(base_argv, "--variant", "no_sentiment")
    base_argv = _with_default_arg(base_argv, "--gate-mode", "cross_attn")
    base_argv = _with_default_arg(base_argv, "--n-heads", "4")
    base_argv = _with_default_arg(base_argv, "--output-dir", DEFAULT_OUTPUT_DIR)
    print("[Launcher] Starting no_sentiment training...", flush=True)
    main(argv=base_argv, default_variant="no_sentiment")
