"""
CLI for transformer MAE diagnostics from per-fold test prediction exports.

Example:
    python scripts/plot_transformer_mae.py \
      --runs-root data/transformer_npy \
      --variants gating,no_gating,no_sentiment \
      --out-dir data/transformer_npy/mae_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.visualization.transformer_mae import generate_mae_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot transformer MAE by fold/variant from test_predictions.csv exports."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="data/transformer_npy",
        help="Root containing variant folders (e.g. gating/, no_gating/, no_sentiment/).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="gating,no_gating,no_sentiment",
        help="Comma-separated variant folder names under --runs-root.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/transformer_npy/mae_plots",
        help="Directory to save MAE CSV summaries and plot.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    if not variants:
        raise ValueError("--variants must contain at least one variant name.")

    print(f"[MAE] runs_root={runs_root}", flush=True)
    print(f"[MAE] variants={variants}", flush=True)
    print(f"[MAE] out_dir={out_dir}", flush=True)

    outputs = generate_mae_artifacts(
        runs_root=runs_root,
        variants=variants,
        out_dir=out_dir,
    )

    print("\n[MAE] Saved artifacts:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
