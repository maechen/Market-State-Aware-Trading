"""
CLI entry point for transformer training diagnostic visualizations.

Usage:
    python scripts/plot_transformer_diagnostics.py
    python scripts/plot_transformer_diagnostics.py --run-dir data/transformer_runs/gating
    python scripts/plot_transformer_diagnostics.py --run-dir data/transformer_runs/gating --out-subdir plots
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

from src.visualization.transformer_diagnostics import generate_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diagnostic visualizations from a MarketTransformer run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="data/transformer_runs/gating",
        help="Path to variant output directory (contains fold1/, fold2/, …, fold_metrics.csv).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="plots",
        help="Sub-directory name for saved figures (created inside --run-dir).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    if not run_dir.exists():
        print(f"[ERROR] run-dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Diagnostics] Generating figures from: {run_dir}", flush=True)
    print(f"[Diagnostics] Output sub-directory: {args.out_subdir}", flush=True)

    saved = generate_all(run_dir, out_subdir=args.out_subdir)

    print(f"\n[Diagnostics] Done — {len(saved)} figures saved to:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
