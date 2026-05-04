"""
Rewrite legacy TDQN labels in rl_only_multi_seed outputs.

This updates CSV/JSON text values and regenerates the known plots with RL-only
labels. Numeric results are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in [THIS_DIR, *THIS_DIR.parents]
    if (parent / "src").is_dir() and (parent / "data").is_dir()
)

ARTIFACT_PREFIX = "rl_only"
REPLACEMENTS = {
    "rl_tdqn_stitched": "rl_only_stitched",
    "rl_tdqn": "rl_only",
    "RL-TDQN": "RL-Only",
    "RL-only TDQN": "RL-only",
    "RL-Only TDQN": "RL-Only",
    "TDQN": "RL-only",
}


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _replace_text(value):
    if isinstance(value, str):
        updated = value
        for old, new in REPLACEMENTS.items():
            updated = updated.replace(old, new)
        return updated
    if isinstance(value, list):
        return [_replace_text(item) for item in value]
    if isinstance(value, dict):
        return {_replace_text(key): _replace_text(item) for key, item in value.items()}
    return value


def _rewrite_csv(path: Path) -> bool:
    df = pd.read_csv(path)
    changed = False
    for col in df.select_dtypes(include=["object"]).columns:
        updated = df[col].map(_replace_text)
        if not updated.equals(df[col]):
            df[col] = updated
            changed = True
    if changed:
        df.to_csv(path, index=False)
    return changed


def _rewrite_json(path: Path) -> bool:
    with open(path, encoding="utf-8") as fh:
        original = json.load(fh)
    updated = _replace_text(original)
    if updated != original:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(updated, fh, indent=2)
        return True
    return False


def save_portfolio_growth_plot(portfolio_curves_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = portfolio_curves_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "account_value"]).sort_values("date")
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for (fold_name, split_name), curve_df in plot_df.groupby(["fold", "split"]):
        linestyle = "-" if split_name == "test" else "--"
        ax.plot(
            curve_df["date"],
            curve_df["account_value"],
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{fold_name} {split_name.upper()}",
        )
    ax.set_title("SPY Portfolio Growth (RL-Only)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_strategy_comparison_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = comparison_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "strategy", "account_value"]).sort_values("date")
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy_name, strategy_df in plot_df.groupby("strategy"):
        ax.plot(strategy_df["date"], strategy_df["account_value"], linewidth=2.0, label=strategy_name)
    ax.set_title("Stitched Out-of-Sample Equity Curve Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_fold_comparison_plot(per_fold_comparison_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = per_fold_comparison_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "account_value", "strategy", "fold"]).sort_values(
        ["fold", "strategy", "date"]
    )
    if plot_df.empty:
        return

    folds = sorted(plot_df["fold"].unique())
    ncols = min(len(folds), 4)
    nrows = math.ceil(len(folds) / ncols)
    strategy_styles = {
        "rl_only": {"color": "tab:blue", "label": "RL-Only"},
        "buy_hold": {"color": "tab:green", "label": "Buy & Hold"},
        "momentum_126": {"color": "tab:orange", "label": "Momentum (126d)"},
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, fold_name in enumerate(folds):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        for strategy_name, strategy_df in plot_df[plot_df["fold"] == fold_name].groupby("strategy"):
            style = strategy_styles.get(strategy_name, {"color": None, "label": strategy_name})
            ax.plot(
                strategy_df["date"],
                strategy_df["account_value"],
                linewidth=1.5,
                color=style["color"],
                label=style["label"],
            )
        ax.set_title(fold_name, fontsize=10)
        ax.set_ylabel("Portfolio Value ($)", fontsize=8)
        ax.legend(loc="best", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linestyle=":")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

    for idx in range(len(folds), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Per-Fold Test Period: RL-Only vs Buy & Hold vs Momentum (each fold starts at $10k)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _regenerate_plots(output_dir: Path) -> int:
    plot_count = 0
    for curves_path, plot_path in (
        (
            output_dir / f"{ARTIFACT_PREFIX}_portfolio_curves.csv",
            output_dir / f"{ARTIFACT_PREFIX}_portfolio_growth.png",
        ),
        (
            output_dir / f"average_{ARTIFACT_PREFIX}_portfolio_curves.csv",
            output_dir / "average_portfolio_growth.png",
        ),
    ):
        if not curves_path.exists():
            continue
        curves_df = pd.read_csv(curves_path, parse_dates=["date"])
        save_portfolio_growth_plot(
            curves_df[["date", "account_value", "fold", "split"]],
            plot_path,
        )
        plot_count += 1

    for comparison_path, plot_path in (
        (
            output_dir / f"{ARTIFACT_PREFIX}_stitched_comparison.csv",
            output_dir / f"{ARTIFACT_PREFIX}_stitched_comparison.png",
        ),
        (
            output_dir / f"average_{ARTIFACT_PREFIX}_stitched_comparison.csv",
            output_dir / "average_stitched_comparison.png",
        ),
    ):
        if not comparison_path.exists():
            continue
        comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
        save_strategy_comparison_plot(
            comparison_df[["date", "strategy", "account_value"]],
            plot_path,
        )
        plot_count += 1

    for per_fold_path, plot_path in (
        (
            output_dir / f"{ARTIFACT_PREFIX}_per_fold_test_comparison.csv",
            output_dir / f"{ARTIFACT_PREFIX}_per_fold_test_comparison.png",
        ),
        (
            output_dir / f"average_{ARTIFACT_PREFIX}_per_fold_test_comparison.csv",
            output_dir / "average_per_fold_test_comparison.png",
        ),
    ):
        if not per_fold_path.exists():
            continue
        per_fold_df = pd.read_csv(per_fold_path, parse_dates=["date"])
        save_per_fold_comparison_plot(
            per_fold_df[["date", "strategy", "account_value", "fold"]],
            plot_path,
        )
        plot_count += 1
    return plot_count


def rewrite_outputs(output_dir: Path, regenerate_plots: bool) -> dict[str, int]:
    counts = {"csv": 0, "json": 0, "plots": 0, "errors": 0}
    for path in output_dir.rglob("*.csv"):
        try:
            if _rewrite_csv(path):
                counts["csv"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"[WARN] Could not rewrite CSV {path}: {exc}", flush=True)

    for path in output_dir.rglob("*.json"):
        try:
            if _rewrite_json(path):
                counts["json"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"[WARN] Could not rewrite JSON {path}: {exc}", flush=True)

    if regenerate_plots:
        for path in [output_dir, *[p for p in output_dir.rglob("*") if p.is_dir()]]:
            try:
                counts["plots"] += _regenerate_plots(path)
            except Exception as exc:
                counts["errors"] += 1
                print(f"[WARN] Could not regenerate plots in {path}: {exc}", flush=True)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite legacy TDQN labels in rl_only_multi_seed output artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/baselines/rl_only_multi_seed",
        help="Root output directory created by rl_only_multi_seed.py.",
    )
    parser.add_argument(
        "--no-regenerate-plots",
        action="store_true",
        help="Only rewrite CSV/JSON text and leave PNG files unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    counts = rewrite_outputs(
        output_dir=output_dir,
        regenerate_plots=not args.no_regenerate_plots,
    )
    print(
        "Rewrote "
        f"{counts['csv']} CSV file(s), "
        f"{counts['json']} JSON file(s), "
        f"regenerated {counts['plots']} plot(s), "
        f"{counts['errors']} error(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
