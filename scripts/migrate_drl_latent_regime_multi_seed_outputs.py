"""
Migrate existing latent-regime multi-seed outputs to transformer-aware names.

The older DRL latent-regime runs wrote artifacts with ``rl_only`` filenames and
``rl_tdqn`` strategy labels. This script updates existing outputs from
scripts/drl_latent_regime_multi_seed.py without rerunning training:

* renames artifact files from rl_only_* to drl_latent_regime_*
* rewrites CSV/JSON strategy values to transformer_tdqn names
* regenerates affected PNG plots from the updated CSVs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "baselines" / "drl_latent_regime_multi_seed"

OLD_PREFIX = "rl_only"
NEW_PREFIX = "drl_latent_regime"

STRATEGY_REPLACEMENTS = {
    "rl_tdqn": "transformer_tdqn",
    "rl_tdqn_stitched": "transformer_tdqn_stitched",
}

STRATEGY_DISPLAY_LABELS = {
    "transformer_tdqn": "Transformer-TDQN",
    "transformer_tdqn_stitched": "Transformer-TDQN (stitched)",
    "buy_hold": "Buy & Hold",
    "momentum_126": "Momentum (126d)",
}


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _new_name(filename: str) -> str:
    return filename.replace(OLD_PREFIX, NEW_PREFIX)


def _iter_renames(output_dir: Path) -> list[tuple[Path, Path]]:
    renames: list[tuple[Path, Path]] = []
    for source_path in output_dir.rglob("*"):
        if not source_path.is_file() or OLD_PREFIX not in source_path.name:
            continue
        renames.append((source_path, source_path.with_name(_new_name(source_path.name))))
    return sorted(renames, key=lambda item: len(item[0].parts), reverse=True)


def rename_artifacts(output_dir: Path, dry_run: bool) -> int:
    renames = _iter_renames(output_dir)
    count = 0
    for source_path, target_path in renames:
        if source_path == target_path:
            continue
        if target_path.exists():
            print(f"[SKIP] Target exists: {target_path.relative_to(PROJECT_ROOT)}")
            continue

        print(f"[RENAME] {source_path.relative_to(PROJECT_ROOT)} -> {target_path.relative_to(PROJECT_ROOT)}")
        if not dry_run:
            source_path.rename(target_path)
        count += 1
    return count


def update_csv_strategy_values(output_dir: Path, dry_run: bool) -> int:
    count = 0
    for csv_path in sorted(output_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Could not read CSV {csv_path.relative_to(PROJECT_ROOT)}: {exc}")
            continue

        changed = False
        for col in df.select_dtypes(include=["object"]).columns:
            updated = df[col].replace(STRATEGY_REPLACEMENTS)
            if not updated.equals(df[col]):
                df[col] = updated
                changed = True

        if changed:
            print(f"[CSV] Updated strategy labels in {csv_path.relative_to(PROJECT_ROOT)}")
            if not dry_run:
                df.to_csv(csv_path, index=False)
            count += 1
    return count


def _replace_json_strings(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        updated = value
        for old, new in {**STRATEGY_REPLACEMENTS, OLD_PREFIX: NEW_PREFIX}.items():
            updated = updated.replace(old, new)
        return updated, updated != value

    if isinstance(value, list):
        changed = False
        updated_items = []
        for item in value:
            updated_item, item_changed = _replace_json_strings(item)
            updated_items.append(updated_item)
            changed = changed or item_changed
        return updated_items, changed

    if isinstance(value, dict):
        changed = False
        updated_dict = {}
        for key, item in value.items():
            updated_key, key_changed = _replace_json_strings(key)
            updated_item, item_changed = _replace_json_strings(item)
            updated_dict[updated_key] = updated_item
            changed = changed or key_changed or item_changed
        return updated_dict, changed

    return value, False


def update_json_labels(output_dir: Path, dry_run: bool) -> int:
    count = 0
    for json_path in sorted(output_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Could not read JSON {json_path.relative_to(PROJECT_ROOT)}: {exc}")
            continue

        updated, changed = _replace_json_strings(data)
        if changed:
            print(f"[JSON] Updated labels in {json_path.relative_to(PROJECT_ROOT)}")
            if not dry_run:
                json_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
            count += 1
    return count


def _display_strategy_name(strategy_name: str) -> str:
    return STRATEGY_DISPLAY_LABELS.get(strategy_name, strategy_name)


def _save_portfolio_growth_plot(csv_path: Path, output_path: Path, dry_run: bool) -> bool:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["date", "account_value"]).sort_values(["fold", "split", "date"])
    if df.empty:
        return False

    print(f"[PLOT] {output_path.relative_to(PROJECT_ROOT)}")
    if dry_run:
        return True

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for (fold_name, split_name), group in df.groupby(["fold", "split"], dropna=False):
        ax.plot(group["date"], group["account_value"], linewidth=1.3, label=f"{fold_name} {split_name}")
    ax.set_title("SPY Portfolio Growth (Transformer-Latent TDQN)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _save_strategy_comparison_plot(csv_path: Path, output_path: Path, dry_run: bool) -> bool:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["date", "strategy", "account_value"]).sort_values("date")
    if df.empty:
        return False

    print(f"[PLOT] {output_path.relative_to(PROJECT_ROOT)}")
    if dry_run:
        return True

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy_name, group in df.groupby("strategy", dropna=False):
        ax.plot(
            group["date"],
            group["account_value"],
            linewidth=2.0,
            label=_display_strategy_name(str(strategy_name)),
        )
    ax.set_title("Stitched Out-of-Sample Equity Curve Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _save_per_fold_comparison_plot(csv_path: Path, output_path: Path, dry_run: bool) -> bool:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["date", "strategy", "account_value", "fold"]).sort_values(
        ["fold", "strategy", "date"]
    )
    if df.empty:
        return False

    print(f"[PLOT] {output_path.relative_to(PROJECT_ROOT)}")
    if dry_run:
        return True

    import math
    import matplotlib.pyplot as plt

    folds = sorted(df["fold"].unique())
    ncols = min(len(folds), 4)
    nrows = math.ceil(len(folds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    colors = {
        "transformer_tdqn": "tab:blue",
        "buy_hold": "tab:green",
        "momentum_126": "tab:orange",
    }
    for idx, fold_name in enumerate(folds):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        fold_df = df[df["fold"] == fold_name]
        for strategy_name, group in fold_df.groupby("strategy", dropna=False):
            strategy_name = str(strategy_name)
            ax.plot(
                group["date"],
                group["account_value"],
                linewidth=1.5,
                color=colors.get(strategy_name),
                label=_display_strategy_name(strategy_name),
            )
        ax.set_title(str(fold_name), fontsize=10)
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
        "Per-Fold Test Period: Transformer-TDQN vs Buy & Hold vs Momentum (each fold starts at $10k)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def regenerate_plots(output_dir: Path, dry_run: bool) -> int:
    plot_count = 0
    for csv_path in sorted(output_dir.rglob(f"{NEW_PREFIX}_portfolio_curves.csv")):
        plot_count += int(
            _save_portfolio_growth_plot(
                csv_path,
                csv_path.with_name(f"{NEW_PREFIX}_portfolio_growth.png"),
                dry_run,
            )
        )

    for csv_path in sorted(output_dir.rglob(f"{NEW_PREFIX}_stitched_comparison.csv")):
        plot_count += int(
            _save_strategy_comparison_plot(
                csv_path,
                csv_path.with_name(f"{NEW_PREFIX}_stitched_comparison.png"),
                dry_run,
            )
        )

    for csv_path in sorted(output_dir.rglob(f"{NEW_PREFIX}_per_fold_test_comparison.csv")):
        plot_count += int(
            _save_per_fold_comparison_plot(
                csv_path,
                csv_path.with_name(f"{NEW_PREFIX}_per_fold_test_comparison.png"),
                dry_run,
            )
        )

    averages_dir = output_dir / "averages"
    for variant_dir in [path for path in averages_dir.iterdir() if path.is_dir()] if averages_dir.exists() else []:
        portfolio_csv = variant_dir / f"average_{NEW_PREFIX}_portfolio_curves.csv"
        if portfolio_csv.exists():
            plot_count += int(
                _save_portfolio_growth_plot(
                    portfolio_csv,
                    averages_dir / f"{variant_dir.name}_average_portfolio_growth.png",
                    dry_run,
                )
            )

        stitched_csv = variant_dir / f"average_{NEW_PREFIX}_stitched_comparison.csv"
        if stitched_csv.exists():
            plot_count += int(
                _save_strategy_comparison_plot(
                    stitched_csv,
                    averages_dir / f"{variant_dir.name}_average_stitched_comparison.png",
                    dry_run,
                )
            )

        per_fold_csv = variant_dir / f"average_{NEW_PREFIX}_per_fold_test_comparison.csv"
        if per_fold_csv.exists():
            plot_count += int(
                _save_per_fold_comparison_plot(
                    per_fold_csv,
                    averages_dir / f"{variant_dir.name}_average_per_fold_test_comparison.png",
                    dry_run,
                )
            )

    return plot_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate existing drl_latent_regime_multi_seed artifacts to transformer-aware names."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        help="Root output directory created by drl_latent_regime_multi_seed.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying files.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Rename/rewrite data files but do not regenerate PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    renamed = rename_artifacts(output_dir, dry_run=args.dry_run)
    csv_updates = update_csv_strategy_values(output_dir, dry_run=args.dry_run)
    json_updates = update_json_labels(output_dir, dry_run=args.dry_run)
    plot_updates = 0 if args.skip_plots else regenerate_plots(output_dir, dry_run=args.dry_run)

    action = "Would update" if args.dry_run else "Updated"
    print(
        f"{action}: {renamed} renamed file(s), {csv_updates} CSV file(s), "
        f"{json_updates} JSON file(s), {plot_updates} plot file(s)."
    )


if __name__ == "__main__":
    main()
