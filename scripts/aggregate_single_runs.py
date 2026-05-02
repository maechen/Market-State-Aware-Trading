"""
Aggregate several single TDQN runs.

Example:

python scripts/aggregate_single_runs.py ^
 --run-dirs ^
 data/baselines/drl_latent_regime_no_dir_42 ^
 data/baselines/drl_latent_regime_no_dir_52 ^
 data/baselines/drl_latent_regime_no_dir_63 ^
 data/baselines/drl_latent_regime_no_dir_69 ^
 data/baselines/drl_latent_regime_no_dir_81 ^
 --output-dir data/baselines/no_dir_aggregated_tdqn ^
 --variants gating,no_gating,no_sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"

for p in (str(THIS_DIR), str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from drl_latent_regime_training import (  # noqa: E402
    compute_performance_metrics,
    save_per_fold_comparison_plot,
    save_portfolio_growth_plot,
    save_strategy_comparison_plot,
)


VARIANTS = ("gating", "no_gating", "no_sentiment")

LEGACY_ARTIFACT_PREFIX = "rl_only"
ARTIFACT_PREFIX = "drl_latent_regime"

LEGACY_MODEL_STRATEGY = "rl_tdqn"
LEGACY_STITCHED_MODEL_STRATEGY = "rl_tdqn_stitched"
MODEL_STRATEGY = "transformer_tdqn"
STITCHED_MODEL_STRATEGY = "transformer_tdqn_stitched"

STRATEGY_LABELS = {
    MODEL_STRATEGY: "Transformer-TDQN",
    STITCHED_MODEL_STRATEGY: "Transformer-TDQN (stitched)",
    LEGACY_MODEL_STRATEGY: "Transformer-TDQN",
    LEGACY_STITCHED_MODEL_STRATEGY: "Transformer-TDQN (stitched)",
    "buy_hold": "Buy & Hold",
    "momentum_126": "Momentum (126d)",
}

KEY_COLUMNS = {
    f"{ARTIFACT_PREFIX}_fold_metrics.csv": ["fold", "split"],
    f"{ARTIFACT_PREFIX}_per_fold_test_metrics.csv": ["fold", "strategy"],
    f"{ARTIFACT_PREFIX}_stitched_comparison_metrics.csv": ["strategy"],
}


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _legacy_name(filename: str) -> str:
    if filename.startswith(f"{ARTIFACT_PREFIX}_"):
        return f"{LEGACY_ARTIFACT_PREFIX}_{filename[len(ARTIFACT_PREFIX) + 1:]}"
    return filename


def _normalize_strategy_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "strategy" in df.columns:
        df["strategy"] = df["strategy"].replace(
            {
                LEGACY_MODEL_STRATEGY: MODEL_STRATEGY,
                LEGACY_STITCHED_MODEL_STRATEGY: STITCHED_MODEL_STRATEGY,
            }
        )
    return df


def _with_display_strategy_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "strategy" in df.columns:
        df["strategy"] = df["strategy"].replace(STRATEGY_LABELS)
    return df


def _infer_seed_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    for token in reversed(name.split("_")):
        if token.isdigit():
            return token
    return name


def _read_run_csv(run_dir: Path, variant: str, filename: str) -> pd.DataFrame | None:
    path = run_dir / variant / filename

    if not path.exists():
        legacy_path = run_dir / variant / _legacy_name(filename)
        if legacy_path.exists():
            path = legacy_path

    if not path.exists():
        print(f"[WARN] Missing artifact: {path}")
        return None

    df = pd.read_csv(path)
    df = _normalize_strategy_names(df)
    df["run"] = run_dir.name
    df["seed"] = _infer_seed_from_run_dir(run_dir)
    df["variant"] = variant
    return df


def _collect(
    run_dirs: list[Path],
    variants: list[str],
    filename: str,
) -> pd.DataFrame | None:
    frames = []

    for run_dir in run_dirs:
        for variant in variants:
            df = _read_run_csv(run_dir, variant, filename)
            if df is not None:
                frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def _write_per_variant(summary_dir: Path, filename: str, df: pd.DataFrame) -> None:
    if "variant" not in df.columns:
        return

    for variant, variant_df in df.groupby("variant", dropna=False):
        variant_dir = summary_dir / str(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_df.drop(columns=["variant"]).to_csv(variant_dir / filename, index=False)


def _average_metric_file(
    run_dirs: list[Path],
    summary_dir: Path,
    variants: list[str],
    filename: str,
) -> pd.DataFrame | None:
    df = _collect(run_dirs, variants, filename)
    if df is None:
        return None

    df.to_csv(summary_dir / f"all_runs_{filename}", index=False)

    key_cols = ["variant", *KEY_COLUMNS[filename]]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"seed"}
    ]

    averaged = (
        df.groupby(key_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    averaged.columns = [
        "_".join(part for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in averaged.columns
    ]

    outname = f"average_{filename}"
    averaged.to_csv(summary_dir / outname, index=False)
    _write_per_variant(summary_dir, outname, averaged)
    return averaged


def _average_curve_file(
    run_dirs: list[Path],
    summary_dir: Path,
    variants: list[str],
    filename: str,
    group_columns: list[str],
) -> pd.DataFrame | None:
    df = _collect(run_dirs, variants, filename)
    if df is None:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "account_value"])

    if df.empty:
        return None

    df.to_csv(summary_dir / f"all_runs_{filename}", index=False)

    averaged = (
        df.groupby(["variant", *group_columns], dropna=False)["account_value"]
        .agg(
            account_value="mean",
            account_value_std="std",
            seed_count="count",
        )
        .reset_index()
        .sort_values(["variant", *group_columns])
    )

    outname = f"average_{filename}"
    averaged.to_csv(summary_dir / outname, index=False)
    _write_per_variant(summary_dir, outname, averaged)
    return averaged


def _decision_from_mean(value: float, threshold: float = 1.0 / 3.0) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _aggregate_actions(
    run_dirs: list[Path],
    summary_dir: Path,
    variants: list[str],
    filename: str,
    group_columns: list[str],
) -> pd.DataFrame | None:
    df = _collect(run_dirs, variants, filename)
    if df is None:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return None

    df.to_csv(summary_dir / f"all_runs_{filename}", index=False)

    rows = []

    for keys, group in df.groupby(["variant", *group_columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(["variant", *group_columns], keys))

        for col in (
            "signal",
            "actual_action",
            "position",
            "position_after_trade",
            "weight",
            "trade_notional",
            "pnl",
            "drawdown",
        ):
            if col in group.columns:
                row[f"mean_{col}"] = float(
                    pd.to_numeric(group[col], errors="coerce").mean()
                )

        if "signal" in group.columns:
            mean_signal = float(row["mean_signal"])
            row["ensemble_signal"] = _decision_from_mean(mean_signal)

            mode = (
                pd.to_numeric(group["signal"], errors="coerce")
                .dropna()
                .astype(int)
                .mode()
            )
            row["signal_vote"] = int(mode.iloc[0]) if not mode.empty else 0

        if "position" in group.columns:
            row["ensemble_position"] = _decision_from_mean(
                float(row["mean_position"])
            )

        rows.append(row)

    averaged = pd.DataFrame(rows).sort_values(["variant", *group_columns])

    outname = f"ensemble_{filename}"
    averaged.to_csv(summary_dir / outname, index=False)
    _write_per_variant(summary_dir, outname, averaged)
    return averaged


def _save_average_plots(summary_dir: Path) -> None:
    curves_path = summary_dir / f"average_{ARTIFACT_PREFIX}_portfolio_curves.csv"
    if curves_path.exists():
        curves_df = pd.read_csv(curves_path, parse_dates=["date"])
        for variant, variant_df in curves_df.groupby("variant"):
            save_portfolio_growth_plot(
                portfolio_curves_df=variant_df[
                    ["date", "account_value", "fold", "split"]
                ],
                output_path=summary_dir / f"{variant}_average_portfolio_growth.png",
            )

    comparison_path = summary_dir / f"average_{ARTIFACT_PREFIX}_stitched_comparison.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
        for variant, variant_df in comparison_df.groupby("variant"):
            save_strategy_comparison_plot(
                comparison_df=_with_display_strategy_names(
                    variant_df[["date", "strategy", "account_value"]]
                ),
                output_path=summary_dir / f"{variant}_average_stitched_comparison.png",
            )

    per_fold_path = summary_dir / f"average_{ARTIFACT_PREFIX}_per_fold_test_comparison.csv"
    if per_fold_path.exists():
        per_fold_df = pd.read_csv(per_fold_path, parse_dates=["date"])
        for variant, variant_df in per_fold_df.groupby("variant"):
            save_per_fold_comparison_plot(
                per_fold_comparison_df=_with_display_strategy_names(
                    variant_df[["date", "strategy", "account_value", "fold"]]
                ),
                output_path=summary_dir / f"{variant}_average_per_fold_test_comparison.png",
            )


def _write_average_decision_summary(
    summary_dir: Path,
    run_dirs: list[Path],
) -> pd.DataFrame | None:
    comparison_path = summary_dir / f"average_{ARTIFACT_PREFIX}_stitched_comparison.csv"
    actions_path = summary_dir / f"ensemble_{ARTIFACT_PREFIX}_stitched_test_actions.csv"

    if not comparison_path.exists():
        return None

    comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
    rows = []

    actions_all = None
    if actions_path.exists():
        actions_all = pd.read_csv(actions_path, parse_dates=["date"])

    for (variant, strategy), strategy_df in comparison_df.groupby(["variant", "strategy"]):
        actions_df = None

        if actions_all is not None and strategy == STITCHED_MODEL_STRATEGY:
            actions_df = actions_all[actions_all["variant"] == variant].rename(
                columns={
                    "ensemble_signal": "signal",
                    "mean_actual_action": "actual_action",
                    "mean_trade_notional": "trade_notional",
                }
            )

            if "ensemble_position" in actions_df.columns:
                actions_df["position_after_trade"] = actions_df["ensemble_position"]
                actions_df["position"] = actions_df["ensemble_position"]

        metrics = compute_performance_metrics(
            strategy_df[["date", "account_value"]]
            .sort_values("date")
            .reset_index(drop=True),
            actions_df=actions_df,
        )

        rows.append(
            {
                "variant": variant,
                "strategy": strategy,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)

    outname = "average_stitched_comparison_metrics_recomputed.csv"
    metrics_df.to_csv(summary_dir / outname, index=False)
    _write_per_variant(summary_dir, outname, metrics_df)

    rl_df = metrics_df[metrics_df["strategy"] == STITCHED_MODEL_STRATEGY].copy()

    if not rl_df.empty:
        rl_df = rl_df.sort_values(
            ["sharpe_ratio", "total_return", "max_drawdown"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        rl_df.to_csv(summary_dir / "average_variant_ranking.csv", index=False)

        best = rl_df.iloc[0].to_dict()

        decision = {
            "selected_variant": best["variant"],
            "selection_rule": (
                "highest average stitched RL Sharpe ratio, "
                "then total return, then max drawdown"
            ),
            "runs_used": [str(p) for p in run_dirs],
            "selected_metrics": best,
        }

        with open(
            summary_dir / "average_decision_summary.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(decision, fh, indent=2)

    return metrics_df


def aggregate_single_runs(
    run_dirs: list[Path],
    output_dir: Path,
    variants: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in KEY_COLUMNS:
        _average_metric_file(
            run_dirs=run_dirs,
            summary_dir=output_dir,
            variants=variants,
            filename=filename,
        )

    _average_curve_file(
        run_dirs,
        output_dir,
        variants,
        f"{ARTIFACT_PREFIX}_portfolio_curves.csv",
        ["date", "fold", "split"],
    )

    _average_curve_file(
        run_dirs,
        output_dir,
        variants,
        f"{ARTIFACT_PREFIX}_stitched_test_equity.csv",
        ["date", "fold"],
    )

    _average_curve_file(
        run_dirs,
        output_dir,
        variants,
        f"{ARTIFACT_PREFIX}_stitched_comparison.csv",
        ["date", "strategy"],
    )

    _average_curve_file(
        run_dirs,
        output_dir,
        variants,
        f"{ARTIFACT_PREFIX}_per_fold_test_comparison.csv",
        ["date", "fold", "strategy"],
    )

    _aggregate_actions(
        run_dirs,
        output_dir,
        variants,
        f"{ARTIFACT_PREFIX}_stitched_test_actions.csv",
        ["date", "fold"],
    )

    _save_average_plots(output_dir)
    _write_average_decision_summary(output_dir, run_dirs)

    run_config = {
        "run_dirs": [str(p) for p in run_dirs],
        "variants": variants,
    }

    with open(output_dir / "aggregation_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    print(f"Aggregation complete: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate several single TDQN runs into averaged summaries."
    )

    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Single-run output directories.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for aggregated files.",
    )

    parser.add_argument(
        "--variants",
        type=str,
        default="gating,no_gating,no_sentiment",
        help="Comma-separated variants.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dirs = [_resolve_path(p) for p in args.run_dirs]
    output_dir = _resolve_path(args.output_dir)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {list(VARIANTS)}")

    aggregate_single_runs(
        run_dirs=run_dirs,
        output_dir=output_dir,
        variants=variants,
    )


if __name__ == "__main__":
    main()