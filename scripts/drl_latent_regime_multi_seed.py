"""
Run the latent-regime TDQN experiment across variants and seeds.

Execution order is repetition-major:
  rep 1: gating, no_gating, no_sentiment
  rep 2: gating, no_gating, no_sentiment
  ...

Each repetition uses a different seed. Per-seed outputs are preserved, then
averaged summaries and ensemble decision files are written at the end.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (str(THIS_DIR), str(PROJECT_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drl_latent_regime_training import (  # noqa: E402
    RLBaselineConfig,
    compute_performance_metrics,
    run_walkforward_rl_only_baseline,
    save_per_fold_comparison_plot,
    save_portfolio_growth_plot,
    save_strategy_comparison_plot,
)


VARIANTS = ("gating", "no_gating", "no_sentiment")
KEY_COLUMNS = {
    "rl_only_fold_metrics.csv": ["fold", "split"],
    "rl_only_per_fold_test_metrics.csv": ["fold", "strategy"],
    "rl_only_stitched_comparison_metrics.csv": ["strategy"],
}


def _write_per_variant_average(summary_dir: Path, filename: str, df: pd.DataFrame) -> None:
    if "variant" not in df.columns:
        return
    for variant, variant_df in df.groupby("variant", dropna=False):
        variant_dir = summary_dir / str(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_df.drop(columns=["variant"]).to_csv(variant_dir / filename, index=False)


def _parse_seeds(seed_text: str | None, repetitions: int) -> list[int]:
    if seed_text:
        seeds = [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
        if len(seeds) != repetitions:
            raise ValueError(
                f"--seeds provided {len(seeds)} seed(s), but --repetitions is {repetitions}."
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("--seeds must contain unique values.")
        return seeds

    seeds = []
    seen: set[int] = set()
    while len(seeds) < repetitions:
        # Stable-baselines accepts signed 32-bit seeds; keep them positive and readable.
        seed = secrets.randbelow(2_147_483_647)
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _elapsed(start: float) -> str:
    seconds = int(time.monotonic() - start)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _banner(message: str) -> None:
    bar = "=" * min(88, max(24, len(message) + 4))
    print(f"\n{bar}\n  {message}\n{bar}", flush=True)


def _read_variant_csv(run_output_dir: Path, seed: int, variant: str, filename: str) -> pd.DataFrame | None:
    path = run_output_dir / f"seed_{seed}" / variant / filename
    if not path.exists():
        print(f"[WARN] Missing artifact, skipping: {path}", flush=True)
        return None
    df = pd.read_csv(path)
    df["seed"] = seed
    df["variant"] = variant
    return df


def _average_metric_file(
    run_output_dir: Path,
    summary_dir: Path,
    seeds: list[int],
    variants: list[str],
    filename: str,
) -> pd.DataFrame | None:
    frames = [
        frame
        for variant in variants
        for seed in seeds
        if (frame := _read_variant_csv(run_output_dir, seed, variant, filename)) is not None
    ]
    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(summary_dir / f"all_runs_{filename}", index=False)

    key_cols = ["variant", *KEY_COLUMNS[filename]]
    numeric_cols = [
        col
        for col in all_df.select_dtypes(include=[np.number]).columns
        if col not in {"seed"}
    ]
    averaged = (
        all_df.groupby(key_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    averaged.columns = [
        "_".join(part for part in col if part) if isinstance(col, tuple) else col
        for col in averaged.columns
    ]
    output_name = f"average_{filename}"
    averaged.to_csv(summary_dir / output_name, index=False)
    _write_per_variant_average(summary_dir, output_name, averaged)
    return averaged


def _average_curve_file(
    run_output_dir: Path,
    summary_dir: Path,
    seeds: list[int],
    variants: list[str],
    filename: str,
    group_columns: list[str],
) -> pd.DataFrame | None:
    frames = [
        frame
        for variant in variants
        for seed in seeds
        if (frame := _read_variant_csv(run_output_dir, seed, variant, filename)) is not None
    ]
    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df.dropna(subset=["date", "account_value"])
    if all_df.empty:
        return None

    all_df.to_csv(summary_dir / f"all_runs_{filename}", index=False)
    averaged = (
        all_df.groupby(["variant", *group_columns], dropna=False)["account_value"]
        .agg(account_value="mean", account_value_std="std", seed_count="count")
        .reset_index()
        .sort_values(["variant", *group_columns])
    )
    output_name = f"average_{filename}"
    averaged.to_csv(summary_dir / output_name, index=False)
    _write_per_variant_average(summary_dir, output_name, averaged)
    return averaged


def _decision_from_mean(value: float, threshold: float = 1.0 / 3.0) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _aggregate_actions(
    run_output_dir: Path,
    summary_dir: Path,
    seeds: list[int],
    variants: list[str],
    filename: str,
    group_columns: list[str],
) -> pd.DataFrame | None:
    frames = [
        frame
        for variant in variants
        for seed in seeds
        if (frame := _read_variant_csv(run_output_dir, seed, variant, filename)) is not None
    ]
    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df.dropna(subset=["date"])
    if all_df.empty:
        return None

    all_df.to_csv(summary_dir / f"all_runs_{filename}", index=False)
    output_rows: list[dict] = []
    for keys, group in all_df.groupby(["variant", *group_columns], dropna=False):
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
                row[f"mean_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        if "signal" in group.columns:
            mean_signal = float(row["mean_signal"])
            row["ensemble_signal"] = _decision_from_mean(mean_signal)
            row["signal_vote"] = int(
                pd.to_numeric(group["signal"], errors="coerce")
                .dropna()
                .astype(int)
                .mode()
                .iloc[0]
            )
        if "position" in group.columns:
            row["ensemble_position"] = _decision_from_mean(float(row["mean_position"]))
        output_rows.append(row)

    averaged = pd.DataFrame(output_rows).sort_values(["variant", *group_columns])
    output_name = f"ensemble_{filename}"
    averaged.to_csv(summary_dir / output_name, index=False)
    _write_per_variant_average(summary_dir, output_name, averaged)
    return averaged


def _save_average_plots(summary_dir: Path) -> None:
    curves_path = summary_dir / "average_rl_only_portfolio_curves.csv"
    if curves_path.exists():
        curves_df = pd.read_csv(curves_path, parse_dates=["date"])
        for variant, variant_df in curves_df.groupby("variant"):
            save_portfolio_growth_plot(
                portfolio_curves_df=variant_df[["date", "account_value", "fold", "split"]],
                output_path=summary_dir / f"{variant}_average_portfolio_growth.png",
            )

    comparison_path = summary_dir / "average_rl_only_stitched_comparison.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
        for variant, variant_df in comparison_df.groupby("variant"):
            save_strategy_comparison_plot(
                comparison_df=variant_df[["date", "strategy", "account_value"]],
                output_path=summary_dir / f"{variant}_average_stitched_comparison.png",
            )

    per_fold_path = summary_dir / "average_rl_only_per_fold_test_comparison.csv"
    if per_fold_path.exists():
        per_fold_df = pd.read_csv(per_fold_path, parse_dates=["date"])
        for variant, variant_df in per_fold_df.groupby("variant"):
            save_per_fold_comparison_plot(
                per_fold_comparison_df=variant_df[["date", "strategy", "account_value", "fold"]],
                output_path=summary_dir / f"{variant}_average_per_fold_test_comparison.png",
            )


def _write_average_decision_summary(summary_dir: Path, seeds: list[int]) -> pd.DataFrame | None:
    comparison_path = summary_dir / "average_rl_only_stitched_comparison.csv"
    actions_path = summary_dir / "ensemble_rl_only_stitched_test_actions.csv"
    if not comparison_path.exists():
        return None

    comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
    rows: list[dict] = []
    for (variant, strategy), strategy_df in comparison_df.groupby(["variant", "strategy"]):
        actions_df = None
        if actions_path.exists() and strategy == "rl_tdqn_stitched":
            actions_all = pd.read_csv(actions_path, parse_dates=["date"])
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
            strategy_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
            actions_df=actions_df,
        )
        rows.append({"variant": variant, "strategy": strategy, **metrics})

    metrics_df = pd.DataFrame(rows)
    output_name = "average_stitched_comparison_metrics_recomputed.csv"
    metrics_df.to_csv(summary_dir / output_name, index=False)
    _write_per_variant_average(summary_dir, output_name, metrics_df)

    rl_df = metrics_df[metrics_df["strategy"] == "rl_tdqn_stitched"].copy()
    if rl_df.empty:
        return metrics_df
    rl_df = rl_df.sort_values(
        ["sharpe_ratio", "total_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = rl_df.iloc[0].to_dict()
    decision = {
        "selected_variant": best["variant"],
        "selection_rule": "highest average stitched RL Sharpe ratio, then total return, then max drawdown",
        "seeds_used": seeds,
        "selected_metrics": best,
    }
    with open(summary_dir / "average_decision_summary.json", "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2)
    rl_df.to_csv(summary_dir / "average_variant_ranking.csv", index=False)
    return metrics_df


def aggregate_outputs(
    output_dir: Path,
    seeds: list[int],
    variants: list[str],
    config_template: RLBaselineConfig,
) -> None:
    run_output_dir = output_dir / "runs"
    summary_dir = output_dir / "averages"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for filename in KEY_COLUMNS:
        _average_metric_file(run_output_dir, summary_dir, seeds, variants, filename)

    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        variants,
        "rl_only_portfolio_curves.csv",
        ["date", "fold", "split"],
    )
    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        variants,
        "rl_only_stitched_test_equity.csv",
        ["date", "fold"],
    )
    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        variants,
        "rl_only_stitched_comparison.csv",
        ["date", "strategy"],
    )
    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        variants,
        "rl_only_per_fold_test_comparison.csv",
        ["date", "fold", "strategy"],
    )

    _aggregate_actions(
        run_output_dir,
        summary_dir,
        seeds,
        variants,
        "rl_only_stitched_test_actions.csv",
        ["date", "fold"],
    )

    _save_average_plots(summary_dir)
    _write_average_decision_summary(summary_dir, seeds)

    run_config = {
        "base_config": asdict(config_template),
        "variants": variants,
        "seeds": seeds,
        "repetitions": len(seeds),
        "run_order": [
            {"repetition": rep_idx + 1, "seed": seed, "variant": variant}
            for rep_idx, seed in enumerate(seeds)
            for variant in variants
        ],
    }
    with open(summary_dir / "multi_seed_run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run drl_latent_regime_training.py for gating, no_gating, and "
            "no_sentiment across multiple seeds, then average the results."
        )
    )
    parser.add_argument("--input-path", type=str, default="data/spy_market_data.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/baselines/drl_latent_regime_multi_seed",
        help="Root for per-seed runs and averaged outputs.",
    )
    parser.add_argument("--transformer-root", type=str, default="data/transformer_npy")
    parser.add_argument("--regime-root", type=str, default="data/training")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--train-timesteps", type=int, default=200_000)
    parser.add_argument("--initial-amount", type=float, default=10_000.0)
    parser.add_argument("--transaction-cost-pct", type=float, default=1e-3)
    parser.add_argument("--reward-window-k", type=int, default=3)
    parser.add_argument("--progress-interval-steps", type=int, default=25_000)
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Optional comma-separated seeds. Must match --repetitions length. "
            "If omitted, unique random seeds are generated."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="gating,no_gating,no_sentiment",
        help="Comma-separated variants. Default preserves the requested order.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        default=False,
        help="Only aggregate existing per-seed run outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown variant(s): {unknown_variants}. Valid variants: {list(VARIANTS)}")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be >= 1.")

    seeds = _parse_seeds(args.seeds, args.repetitions)
    output_dir = _resolve_path(args.output_dir)
    run_output_dir = output_dir / "runs"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()
    template_config = RLBaselineConfig(
        input_path=args.input_path,
        transformer_root=args.transformer_root,
        variant=variants[0],
        regime_root=args.regime_root,
        output_dir=str(run_output_dir),
        ticker=args.ticker,
        train_timesteps=args.train_timesteps,
        initial_amount=args.initial_amount,
        transaction_cost_pct=args.transaction_cost_pct,
        reward_window_k=args.reward_window_k,
        seed=seeds[0],
        progress_interval_steps=args.progress_interval_steps,
    )

    _banner(
        "Multi-seed latent-regime TDQN: "
        f"{len(seeds)} seed(s) x {len(variants)} variant(s)"
    )
    print(f"Seeds: {seeds}", flush=True)
    print(f"Variants: {variants}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    if not args.skip_runs:
        for rep_idx, seed in enumerate(seeds, start=1):
            for variant in variants:
                run_start = time.monotonic()
                _banner(f"Repetition {rep_idx}/{len(seeds)} | seed {seed} | {variant}")
                config = RLBaselineConfig(
                    input_path=args.input_path,
                    transformer_root=args.transformer_root,
                    variant=variant,
                    regime_root=args.regime_root,
                    output_dir=str(run_output_dir / f"seed_{seed}"),
                    ticker=args.ticker,
                    train_timesteps=args.train_timesteps,
                    initial_amount=args.initial_amount,
                    transaction_cost_pct=args.transaction_cost_pct,
                    reward_window_k=args.reward_window_k,
                    seed=seed,
                    progress_interval_steps=args.progress_interval_steps,
                )
                metrics_df = run_walkforward_rl_only_baseline(
                    config=config,
                    selected_folds=args.folds,
                    max_folds=args.max_folds,
                )
                print(
                    f"Completed {variant} seed {seed} in {_elapsed(run_start)}. "
                    f"Fold metric rows: {len(metrics_df)}",
                    flush=True,
                )

    _banner("Averaging per-seed outputs")
    aggregate_outputs(
        output_dir=output_dir,
        seeds=seeds,
        variants=variants,
        config_template=template_config,
    )
    print(f"Done in {_elapsed(total_start)}.", flush=True)
    print(f"Averaged artifacts: {output_dir / 'averages'}", flush=True)


if __name__ == "__main__":
    main()
