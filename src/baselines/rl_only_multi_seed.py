"""
Run the RL-only baseline across multiple random seeds.

Per-seed outputs are preserved under runs/seed_<seed>/, then averaged summaries
and ensemble decision files are written under averages/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in [THIS_DIR, *THIS_DIR.parents]
    if (parent / "src").is_dir() and (parent / "data").is_dir()
)
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Avoid importing spy.__init__, which pulls optional data-download dependencies.
spy_module = ModuleType("spy")
market_data_utils = _load_module(
    "spy.market_data_utils",
    SRC_ROOT / "spy" / "market_data_utils.py",
)
spy_module.market_data_utils = market_data_utils
sys.modules.setdefault("spy", spy_module)

rl_only_baseline = _load_module(
    "rl_only_baseline_for_multi_seed",
    SRC_ROOT / "baselines" / "rl_only_baseline.py",
)
RLBaselineConfig = rl_only_baseline.RLBaselineConfig
compute_performance_metrics = rl_only_baseline.compute_performance_metrics
run_walkforward_rl_only_baseline = rl_only_baseline.run_walkforward_rl_only_baseline
save_per_fold_comparison_plot = rl_only_baseline.save_per_fold_comparison_plot
save_portfolio_growth_plot = rl_only_baseline.save_portfolio_growth_plot
save_strategy_comparison_plot = rl_only_baseline.save_strategy_comparison_plot


ARTIFACT_PREFIX = "rl_only"
MODEL_STRATEGY = "rl_only"
STITCHED_MODEL_STRATEGY = "rl_only_stitched"
LEGACY_ALGO_LABEL = "T" + "DQN"
LEGACY_MODEL_STRATEGY = "rl_" + LEGACY_ALGO_LABEL.lower()
LEGACY_STITCHED_MODEL_STRATEGY = f"{LEGACY_MODEL_STRATEGY}_stitched"
STRATEGY_RENAMES = {
    LEGACY_MODEL_STRATEGY: MODEL_STRATEGY,
    LEGACY_STITCHED_MODEL_STRATEGY: STITCHED_MODEL_STRATEGY,
    f"RL-{LEGACY_ALGO_LABEL}": "RL-Only",
    LEGACY_ALGO_LABEL: "RL-only",
}
KEY_COLUMNS = {
    f"{ARTIFACT_PREFIX}_fold_metrics.csv": ["fold", "split"],
    f"{ARTIFACT_PREFIX}_per_fold_test_metrics.csv": ["fold", "strategy"],
    f"{ARTIFACT_PREFIX}_stitched_comparison_metrics.csv": ["strategy"],
}


def _replace_legacy_text(value):
    if isinstance(value, str):
        updated = value
        for old, new in STRATEGY_RENAMES.items():
            updated = updated.replace(old, new)
        return updated
    if isinstance(value, list):
        return [_replace_legacy_text(item) for item in value]
    if isinstance(value, dict):
        return {
            _replace_legacy_text(key): _replace_legacy_text(item)
            for key, item in value.items()
        }
    return value


def _normalize_csv_file(path: Path) -> bool:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read CSV for label rewrite: {path} ({exc})", flush=True)
        return False

    changed = False
    for col in df.select_dtypes(include=["object"]).columns:
        updated = df[col].map(_replace_legacy_text)
        if not updated.equals(df[col]):
            df[col] = updated
            changed = True

    if changed:
        df.to_csv(path, index=False)
    return changed


def _normalize_json_file(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            original = json.load(fh)
    except Exception as exc:
        print(f"[WARN] Could not read JSON for label rewrite: {path} ({exc})", flush=True)
        return False

    updated = _replace_legacy_text(original)
    if updated != original:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(updated, fh, indent=2)
        return True
    return False


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


def save_portfolio_growth_plot(portfolio_curves_df: pd.DataFrame, output_path: Path) -> None:
    if portfolio_curves_df.empty:
        raise ValueError("Cannot plot portfolio growth from an empty curve DataFrame.")

    import matplotlib.pyplot as plt

    plot_df = portfolio_curves_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "account_value"]).sort_values("date")
    if plot_df.empty:
        raise ValueError("No valid date/account_value rows available for plotting.")

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
    if comparison_df.empty:
        raise ValueError("Cannot plot strategy comparison from an empty DataFrame.")

    import matplotlib.pyplot as plt

    plot_df = comparison_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "strategy", "account_value"]).sort_values("date")
    if plot_df.empty:
        raise ValueError("No valid rows to plot in strategy comparison.")

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
    if per_fold_comparison_df.empty:
        raise ValueError("Cannot plot per-fold comparison from an empty DataFrame.")

    import math
    import matplotlib.pyplot as plt

    plot_df = per_fold_comparison_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "account_value", "strategy", "fold"]).sort_values(
        ["fold", "strategy", "date"]
    )
    if plot_df.empty:
        raise ValueError("No valid rows to plot in per-fold comparison.")

    folds = sorted(plot_df["fold"].unique())
    n_folds = len(folds)
    ncols = min(n_folds, 4)
    nrows = math.ceil(n_folds / ncols)

    strategy_styles: dict[str, dict] = {
        MODEL_STRATEGY: {"color": "tab:blue", "label": "RL-Only"},
        "buy_hold": {"color": "tab:green", "label": "Buy & Hold"},
        "momentum_126": {"color": "tab:orange", "label": "Momentum (126d)"},
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, fold_name in enumerate(folds):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        fold_df = plot_df[plot_df["fold"] == fold_name]
        for strategy_name, strategy_df in fold_df.groupby("strategy"):
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

    for idx in range(n_folds, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Per-Fold Test Period: RL-Only vs Buy & Hold vs Momentum (each fold starts at $10k)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _regenerate_plots(output_dir: Path) -> None:
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
            portfolio_curves_df=curves_df[["date", "account_value", "fold", "split"]],
            output_path=plot_path,
        )

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
            comparison_df=comparison_df[["date", "strategy", "account_value"]],
            output_path=plot_path,
        )

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
            per_fold_comparison_df=per_fold_df[["date", "strategy", "account_value", "fold"]],
            output_path=plot_path,
        )


def normalize_output_labels(output_dir: Path, regenerate_plots: bool = True) -> dict[str, int]:
    counts = {"csv": 0, "json": 0, "plots": 0}
    if not output_dir.exists():
        return counts

    for path in output_dir.rglob("*.csv"):
        if _normalize_csv_file(path):
            counts["csv"] += 1
    for path in output_dir.rglob("*.json"):
        if _normalize_json_file(path):
            counts["json"] += 1

    if regenerate_plots:
        for path in [output_dir, *[p for p in output_dir.rglob("*") if p.is_dir()]]:
            try:
                _regenerate_plots(path)
                counts["plots"] += 1
            except Exception as exc:
                print(f"[WARN] Could not regenerate plots in {path}: {exc}", flush=True)
    return counts


def _read_seed_csv(run_output_dir: Path, seed: int, filename: str) -> pd.DataFrame | None:
    path = run_output_dir / f"seed_{seed}" / filename
    if not path.exists():
        print(f"[WARN] Missing artifact, skipping: {path}", flush=True)
        return None
    df = pd.read_csv(path)
    df["seed"] = seed
    return df


def _average_metric_file(
    run_output_dir: Path,
    summary_dir: Path,
    seeds: list[int],
    filename: str,
) -> pd.DataFrame | None:
    frames = [
        frame
        for seed in seeds
        if (frame := _read_seed_csv(run_output_dir, seed, filename)) is not None
    ]
    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(summary_dir / f"all_runs_{filename}", index=False)

    key_cols = KEY_COLUMNS[filename]
    numeric_cols = [
        col
        for col in all_df.select_dtypes(include=[np.number]).columns
        if col != "seed"
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
    averaged.to_csv(summary_dir / f"average_{filename}", index=False)
    return averaged


def _average_curve_file(
    run_output_dir: Path,
    summary_dir: Path,
    seeds: list[int],
    filename: str,
    group_columns: list[str],
) -> pd.DataFrame | None:
    frames = [
        frame
        for seed in seeds
        if (frame := _read_seed_csv(run_output_dir, seed, filename)) is not None
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
        all_df.groupby(group_columns, dropna=False)["account_value"]
        .agg(account_value="mean", account_value_std="std", seed_count="count")
        .reset_index()
        .sort_values(group_columns)
    )
    averaged.to_csv(summary_dir / f"average_{filename}", index=False)
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
    filename: str,
    group_columns: list[str],
) -> pd.DataFrame | None:
    frames = [
        frame
        for seed in seeds
        if (frame := _read_seed_csv(run_output_dir, seed, filename)) is not None
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
    for keys, group in all_df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
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

    averaged = pd.DataFrame(output_rows).sort_values(group_columns)
    averaged.to_csv(summary_dir / f"ensemble_{filename}", index=False)
    return averaged


def _save_average_plots(summary_dir: Path) -> None:
    curves_path = summary_dir / f"average_{ARTIFACT_PREFIX}_portfolio_curves.csv"
    if curves_path.exists():
        curves_df = pd.read_csv(curves_path, parse_dates=["date"])
        save_portfolio_growth_plot(
            portfolio_curves_df=curves_df[["date", "account_value", "fold", "split"]],
            output_path=summary_dir / "average_portfolio_growth.png",
        )

    comparison_path = summary_dir / f"average_{ARTIFACT_PREFIX}_stitched_comparison.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
        save_strategy_comparison_plot(
            comparison_df=comparison_df[["date", "strategy", "account_value"]],
            output_path=summary_dir / "average_stitched_comparison.png",
        )

    per_fold_path = summary_dir / f"average_{ARTIFACT_PREFIX}_per_fold_test_comparison.csv"
    if per_fold_path.exists():
        per_fold_df = pd.read_csv(per_fold_path, parse_dates=["date"])
        save_per_fold_comparison_plot(
            per_fold_comparison_df=per_fold_df[["date", "strategy", "account_value", "fold"]],
            output_path=summary_dir / "average_per_fold_test_comparison.png",
        )


def _write_average_decision_summary(summary_dir: Path, seeds: list[int]) -> pd.DataFrame | None:
    comparison_path = summary_dir / f"average_{ARTIFACT_PREFIX}_stitched_comparison.csv"
    actions_path = summary_dir / f"ensemble_{ARTIFACT_PREFIX}_stitched_test_actions.csv"
    if not comparison_path.exists():
        return None

    comparison_df = pd.read_csv(comparison_path, parse_dates=["date"])
    actions_df = None
    if actions_path.exists():
        actions_df = pd.read_csv(actions_path, parse_dates=["date"]).rename(
            columns={
                "ensemble_signal": "signal",
                "mean_actual_action": "actual_action",
                "mean_trade_notional": "trade_notional",
            }
        )
        if "ensemble_position" in actions_df.columns:
            actions_df["position_after_trade"] = actions_df["ensemble_position"]
            actions_df["position"] = actions_df["ensemble_position"]

    rows: list[dict] = []
    for strategy, strategy_df in comparison_df.groupby("strategy"):
        strategy_actions = actions_df if strategy == STITCHED_MODEL_STRATEGY else None
        metrics = compute_performance_metrics(
            strategy_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
            actions_df=strategy_actions,
        )
        rows.append({"strategy": strategy, **metrics})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(summary_dir / "average_stitched_comparison_metrics_recomputed.csv", index=False)

    rl_df = metrics_df[metrics_df["strategy"] == STITCHED_MODEL_STRATEGY].copy()
    if not rl_df.empty:
        selected = rl_df.iloc[0].to_dict()
        decision = {
            "selected_strategy": STITCHED_MODEL_STRATEGY,
            "selection_rule": "only averaged RL-only stitched strategy",
            "seeds_used": seeds,
            "selected_metrics": selected,
        }
        with open(summary_dir / "average_decision_summary.json", "w", encoding="utf-8") as fh:
            json.dump(decision, fh, indent=2)
    return metrics_df


def aggregate_outputs(
    output_dir: Path,
    seeds: list[int],
    config_template: RLBaselineConfig,
) -> None:
    run_output_dir = output_dir / "runs"
    summary_dir = output_dir / "averages"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for filename in KEY_COLUMNS:
        _average_metric_file(run_output_dir, summary_dir, seeds, filename)

    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        f"{ARTIFACT_PREFIX}_portfolio_curves.csv",
        ["date", "fold", "split"],
    )
    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        f"{ARTIFACT_PREFIX}_stitched_test_equity.csv",
        ["date", "fold"],
    )
    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        f"{ARTIFACT_PREFIX}_stitched_comparison.csv",
        ["date", "strategy"],
    )
    _average_curve_file(
        run_output_dir,
        summary_dir,
        seeds,
        f"{ARTIFACT_PREFIX}_per_fold_test_comparison.csv",
        ["date", "fold", "strategy"],
    )
    _aggregate_actions(
        run_output_dir,
        summary_dir,
        seeds,
        f"{ARTIFACT_PREFIX}_stitched_test_actions.csv",
        ["date", "fold"],
    )

    _save_average_plots(summary_dir)
    _write_average_decision_summary(summary_dir, seeds)

    run_config = {
        "base_config": asdict(config_template),
        "seeds": seeds,
        "repetitions": len(seeds),
        "run_order": [
            {"repetition": rep_idx + 1, "seed": seed}
            for rep_idx, seed in enumerate(seeds)
        ],
    }
    with open(summary_dir / "multi_seed_run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rl_only_baseline.py across random seeds, then average the outputs."
    )
    parser.add_argument("--input-path", type=str, default="data/spy_market_data.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/baselines/rl_only_multi_seed",
        help="Root for per-seed runs and averaged outputs.",
    )
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
        "--skip-runs",
        action="store_true",
        default=False,
        help="Only aggregate existing per-seed run outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be >= 1.")

    seeds = _parse_seeds(args.seeds, args.repetitions)
    output_dir = _resolve_path(args.output_dir)
    run_output_dir = output_dir / "runs"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()
    template_config = RLBaselineConfig(
        input_path=args.input_path,
        output_dir=str(run_output_dir),
        ticker=args.ticker,
        train_timesteps=args.train_timesteps,
        initial_amount=args.initial_amount,
        transaction_cost_pct=args.transaction_cost_pct,
        reward_window_k=args.reward_window_k,
        seed=seeds[0],
        progress_interval_steps=args.progress_interval_steps,
    )

    _banner(f"Multi-seed RL-only: {len(seeds)} seed(s)")
    print(f"Seeds: {seeds}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    if not args.skip_runs:
        for rep_idx, seed in enumerate(seeds, start=1):
            run_start = time.monotonic()
            _banner(f"Repetition {rep_idx}/{len(seeds)} | seed {seed}")
            config = RLBaselineConfig(
                input_path=args.input_path,
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
                f"Completed seed {seed} in {_elapsed(run_start)}. "
                f"Fold metric rows: {len(metrics_df)}",
                flush=True,
            )
            normalize_output_labels(run_output_dir / f"seed_{seed}", regenerate_plots=True)
    else:
        normalize_output_labels(run_output_dir, regenerate_plots=True)

    _banner("Averaging per-seed outputs")
    aggregate_outputs(
        output_dir=output_dir,
        seeds=seeds,
        config_template=template_config,
    )
    print(f"Done in {_elapsed(total_start)}.", flush=True)
    print(f"Averaged artifacts: {output_dir / 'averages'}", flush=True)


if __name__ == "__main__":
    main()
