"""
Regime + augmented + ensemble RL baseline.

Builds on `rl_only_augmented_ensemble.py` by appending HMM regime features
(label + per-state probabilities) to the observation. Latent transformer
vectors are NOT included.

Per-episode price augmentation only mutates OHLC / indicator columns; regime
columns are date-keyed and pass through every reset unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
CACHE_ROOT = Path(tempfile.gettempdir()) / "xdg-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.walkforward_folds import FOLDS
from spy.market_data_utils import create_walk_forward_split, load_data

# Load rl_only_augmented_ensemble by file path; it in turn loads rl_only_baseline.
_AUG_ENS_PATH = Path(__file__).resolve().parent / "rl_only_augmented_ensemble.py"
_aug_spec = importlib.util.spec_from_file_location(
    "rl_only_augmented_ensemble_module", _AUG_ENS_PATH
)
if _aug_spec is None or _aug_spec.loader is None:
    raise ImportError(f"Could not load {_AUG_ENS_PATH}")
_aug_module = importlib.util.module_from_spec(_aug_spec)
sys.modules["rl_only_augmented_ensemble_module"] = _aug_module
_aug_spec.loader.exec_module(_aug_module)

AugmentedTradingEnv = _aug_module.AugmentedTradingEnv
SingleAssetTradingEnv = _aug_module.SingleAssetTradingEnv
precompute_training_rewards = _aug_module.precompute_training_rewards
compute_normalization_coeffs = _aug_module.compute_normalization_coeffs
prepare_env_dataframe = _aug_module.prepare_env_dataframe
compute_performance_metrics = _aug_module.compute_performance_metrics
build_stitched_classic_baselines = _aug_module.build_stitched_classic_baselines
save_portfolio_growth_plot = _aug_module.save_portfolio_growth_plot
save_strategy_comparison_plot = _aug_module.save_strategy_comparison_plot
save_per_fold_comparison_plot = _aug_module.save_per_fold_comparison_plot
_resolve_folds = _aug_module._resolve_folds
_load_tdqn_dependencies = _aug_module._load_tdqn_dependencies
_USE_GYMNASIUM = _aug_module._USE_GYMNASIUM
_Spaces = _aug_module._baseline._Spaces


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeAugmentedEnsembleConfig:
    """Config for regime + augmented + ensemble RL runs."""

    input_path: str = "data/spy_market_data.csv"
    regime_root: str = "data/training"
    output_dir: str = "data/baselines/rl_regime_augmented_ensemble"
    ticker: str = "SPY"

    train_timesteps: int = 200_000
    initial_amount: float = 10_000.0
    transaction_cost_pct: float = 1e-3
    reward_window_k: int = 3

    # Ensemble.
    n_seeds: int = 5
    base_seed: int = 42

    # Augmentation. Modes are sampled uniformly per training episode.
    augmentation_modes: tuple[str, ...] = ("none", "shift", "filter", "noise")
    shift_range: tuple[float, float] = (-0.005, 0.005)
    filter_alpha_range: tuple[float, float] = (0.10, 0.30)
    noise_sigma: float = 0.001

    # DQN hparams.
    dqn_learning_rate: float = 1e-4
    dqn_batch_size: int = 64
    dqn_buffer_size: int = 100_000
    dqn_learning_starts: int = 2_000
    dqn_train_freq: int = 4
    dqn_gradient_steps: int = 1
    dqn_target_update_interval: int = 500
    dqn_tau: float = 1.0
    dqn_gamma: float = 0.99
    dqn_exploration_fraction: float = 0.2
    dqn_exploration_final_eps: float = 0.02
    net_arch: tuple[int, ...] = (128, 128)


# ---------------------------------------------------------------------------
# Regime data loading & merging
# ---------------------------------------------------------------------------


def load_fold_regimes(
    regime_root: str | Path,
    fold_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Load regime label/probability columns for one fold/split."""
    regime_path = Path(regime_root) / fold_name / f"spy_{split_name}_labeled.csv"
    if not regime_path.exists():
        raise FileNotFoundError(
            f"Missing regime file for {fold_name} {split_name}: {regime_path}"
        )

    regime_df = pd.read_csv(regime_path, parse_dates=["Date"])
    regime_df = regime_df.rename(columns={"Date": "date"})
    regime_columns = [
        col
        for col in regime_df.columns
        if col == "regime_label" or col.startswith("regime_prob_")
    ]
    if not regime_columns:
        raise ValueError(f"No regime columns found in {regime_path}.")

    regime_df["date"] = pd.to_datetime(regime_df["date"], errors="coerce").dt.normalize()
    return (
        regime_df[["date", *regime_columns]]
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _regime_columns_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c == "regime_label" or c.startswith("regime_prob_")]


def _attach_regime(env_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join regime columns onto an env-ready DataFrame on `date`."""
    out = env_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    regime_aligned = regime_df.copy()
    regime_aligned["date"] = pd.to_datetime(
        regime_aligned["date"], errors="coerce"
    ).dt.normalize()
    merged = out.merge(regime_aligned, on="date", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlap between env DataFrame and regime DataFrame on `date`.")
    return merged.sort_values("date").reset_index(drop=True)


def _compute_norm_coeffs_with_regime(
    train_df: pd.DataFrame, regime_columns: list[str]
) -> dict[str, tuple[float, float]]:
    """Base normalization + per-regime-column min/max from train split."""
    coeffs = compute_normalization_coeffs(train_df)
    for col in regime_columns:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        coeffs[col] = (col_min, col_max)
    return coeffs


# ---------------------------------------------------------------------------
# Regime-aware env wrappers
# ---------------------------------------------------------------------------


def _regime_obs_chunk(env) -> np.ndarray:
    """MinMax-normalized regime feature values at the current step."""
    cols = env._regime_columns
    if not cols:
        return np.empty(0, dtype=np.float32)
    row = env._df.iloc[env._step_idx]
    out = np.empty(len(cols), dtype=np.float32)
    for i, col in enumerate(cols):
        lo, hi = env._norm_coeffs.get(col, (0.0, 1.0))
        span = hi - lo
        if span < 1e-8:
            out[i] = 0.0
        else:
            out[i] = float(np.clip((float(row[col]) - lo) / span, 0.0, 1.0))
    return out


def _extend_obs_space(env, n_extra: int) -> None:
    """Append `n_extra` [0, 1] dims to the env's Box observation_space."""
    if n_extra <= 0:
        return
    low_old = env.observation_space.low
    high_old = env.observation_space.high
    low = np.concatenate([low_old, np.zeros(n_extra, dtype=np.float32)])
    high = np.concatenate([high_old, np.ones(n_extra, dtype=np.float32)])
    env.observation_space = _Spaces.Box(low=low, high=high, dtype=np.float32)


class RegimeTradingEnv(SingleAssetTradingEnv):
    """SingleAssetTradingEnv + regime features appended to the observation."""

    def __init__(self, *, regime_columns: list[str], **kwargs):
        self._regime_columns = list(regime_columns)
        super().__init__(**kwargs)
        _extend_obs_space(self, len(self._regime_columns))

    def _build_obs(self) -> np.ndarray:
        base = super()._build_obs()
        return np.concatenate([base, _regime_obs_chunk(self)]).astype(np.float32)


class RegimeAugmentedTradingEnv(AugmentedTradingEnv):
    """AugmentedTradingEnv + regime features appended to the observation.

    Augmentation only mutates OHLC / indicators; regime columns ride through
    `base_df` unchanged across resets.
    """

    def __init__(self, *, regime_columns: list[str], **kwargs):
        self._regime_columns = list(regime_columns)
        super().__init__(**kwargs)
        _extend_obs_space(self, len(self._regime_columns))

    def _build_obs(self) -> np.ndarray:
        base = super()._build_obs()
        return np.concatenate([base, _regime_obs_chunk(self)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------


def _train_one_model(
    train_df: pd.DataFrame,
    regime_columns: list[str],
    config: RegimeAugmentedEnsembleConfig,
    seed: int,
    norm_coeffs: dict,
):
    DQN, torch = _load_tdqn_dependencies()
    from stable_baselines3.common.callbacks import BaseCallback

    env = RegimeAugmentedTradingEnv(
        regime_columns=regime_columns,
        base_df=train_df,
        augmentation_config=config,
        augmentation_seed=seed,
        initial_amount=config.initial_amount,
        transaction_cost_pct=config.transaction_cost_pct,
        reward_window_k=config.reward_window_k,
        mode="train",
        norm_coeffs=norm_coeffs,
    )

    class _CounterfactualCallback(BaseCallback):
        """Push counterfactual transitions for non-taken actions into the
        replay buffer (Théate & Ernst, 2020)."""

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [{}])
            actions = self.locals.get("actions", [])
            new_obs = self.locals.get("new_obs")
            dones = self.locals.get("dones")
            if new_obs is None or dones is None:
                return True
            last_obs = self.model._last_obs
            for i, info in enumerate(infos):
                cf_rewards = info.get("counterfactual_rewards", {})
                if not cf_rewards:
                    continue
                actual_action = int(np.asarray(actions[i]).reshape(-1)[0])
                for cf_idx, cf_reward in cf_rewards.items():
                    if cf_idx == actual_action:
                        continue
                    self.model.replay_buffer.add(
                        last_obs[i : i + 1],
                        new_obs[i : i + 1],
                        np.array([[cf_idx]], dtype=np.int64),
                        np.array([cf_reward], dtype=np.float32),
                        np.array([bool(dones[i])]),
                        [{}],
                    )
            return True

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config.dqn_learning_rate,
        buffer_size=config.dqn_buffer_size,
        learning_starts=config.dqn_learning_starts,
        batch_size=config.dqn_batch_size,
        tau=config.dqn_tau,
        target_update_interval=config.dqn_target_update_interval,
        train_freq=config.dqn_train_freq,
        gradient_steps=config.dqn_gradient_steps,
        gamma=config.dqn_gamma,
        exploration_fraction=config.dqn_exploration_fraction,
        exploration_final_eps=config.dqn_exploration_final_eps,
        policy_kwargs=dict(
            net_arch=list(config.net_arch),
            activation_fn=torch.nn.ReLU,
        ),
        seed=seed,
        verbose=0,
    )
    model.learn(
        total_timesteps=config.train_timesteps,
        progress_bar=False,
        callback=_CounterfactualCallback(),
    )
    return model


def train_fold_ensemble(
    train_df: pd.DataFrame,
    regime_columns: list[str],
    config: RegimeAugmentedEnsembleConfig,
) -> tuple[list, dict]:
    """Train `n_seeds` DQN models on the same train split with different seeds."""
    norm_coeffs = _compute_norm_coeffs_with_regime(train_df, regime_columns)
    models = []
    for i in range(config.n_seeds):
        seed = config.base_seed + i
        print(f"  training seed {i + 1}/{config.n_seeds} (seed={seed})", flush=True)
        models.append(
            _train_one_model(train_df, regime_columns, config, seed, norm_coeffs)
        )
    return models, norm_coeffs


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------


def _ensemble_action(models: list, obs: np.ndarray) -> int:
    """Average Q-values across models, return argmax action."""
    import torch

    q_sum = None
    for model in models:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            q = model.q_net(obs_tensor).cpu().numpy()
        q_sum = q if q_sum is None else q_sum + q
    return int(np.argmax(q_sum.squeeze(0)))


def evaluate_ensemble_on_split(
    models: list,
    split_df: pd.DataFrame,
    regime_columns: list[str],
    config: RegimeAugmentedEnsembleConfig,
    initial_state: dict | None = None,
    norm_coeffs: dict | None = None,
):
    """Deterministic ensemble rollout on an un-augmented eval split."""
    env = RegimeTradingEnv(
        regime_columns=regime_columns,
        df=split_df,
        initial_amount=config.initial_amount,
        transaction_cost_pct=config.transaction_cost_pct,
        reward_window_k=config.reward_window_k,
        mode="eval",
        initial_state=initial_state,
        norm_coeffs=norm_coeffs,
    )

    if _USE_GYMNASIUM:
        obs, _ = env.reset(options=initial_state)
    else:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

    done = False
    max_steps = len(split_df) + 5
    step_count = 0
    while not done:
        action_int = _ensemble_action(models, obs)
        step_out = env.step(action_int)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, _ = step_out
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError("Evaluation exceeded expected number of steps.")

    start_val = (
        float(initial_state["account_value"])
        if initial_state is not None
        else config.initial_amount
    )
    dates = list(split_df["date"])
    values = [start_val] + env.account_value_history

    account_value_df = pd.DataFrame({"date": dates, "account_value": values})
    account_value_df["date"] = pd.to_datetime(account_value_df["date"], errors="coerce")
    account_value_df = (
        account_value_df.dropna(subset=["date", "account_value"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    n = len(env.date_history)
    account_arr = np.array(env.account_value_history, dtype=float)
    prev_account = (
        np.concatenate([[start_val], account_arr[:-1]])
        if n > 1
        else np.array([start_val])
    )
    step_pnl = account_arr - prev_account
    running_peak = np.maximum.accumulate(account_arr) if n > 0 else account_arr
    drawdown = running_peak - account_arr if n > 0 else account_arr
    positions = np.array(env.position_history, dtype=float)

    actions_df = pd.DataFrame(
        {
            "date": env.date_history,
            "signal": env.signal_history,
            "actual_action": env.actual_action_history,
            "position": env.position_history,
            "position_after_trade": env.position_history,
            "weight": positions,
            "trade_price": env.price_history,
            "trade_notional": env.trade_notional_history,
            "pnl": step_pnl,
            "drawdown": drawdown,
        }
    )
    actions_df["date"] = pd.to_datetime(actions_df["date"], errors="coerce")

    terminal_state = {
        "account_value": env.account_value_history[-1] if env.account_value_history else start_val,
        "position": env.position_history[-1] if env.position_history else 0,
    }
    return account_value_df, actions_df, terminal_state


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------


_OUT_PREFIX = "rl_regime_aug_ens"
_RL_STRATEGY_NAME = "rl_tdqn_regime_aug_ens"


def run_walkforward_regime_augmented_ensemble(
    config: RegimeAugmentedEnsembleConfig,
    selected_folds: str | None = None,
    max_folds: int | None = None,
) -> pd.DataFrame:
    full_df = load_data(path=config.input_path)
    folds = _resolve_folds(selected_folds=selected_folds, max_folds=max_folds)

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    portfolio_curves: list[pd.DataFrame] = []
    stitched_test_curves: list[pd.DataFrame] = []
    stitched_test_actions: list[pd.DataFrame] = []
    stitched_previous_state: dict | None = None
    per_fold_comparisons: list[pd.DataFrame] = []
    per_fold_test_actions: dict[str, pd.DataFrame] = {}

    for fold in folds:
        print(
            f"Running {fold['name']}: train {fold['train_start']}→{fold['train_end']}, "
            f"val {fold['val_start']}→{fold['val_end']}, "
            f"test {fold['test_start']}→{fold['test_end']}"
        )
        train_raw, val_raw, test_raw = create_walk_forward_split(
            full_df,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )

        train_regimes = load_fold_regimes(config.regime_root, fold["name"], "train")
        val_regimes = load_fold_regimes(config.regime_root, fold["name"], "val")
        test_regimes = load_fold_regimes(config.regime_root, fold["name"], "test")
        regime_columns = _regime_columns_of(train_regimes)

        train_df = _attach_regime(prepare_env_dataframe(train_raw), train_regimes)
        val_df = _attach_regime(prepare_env_dataframe(val_raw), val_regimes)
        test_df = _attach_regime(prepare_env_dataframe(test_raw), test_regimes)

        models, norm_coeffs = train_fold_ensemble(
            train_df=train_df, regime_columns=regime_columns, config=config
        )

        fold_dir = output_root / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_test_account_df: pd.DataFrame | None = None
        for split_name, split_df in (("val", val_df), ("test", test_df)):
            account_df, actions_df, _ = evaluate_ensemble_on_split(
                models=models,
                split_df=split_df,
                regime_columns=regime_columns,
                config=config,
                norm_coeffs=norm_coeffs,
            )
            metrics = compute_performance_metrics(account_df, actions_df=actions_df)

            account_df.to_csv(fold_dir / f"{split_name}_account_value.csv", index=False)
            actions_df.to_csv(fold_dir / f"{split_name}_actions.csv", index=False)
            portfolio_curves.append(
                account_df.assign(fold=fold["name"], split=split_name)[
                    ["date", "account_value", "fold", "split"]
                ]
            )
            metrics_rows.append(
                {
                    "fold": fold["name"],
                    "split": split_name,
                    "train_rows": len(train_df),
                    "eval_rows": len(split_df),
                    "account_points": len(account_df),
                    **metrics,
                }
            )
            if split_name == "test":
                fold_test_account_df = account_df
                per_fold_test_actions[fold["name"]] = actions_df

        if fold_test_account_df is not None:
            fold_test_dates = pd.to_datetime(
                fold_test_account_df["date"], errors="coerce"
            ).dropna()
            fold_classic_df = build_stitched_classic_baselines(
                full_df=full_df,
                stitched_dates=fold_test_dates,
                initial_amount=config.initial_amount,
                momentum_lookback=126,
            )
            fold_rl_curve = fold_test_account_df[["date", "account_value"]].assign(
                strategy="rl_tdqn"
            )
            fold_comparison = pd.concat([fold_rl_curve, fold_classic_df], ignore_index=True)
            fold_comparison["fold"] = fold["name"]
            per_fold_comparisons.append(fold_comparison)

        stitched_account_df, stitched_actions_df, stitched_previous_state = (
            evaluate_ensemble_on_split(
                models=models,
                split_df=test_df,
                regime_columns=regime_columns,
                config=config,
                initial_state=stitched_previous_state,
                norm_coeffs=norm_coeffs,
            )
        )
        stitched_account_df.to_csv(
            fold_dir / "test_account_value_stitched.csv", index=False
        )
        stitched_actions_df.to_csv(fold_dir / "test_actions_stitched.csv", index=False)
        stitched_test_curves.append(
            stitched_account_df.assign(fold=fold["name"])[["date", "account_value", "fold"]]
        )
        stitched_test_actions.append(
            stitched_actions_df.assign(fold=fold["name"])[
                [
                    "date",
                    "signal",
                    "actual_action",
                    "position",
                    "position_after_trade",
                    "weight",
                    "trade_price",
                    "trade_notional",
                    "pnl",
                    "drawdown",
                    "fold",
                ]
            ]
        )

    if per_fold_comparisons:
        per_fold_df = pd.concat(per_fold_comparisons, ignore_index=True)
        per_fold_df["date"] = pd.to_datetime(per_fold_df["date"], errors="coerce")
        per_fold_df = (
            per_fold_df.dropna(subset=["date", "account_value", "strategy"])
            .sort_values(["fold", "strategy", "date"])
            .reset_index(drop=True)
        )
        per_fold_df.to_csv(
            output_root / f"{_OUT_PREFIX}_per_fold_test_comparison.csv", index=False
        )
        save_per_fold_comparison_plot(
            per_fold_comparison_df=per_fold_df,
            output_path=output_root / f"{_OUT_PREFIX}_per_fold_test_comparison.png",
        )
        per_fold_metrics_rows: list[dict] = []
        for (fold_name, strategy_name), strat_df in per_fold_df.groupby(["fold", "strategy"]):
            strat_actions_df = (
                per_fold_test_actions.get(fold_name) if strategy_name == "rl_tdqn" else None
            )
            strat_metrics = compute_performance_metrics(
                strat_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
                actions_df=strat_actions_df,
            )
            per_fold_metrics_rows.append(
                {"fold": fold_name, "strategy": strategy_name, **strat_metrics}
            )
        pd.DataFrame(per_fold_metrics_rows).to_csv(
            output_root / f"{_OUT_PREFIX}_per_fold_test_metrics.csv", index=False
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_root / f"{_OUT_PREFIX}_fold_metrics.csv", index=False)

    if portfolio_curves:
        curves_df = pd.concat(portfolio_curves, ignore_index=True)
        curves_df.to_csv(output_root / f"{_OUT_PREFIX}_portfolio_curves.csv", index=False)
        save_portfolio_growth_plot(
            portfolio_curves_df=curves_df,
            output_path=output_root / f"{_OUT_PREFIX}_portfolio_growth.png",
        )

    if stitched_test_curves:
        stitched_rl_df = pd.concat(stitched_test_curves, ignore_index=True)
        stitched_rl_df["date"] = pd.to_datetime(stitched_rl_df["date"], errors="coerce")
        stitched_rl_df = (
            stitched_rl_df.dropna(subset=["date", "account_value"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        stitched_rl_df.to_csv(
            output_root / f"{_OUT_PREFIX}_stitched_test_equity.csv", index=False
        )
        all_stitched_actions = pd.concat(stitched_test_actions, ignore_index=True)
        all_stitched_actions.to_csv(
            output_root / f"{_OUT_PREFIX}_stitched_test_actions.csv", index=False
        )

        classic_df = build_stitched_classic_baselines(
            full_df=full_df,
            stitched_dates=stitched_rl_df["date"],
            initial_amount=config.initial_amount,
            momentum_lookback=126,
        )
        stitched_rl_long = stitched_rl_df[["date", "account_value"]].assign(
            strategy=_RL_STRATEGY_NAME
        )
        comparison_df = pd.concat([stitched_rl_long, classic_df], ignore_index=True)
        comparison_df["date"] = pd.to_datetime(comparison_df["date"], errors="coerce")
        comparison_df = (
            comparison_df.dropna(subset=["date", "account_value", "strategy"])
            .sort_values(["date", "strategy"])
            .reset_index(drop=True)
        )
        comparison_df.to_csv(
            output_root / f"{_OUT_PREFIX}_stitched_comparison.csv", index=False
        )
        save_strategy_comparison_plot(
            comparison_df=comparison_df,
            output_path=output_root / f"{_OUT_PREFIX}_stitched_comparison.png",
        )
        stitched_metrics_rows: list[dict] = []
        for strategy_name, strategy_df in comparison_df.groupby("strategy"):
            strategy_actions_df = (
                all_stitched_actions if strategy_name == _RL_STRATEGY_NAME else None
            )
            strategy_metrics = compute_performance_metrics(
                strategy_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
                actions_df=strategy_actions_df,
            )
            stitched_metrics_rows.append({"strategy": strategy_name, **strategy_metrics})
        pd.DataFrame(stitched_metrics_rows).to_csv(
            output_root / f"{_OUT_PREFIX}_stitched_comparison_metrics.csv", index=False
        )

    with open(output_root / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2, default=list)

    return metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime + augmented + ensemble RL baseline."
    )
    parser.add_argument("--input-path", type=str, default="data/spy_market_data.csv")
    parser.add_argument(
        "--regime-root",
        type=str,
        default="data/training",
        help="Root containing foldN/spy_{split}_labeled.csv regime files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/baselines/rl_regime_augmented_ensemble",
    )
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--train-timesteps", type=int, default=200_000)
    parser.add_argument("--initial-amount", type=float, default=10_000.0)
    parser.add_argument("--transaction-cost-pct", type=float, default=1e-3)
    parser.add_argument("--reward-window-k", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-modes",
        type=str,
        default="none,shift,filter,noise",
        help="Comma-separated modes sampled per training episode.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.001,
        help="Per-row multiplicative Gaussian σ for `noise` augmentation.",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default="128,128",
        help="Comma-separated hidden layer sizes for MlpPolicy.",
    )
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aug_modes = tuple(
        m.strip() for m in args.augmentation_modes.split(",") if m.strip()
    )
    net_arch = tuple(int(x) for x in args.net_arch.split(",") if x.strip())
    config = RegimeAugmentedEnsembleConfig(
        input_path=args.input_path,
        regime_root=args.regime_root,
        output_dir=args.output_dir,
        ticker=args.ticker,
        train_timesteps=args.train_timesteps,
        initial_amount=args.initial_amount,
        transaction_cost_pct=args.transaction_cost_pct,
        reward_window_k=args.reward_window_k,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        augmentation_modes=aug_modes,
        noise_sigma=args.noise_sigma,
        net_arch=net_arch,
    )
    metrics_df = run_walkforward_regime_augmented_ensemble(
        config=config,
        selected_folds=args.folds,
        max_folds=args.max_folds,
    )
    print("RL regime+augmented+ensemble complete. Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
