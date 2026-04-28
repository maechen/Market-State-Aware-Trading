"""
Combined DRL baseline using FinRL + DQN with transformer latents and regimes.

This script keeps the RL-only baseline structure, while adding:
- transformer latent vectors from `data/transformer_npy/{variant}/foldN`
- GHMM regime labels/posteriors from fold-level labeled CSV files
- default training length of 200k timesteps per fold
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import site
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure matplotlib (pulled in by FinRL) has a writable cache path.
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
CACHE_ROOT = Path(tempfile.gettempdir()) / "xdg-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

# Allow running as `python scripts/drl_latent_regime_training.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.walkforward_folds import FOLDS

try:
    from spy.market_data_utils import create_walk_forward_split, load_data
except ModuleNotFoundError as exc:
    if exc.name != "yfinance":
        raise
    market_utils_path = SRC_ROOT / "spy" / "market_data_utils.py"
    market_utils_spec = importlib.util.spec_from_file_location(
        "spy_market_data_utils_fallback",
        market_utils_path,
    )
    if market_utils_spec is None or market_utils_spec.loader is None:
        raise ImportError(f"Could not load {market_utils_path}") from exc
    market_utils_module = importlib.util.module_from_spec(market_utils_spec)
    market_utils_spec.loader.exec_module(market_utils_module)
    create_walk_forward_split = market_utils_module.create_walk_forward_split
    load_data = market_utils_module.load_data

# Raw market columns expected from `spy_market_data.csv`.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
# Technical indicators requested for the combined DRL baseline.
INDICATOR_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "rolling_vol_20",
    "ma_10",
    "ma_20",
    "ma_50",
]

# FinRL expects OHLCV as lowercase; `close` is consumed directly by the env
# while the rest are fed via technical indicator list.
FINRL_TECH_COLUMNS = [
    "open",
    "high",
    "low",
    "volume",
    "rsi_14",
    "macd",
    "macd_signal",
    "rolling_vol_20",
    "ma_10",
    "ma_20",
    "ma_50",
]

BASE_FINRL_TECH_COLUMNS = list(FINRL_TECH_COLUMNS)


@dataclass(frozen=True)
class RLBaselineConfig:
    """Central config for reproducible fold-wise RL baseline runs."""

    # Data path and output path.
    input_path: str = "data/spy_market_data.csv"
    transformer_root: str = "data/transformer_npy"
    variant: str = "gating"
    regime_root: str = "data/training"
    output_dir: str = "data/baselines/drl_latent_regime"

    # Single-asset ticker used to label FinRL input rows.
    ticker: str = "SPY"

    # Core training/backtest controls.
    train_timesteps: int = 200_000
    initial_amount: float = 10000.0
    transaction_cost_pct: float = 1e-3
    reward_scaling: float = 1e-4
    hmax: int = 100
    downside_penalty_alpha: float = 1.0
    trade_penalty_beta: float = 1e-3
    seed: int = 42

    # Discrete actions are mapped into FinRL's continuous action range [-1, 1].
    action_grid: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

    # DQN hyperparameters (kept explicit so they are easy to tune later).
    dqn_learning_rate: float = 1e-4
    dqn_batch_size: int = 64
    dqn_buffer_size: int = 100_000
    dqn_learning_starts: int = 1_000
    dqn_train_freq: int = 4
    dqn_target_update_interval: int = 500
    dqn_exploration_fraction: float = 0.4
    dqn_exploration_final_eps: float = 0.02


def _parse_action_grid(raw: str) -> tuple[float, ...]:
    """Parse CLI action grid string and enforce valid discrete-action values."""

    # Example input: "-1.0,-0.5,0.0,0.5,1.0".
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise ValueError("Action grid needs at least 2 values (e.g. -1,0,1).")
    # Always keep an explicit hold action.
    if 0.0 not in values:
        raise ValueError("Action grid must include 0.0 (hold action).")
    # Weight deltas are bounded to [-1, 1] (full sell to full buy).
    if any(value < -1.0 or value > 1.0 for value in values):
        raise ValueError("All action grid values must be in [-1.0, 1.0].")
    return values


def _resolve_folds(selected_folds: str | None, max_folds: int | None) -> list[dict[str, str]]:
    """Resolve fold selection from CLI (`--folds`, `--max-folds`)."""

    if selected_folds:
        # Support comma-separated names like "fold1,fold3".
        selected_names = {name.strip() for name in selected_folds.split(",") if name.strip()}
        resolved = [fold for fold in FOLDS if fold["name"] in selected_names]
        missing = selected_names - {fold["name"] for fold in resolved}
        if missing:
            raise ValueError(f"Unknown fold names: {sorted(missing)}")
    else:
        resolved = list(FOLDS)

    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError("--max-folds must be >= 1.")
        resolved = resolved[:max_folds]

    if not resolved:
        raise ValueError("No folds selected.")
    return resolved


def _latent_column_names(latent_dim: int) -> list[str]:
    """Return stable names for transformer latent dimensions."""

    if latent_dim <= 0:
        raise ValueError("Latent vectors must have at least one dimension.")
    return [f"latent_{idx:03d}" for idx in range(latent_dim)]


def load_fold_latents(
    latent_root: str | Path,
    fold_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Load one fold/split of transformer latents and their matching dates."""

    fold_dir = Path(latent_root) / fold_name
    latents_path = fold_dir / f"latents_{split_name}.npy"
    dates_path = fold_dir / f"dates_{split_name}.npy"
    if not latents_path.exists() or not dates_path.exists():
        raise FileNotFoundError(
            f"Missing latent files for {fold_name} {split_name}: "
            f"{latents_path}, {dates_path}"
        )

    latents = np.load(latents_path)
    dates = np.load(dates_path, allow_pickle=True)
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D latents at {latents_path}, got {latents.shape}.")
    if len(dates) != len(latents):
        raise ValueError(
            f"Date/latent length mismatch for {fold_name} {split_name}: "
            f"{len(dates)} dates vs {len(latents)} rows."
        )

    latent_df = pd.DataFrame(latents, columns=_latent_column_names(latents.shape[1]))
    latent_df.insert(0, "date", pd.to_datetime(dates, errors="coerce").normalize())
    return latent_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_fold_regimes(
    regime_root: str | Path,
    fold_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Load one fold/split of GHMM regime labels and posterior probabilities."""

    regime_path = Path(regime_root) / fold_name / f"spy_{split_name}_labeled.csv"
    if not regime_path.exists():
        raise FileNotFoundError(f"Missing regime file for {fold_name} {split_name}: {regime_path}")

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


def prepare_finrl_dataframe(
    split_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Convert a split from `src.spy.load_data` format into FinRL stock env format.
    """
    # Validate all required features before any training logic starts.
    required_columns = OHLCV_COLUMNS + INDICATOR_COLUMNS
    missing_columns = [col for col in required_columns if col not in split_df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required market feature columns for FinRL baseline: {missing_columns}"
        )

    latent_columns = [col for col in latent_df.columns if col.startswith("latent_")]
    regime_columns = [
        col for col in regime_df.columns if col == "regime_label" or col.startswith("regime_prob_")
    ]
    if not latent_columns:
        raise ValueError("No latent columns were loaded.")
    if not regime_columns:
        raise ValueError("No regime columns were loaded.")

    # Original data is Date-indexed; FinRL expects a regular "date" column.
    reset_df = split_df.reset_index()
    if "Date" not in reset_df.columns:
        raise KeyError("Expected `Date` column after reset_index().")

    # Normalize naming to FinRL conventions.
    finrl_df = reset_df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    finrl_df["date"] = pd.to_datetime(finrl_df["date"], errors="coerce").dt.normalize()
    finrl_df = finrl_df.dropna(subset=["date"])

    latent_features = latent_df[["date", *latent_columns]].copy()
    latent_features["date"] = pd.to_datetime(
        latent_features["date"], errors="coerce"
    ).dt.normalize()

    regime_features = regime_df[["date", *regime_columns]].copy()
    regime_features["date"] = pd.to_datetime(
        regime_features["date"], errors="coerce"
    ).dt.normalize()

    # The latent date files already omit the first 19 trading days of each
    # fold/split, so this inner join performs the requested per-fold skip.
    finrl_df = finrl_df.merge(latent_features, on="date", how="inner", validate="one_to_one")
    finrl_df = finrl_df.merge(regime_features, on="date", how="inner", validate="one_to_one")

    # FinRL expects a ticker column even for single-asset training.
    finrl_df["tic"] = ticker

    selected_columns = [
        "date",
        "tic",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "rolling_vol_20",
        "ma_10",
        "ma_20",
        "ma_50",
        *latent_columns,
        *regime_columns,
    ]
    # Keep only columns that will become part of the environment state.
    finrl_df = finrl_df[selected_columns].copy()

    # Drop warmup/gap rows so no indicator NaNs leak into the RL state.
    feature_subset = [col for col in selected_columns if col not in {"date", "tic"}]
    finrl_df = finrl_df.dropna(subset=feature_subset)

    if finrl_df.empty:
        raise ValueError("Split becomes empty after dropping rows with NaN features.")

    # FinRL's downstream utilities typically expect date strings.
    finrl_df["date"] = finrl_df["date"].dt.strftime("%Y-%m-%d")
    finrl_df = finrl_df.sort_values(["date", "tic"]).reset_index(drop=True)
    # StockTradingEnv expects integer index grouped by trading day.
    finrl_df.index = finrl_df["date"].factorize()[0]
    return finrl_df


def _load_rl_dependencies():
    """
    Lazy-import RL packages so module import does not fail unless training runs.
    """

    def _load_stock_trading_env_fallback():
        # FinRL package import can fail if optional broker deps are missing.
        search_roots = [Path(path) for path in site.getsitepackages()]
        user_site = site.getusersitepackages()
        if user_site:
            search_roots.append(Path(user_site))

        for root in search_roots:
            env_path = root / "finrl" / "meta" / "env_stock_trading" / "env_stocktrading.py"
            if not env_path.exists():
                continue

            module_spec = importlib.util.spec_from_file_location(
                "finrl_env_stocktrading_fallback",
                env_path,
            )
            if module_spec is None or module_spec.loader is None:
                continue

            fallback_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(fallback_module)
            stock_env_cls = getattr(fallback_module, "StockTradingEnv", None)
            if stock_env_cls is not None:
                return stock_env_cls

        raise ImportError(
            "Could not locate FinRL env_stocktrading.py in site-packages."
        )

    try:
        finrl_env_module = importlib.import_module(
            "finrl.meta.env_stock_trading.env_stocktrading"
        )
        StockTradingEnv = getattr(finrl_env_module, "StockTradingEnv")
    except Exception:
        try:
            StockTradingEnv = _load_stock_trading_env_fallback()
        except Exception as fallback_exc:
            raise ImportError(
                "Could not import FinRL StockTradingEnv. Install FinRL and ensure "
                "optional broker deps do not block package import."
            ) from fallback_exc

    try:
        sb3_module = importlib.import_module("stable_baselines3")
        DQN = getattr(sb3_module, "DQN")
    except ImportError as exc:
        raise ImportError(
            "Could not import stable_baselines3.DQN. Install stable-baselines3."
        ) from exc

    try:
        gym = importlib.import_module("gymnasium")
    except ImportError:
        try:
            gym = importlib.import_module("gym")
        except ImportError as exc:
            raise ImportError("Could not import gym/gymnasium required by FinRL.") from exc

    return StockTradingEnv, DQN, gym


def _make_env_kwargs(config: RLBaselineConfig) -> dict:
    """Build FinRL environment kwargs from baseline config."""

    # Single-asset baseline.
    stock_dim = 1
    # FinRL state layout: [cash] + [prices] + [shares] + [tech indicators].
    state_space = 1 + 2 * stock_dim + len(FINRL_TECH_COLUMNS) * stock_dim
    return {
        "hmax": config.hmax,
        "initial_amount": config.initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [config.transaction_cost_pct] * stock_dim,
        "sell_cost_pct": [config.transaction_cost_pct] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": FINRL_TECH_COLUMNS,
        "action_space": stock_dim,
        "reward_scaling": config.reward_scaling,
    }


def _build_discrete_env(
    stock_trading_env_cls,
    gym_module,
    split_df: pd.DataFrame,
    env_kwargs: dict,
    action_grid: tuple[float, ...],
    downside_penalty_alpha: float = 0.3,
    trade_penalty_beta: float = 1e-4,
):
    """
    Wrap continuous FinRL env with a discrete action interface for DQN.

    Grid values are portfolio-weight deltas in [-1, 1]. The wrapper translates
    each delta into the equivalent shares-to-trade for FinRL, replaces FinRL's
    reward with the custom shaped reward, and appends three portfolio diagnostics
    to the observation: current_weight, step_return (%), fractional_drawdown.
    """

    class _DiscreteSingleAssetActionWrapper(gym_module.ActionWrapper):
        def __init__(
            self,
            env,
            grid: tuple[float, ...],
            downside_alpha: float,
            trade_beta: float,
        ):
            super().__init__(env)
            self._grid = np.asarray(grid, dtype=np.float32)
            self.action_space = gym_module.spaces.Discrete(len(self._grid))
            self._downside_alpha = float(downside_alpha)
            self._trade_beta = float(trade_beta)
            self._previous_account_value = 0.0
            self._running_peak_account_value = 0.0
            self.observation_space = self._build_augmented_observation_space()
            self.discrete_action_history: list[int] = []
            self.continuous_action_history: list[float] = []

        def _build_augmented_observation_space(self):
            base_space = getattr(self.env, "observation_space", None)
            dtype = np.float32
            lower_bound = np.finfo(dtype).min
            upper_bound = np.finfo(dtype).max

            if isinstance(base_space, gym_module.spaces.Box):
                base_low = np.asarray(base_space.low, dtype=dtype).reshape(-1)
                base_high = np.asarray(base_space.high, dtype=dtype).reshape(-1)
            else:
                if base_space is not None and getattr(base_space, "shape", None):
                    base_dim = int(np.prod(base_space.shape))
                else:
                    base_dim = int(len(np.asarray(getattr(self.env, "state", []), dtype=dtype)))
                base_low = np.full(base_dim, lower_bound, dtype=dtype)
                base_high = np.full(base_dim, upper_bound, dtype=dtype)

            # Extra dims: weight âˆˆ [0,1], step_return âˆˆ [-1,âˆž), frac_drawdown âˆˆ [0,1]
            extra_low = np.asarray([0.0, -1.0, 0.0], dtype=dtype)
            extra_high = np.asarray([1.0, upper_bound, 1.0], dtype=dtype)
            return gym_module.spaces.Box(
                low=np.concatenate([base_low, extra_low]),
                high=np.concatenate([base_high, extra_high]),
                dtype=dtype,
            )

        def _extract_portfolio_state(self) -> tuple[float, float, float, float]:
            state = np.asarray(getattr(self.env, "state", []), dtype=float).reshape(-1)
            if state.size < 3:
                return 0.0, 0.0, 0.0, 0.0
            cash = float(state[0]) if np.isfinite(state[0]) else 0.0
            price = float(state[1]) if np.isfinite(state[1]) else 0.0
            shares = float(state[2]) if np.isfinite(state[2]) else 0.0
            account_value = cash + shares * price
            return cash, price, shares, float(account_value) if np.isfinite(account_value) else 0.0

        def _current_account_value(self) -> float:
            asset_memory = getattr(self.env, "asset_memory", None)
            if isinstance(asset_memory, list) and asset_memory:
                latest = float(asset_memory[-1])
                if np.isfinite(latest):
                    return latest
            _, _, _, account_value = self._extract_portfolio_state()
            return account_value

        def _current_weight(self) -> float:
            _, price, shares, account_value = self._extract_portfolio_state()
            if account_value <= 0.0:
                return 0.0
            weight = (shares * price) / account_value
            return float(np.clip(weight, 0.0, 1.0)) if np.isfinite(weight) else 0.0

        def action(self, action):
            # DQN emits an integer action ID; grid values are weight deltas.
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self._grid) - 1))
            weight_delta = float(self._grid[idx])
            self.discrete_action_history.append(idx)
            self.continuous_action_history.append(weight_delta)

            # Convert weight delta â†’ shares-to-trade for FinRL's action API.
            _, price, shares, account_value = self._extract_portfolio_state()
            if account_value <= 0.0 or price <= 0.0:
                return np.asarray([0.0], dtype=np.float32)
            current_weight = (shares * price) / account_value
            target_weight = float(np.clip(current_weight + weight_delta, 0.0, 1.0))
            target_shares = (target_weight * account_value) / price
            delta_shares = target_shares - shares
            hmax = float(getattr(self.env, "hmax", 1))
            finrl_action = float(np.clip(delta_shares / hmax, -1.0, 1.0))
            return np.asarray([finrl_action], dtype=np.float32)

        def _trade_executed(self) -> bool:
            # Based solely on what the env actually transacted; never the intended action.
            actions_memory = getattr(self.env, "actions_memory", None)
            if isinstance(actions_memory, list) and actions_memory:
                latest_trade = np.asarray(actions_memory[-1]).reshape(-1)
                if latest_trade.size:
                    return bool(np.any(np.abs(latest_trade) > 1e-12))
            return False

        def _augment_observation(self, observation, step_return: float, drawdown_frac: float):
            obs_array = np.asarray(observation, dtype=np.float32).reshape(-1)
            extra = np.asarray(
                [self._current_weight(), step_return, max(0.0, drawdown_frac)],
                dtype=np.float32,
            )
            return np.concatenate([obs_array, extra]).astype(np.float32, copy=False)

        def reset(self, **kwargs):
            reset_output = self.env.reset(**kwargs)
            self.discrete_action_history.clear()
            self.continuous_action_history.clear()
            current_account_value = self._current_account_value()
            if current_account_value <= 0.0:
                current_account_value = float(getattr(self.env, "initial_amount", 0.0))
            self._previous_account_value = current_account_value
            self._running_peak_account_value = current_account_value
            if isinstance(reset_output, tuple):
                obs, info = reset_output
                return self._augment_observation(obs, step_return=0.0, drawdown_frac=0.0), info
            return self._augment_observation(reset_output, step_return=0.0, drawdown_frac=0.0)

        def step(self, action):
            previous_account_value = float(self._previous_account_value)
            step_output = super().step(action)
            current_account_value = self._current_account_value()
            if not np.isfinite(current_account_value):
                current_account_value = previous_account_value

            step_return = (
                current_account_value / previous_account_value - 1.0
                if previous_account_value > 0.0
                else 0.0
            )
            downside_penalty = self._downside_alpha * max(0.0, -step_return) ** 2
            custom_reward = step_return - downside_penalty
            if self._trade_executed():
                custom_reward -= self._trade_beta

            self._running_peak_account_value = max(
                self._running_peak_account_value, current_account_value
            )
            drawdown_frac = (
                (self._running_peak_account_value - current_account_value)
                / self._running_peak_account_value
                if self._running_peak_account_value > 0.0
                else 0.0
            )
            self._previous_account_value = current_account_value

            if len(step_output) == 5:
                obs, _reward, terminated, truncated, info = step_output
                aug_obs = self._augment_observation(obs, step_return=step_return, drawdown_frac=drawdown_frac)
                return aug_obs, float(custom_reward), terminated, truncated, info
            if len(step_output) == 4:
                obs, _reward, done, info = step_output
                aug_obs = self._augment_observation(obs, step_return=step_return, drawdown_frac=drawdown_frac)
                return aug_obs, float(custom_reward), done, info
            raise RuntimeError(f"Unexpected env.step output length: {len(step_output)}")

    base_env = stock_trading_env_cls(df=split_df, **env_kwargs)
    wrapped_env = _DiscreteSingleAssetActionWrapper(
        base_env,
        action_grid,
        downside_alpha=downside_penalty_alpha,
        trade_beta=trade_penalty_beta,
    )
    return wrapped_env, base_env


def _reset_env(env):
    """Compatibility helper for gym (obs) and gymnasium ((obs, info)) reset API."""

    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _step_env(env, action):
    """Compatibility helper for 4-tuple gym and 5-tuple gymnasium step outputs."""

    step_output = env.step(action)
    if len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        done = bool(terminated or truncated)
        return obs, reward, done, info
    if len(step_output) == 4:
        obs, reward, done, info = step_output
        return obs, reward, bool(done), info
    raise RuntimeError(f"Unexpected env.step output length: {len(step_output)}")


def train_fold_dqn(train_df: pd.DataFrame, config: RLBaselineConfig):
    """Train one DQN model on one fold's train split."""

    StockTradingEnv, DQN, gym = _load_rl_dependencies()
    env_kwargs = _make_env_kwargs(config)
    train_env, _ = _build_discrete_env(
        stock_trading_env_cls=StockTradingEnv,
        gym_module=gym,
        split_df=train_df,
        env_kwargs=env_kwargs,
        action_grid=config.action_grid,
        downside_penalty_alpha=config.downside_penalty_alpha,
        trade_penalty_beta=config.trade_penalty_beta,
    )

    # DQN is appropriate only after wrapping env action space as discrete.
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.dqn_learning_rate,
        batch_size=config.dqn_batch_size,
        buffer_size=config.dqn_buffer_size,
        learning_starts=config.dqn_learning_starts,
        train_freq=config.dqn_train_freq,
        target_update_interval=config.dqn_target_update_interval,
        exploration_fraction=config.dqn_exploration_fraction,
        exploration_final_eps=config.dqn_exploration_final_eps,
        seed=config.seed,
        verbose=0,
    )
    # Train in-place and return the fitted policy for val/test evaluation.
    model.learn(total_timesteps=config.train_timesteps, progress_bar=False)
    return model, env_kwargs, StockTradingEnv, gym


def _extract_account_value_frame(
    base_env,
    split_df: pd.DataFrame,
    initial_amount: float,
) -> pd.DataFrame:
    """
    Convert env memory buffers into a clean date/account-value frame.
    """

    date_memory = getattr(base_env, "date_memory", None)
    asset_memory = getattr(base_env, "asset_memory", None)

    if isinstance(date_memory, list) and isinstance(asset_memory, list):
        usable_len = min(len(date_memory), len(asset_memory))
        if usable_len > 0:
            account_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(date_memory[:usable_len], errors="coerce"),
                    "account_value": pd.to_numeric(asset_memory[:usable_len], errors="coerce"),
                }
            )
            account_df = account_df.dropna(subset=["date", "account_value"])
            account_df = account_df.drop_duplicates(subset=["date"], keep="last")
            account_df = account_df.sort_values("date").reset_index(drop=True)
            if not account_df.empty:
                return account_df

    # Fallback keeps evaluation robust even if env internals change.
    fallback_dates = pd.to_datetime(split_df["date"], errors="coerce")
    fallback_dates = fallback_dates.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if fallback_dates.empty:
        raise RuntimeError("Could not build account-value frame; no valid dates in split.")

    fallback_df = pd.DataFrame({"date": fallback_dates, "account_value": initial_amount})
    return fallback_df


def evaluate_dqn_on_split(
    model,
    split_df: pd.DataFrame,
    env_kwargs: dict,
    stock_trading_env_cls,
    gym_module,
    action_grid: tuple[float, ...],
    initial_amount: float,
    downside_penalty_alpha: float = 1.0,
    trade_penalty_beta: float = 1e-3,
    initial: bool = True,
    previous_state: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Run deterministic policy rollout on one split and export trajectories."""

    runtime_env_kwargs = dict(env_kwargs)
    runtime_env_kwargs["initial"] = bool(initial)
    if initial:
        runtime_env_kwargs["previous_state"] = []
    else:
        if previous_state is None:
            raise ValueError("previous_state must be provided when initial=False.")
        runtime_env_kwargs["previous_state"] = list(previous_state)

    eval_env, base_env = _build_discrete_env(
        stock_trading_env_cls=stock_trading_env_cls,
        gym_module=gym_module,
        split_df=split_df,
        env_kwargs=runtime_env_kwargs,
        action_grid=action_grid,
        downside_penalty_alpha=downside_penalty_alpha,
        trade_penalty_beta=trade_penalty_beta,
    )

    # Deterministic policy for reproducible validation/test evaluation.
    obs = _reset_env(eval_env)
    done = False
    # Guard against accidental infinite loops from environment API mismatches.
    max_steps = split_df["date"].nunique() + 5
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_idx = int(np.asarray(action).reshape(-1)[0])
        obs, _reward, done, _info = _step_env(eval_env, action_idx)
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError("Evaluation exceeded expected number of steps; stopping.")

    account_value_df = _extract_account_value_frame(
        base_env=base_env,
        split_df=split_df,
        initial_amount=initial_amount,
    )
    account_value_df["date"] = pd.to_datetime(account_value_df["date"], errors="coerce")
    account_value_df = account_value_df.dropna(subset=["date", "account_value"]).reset_index(drop=True)
    if account_value_df.empty:
        raise RuntimeError("Evaluation produced an empty account-value trajectory.")

    # Ensure independent (non-stitched) trajectories start at configured capital.
    first_date = account_value_df.loc[0, "date"]
    first_value = float(account_value_df.loc[0, "account_value"])
    if initial and (not np.isclose(first_value, initial_amount, rtol=1e-6, atol=1e-6)):
        seed_date = pd.to_datetime(split_df["date"].iloc[0], errors="coerce")
        if pd.isna(seed_date):
            seed_date = first_date
        seed_row = pd.DataFrame({"date": [seed_date], "account_value": [initial_amount]})
        account_value_df = pd.concat([seed_row, account_value_df], ignore_index=True)
        account_value_df = (
            account_value_df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="first")
            .reset_index(drop=True)
        )

    # Action log is aligned to post-initial dates (first row is initial state).
    actions = getattr(eval_env, "discrete_action_history", [])
    action_values = getattr(eval_env, "continuous_action_history", [])
    action_dates = account_value_df["date"].iloc[1 : 1 + len(actions)].reset_index(drop=True)
    executed_actions = getattr(base_env, "actions_memory", [])
    executed_shares: list[float] = []
    for action in executed_actions:
        action_arr = np.asarray(action).reshape(-1)
        executed_shares.append(float(action_arr[0]) if action_arr.size else 0.0)

    unique_dates = (
        pd.to_datetime(split_df["date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    trade_dates = unique_dates.iloc[: len(executed_shares)].reset_index(drop=True)
    price_by_date = (
        split_df[["date", "close"]]
        .assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce"))
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .set_index("date")["close"]
        .astype(float)
    )
    trade_prices = (
        price_by_date.reindex(trade_dates)
        .ffill()
        .bfill()
        .fillna(0.0)
        .reset_index(drop=True)
    )

    initial_shares = 0.0
    if (not initial) and previous_state is not None and len(previous_state) >= 3:
        initial_shares = float(previous_state[2])

    account_values = (
        pd.to_numeric(account_value_df["account_value"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        .reset_index(drop=True)
    )
    action_account_values = account_values.iloc[1 : 1 + len(actions)].reset_index(drop=True)
    previous_account_values = account_values.iloc[: len(actions)].reset_index(drop=True)

    aligned_len = min(
        len(action_dates),
        len(executed_shares),
        len(trade_prices),
        len(action_account_values),
        len(previous_account_values),
    )
    action_dates = action_dates.iloc[:aligned_len].reset_index(drop=True)
    discrete_actions = actions[:aligned_len]
    continuous_actions = action_values[:aligned_len]
    executed_shares = executed_shares[:aligned_len]
    trade_prices = trade_prices.iloc[:aligned_len].reset_index(drop=True)
    action_account_values = action_account_values.iloc[:aligned_len].reset_index(drop=True)
    previous_account_values = previous_account_values.iloc[:aligned_len].reset_index(drop=True)

    position_after_trade = (np.cumsum(executed_shares) + initial_shares).tolist()
    step_pnl = (
        action_account_values.to_numpy(dtype=float)
        - previous_account_values.to_numpy(dtype=float)
    )
    position_market_value = np.asarray(position_after_trade, dtype=float) * trade_prices.to_numpy(dtype=float)
    account_value_arr = action_account_values.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(account_value_arr > 0.0, position_market_value / account_value_arr, 0.0)
    weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    weight = np.clip(weight, 0.0, 1.0)
    running_peak = np.maximum.accumulate(account_value_arr)
    drawdown = running_peak - account_value_arr

    actions_df = pd.DataFrame(
        {
            "date": action_dates,
            "action_index": discrete_actions,
            "action_value": continuous_actions,
            "executed_shares": executed_shares,
            "trade_price": trade_prices,
            "trade_notional": np.abs(np.asarray(executed_shares)) * trade_prices.to_numpy(),
            "position_after_trade": position_after_trade,
            "weight": weight,
            "pnl": step_pnl,
            "drawdown": drawdown,
        }
    )
    terminal_state = list(getattr(base_env, "state", []))
    return account_value_df, actions_df, terminal_state


def compute_performance_metrics(
    account_value_df: pd.DataFrame,
    actions_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute common backtest metrics from account-value time series."""

    if account_value_df.empty:
        raise ValueError("Cannot compute metrics from an empty account-value frame.")

    values = account_value_df["account_value"].astype(float)
    if len(values) < 2:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "calmar_ratio": np.nan,
            "max_drawdown": np.nan,
            "turnover": np.nan,
            "annualized_turnover": np.nan,
            "average_holding_time_days": np.nan,
        }

    # Daily return stream is used for volatility and risk-adjusted metrics.
    daily_returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)

    # Annualization assumes 252 trading days.
    periods = len(values) - 1
    annualized_return = float((1.0 + total_return) ** (252.0 / periods) - 1.0)

    if daily_returns.empty:
        annualized_volatility = np.nan
        sharpe_ratio = np.nan
        sortino_ratio = np.nan
    else:
        annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(252.0))
        sharpe_ratio = (
            float(annualized_return / annualized_volatility)
            if annualized_volatility > 0
            else np.nan
        )

        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = (
            float(downside_returns.std(ddof=0) * np.sqrt(252.0))
            if not downside_returns.empty
            else np.nan
        )
        sortino_ratio = (
            float(annualized_return / downside_volatility)
            if downside_volatility and downside_volatility > 0
            else np.nan
        )

    # Drawdown is measured from running equity peaks.
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    max_drawdown = float(drawdown.min())
    calmar_ratio = (
        float(annualized_return / abs(max_drawdown))
        if max_drawdown < 0
        else np.nan
    )

    turnover = np.nan
    annualized_turnover = np.nan
    average_holding_time_days = np.nan
    if actions_df is not None and not actions_df.empty:
        required_action_cols = {"trade_notional", "position_after_trade"}
        if required_action_cols.issubset(actions_df.columns):
            total_traded_notional = float(
                pd.to_numeric(actions_df["trade_notional"], errors="coerce").fillna(0.0).sum()
            )
            avg_account_value = float(values.mean())
            turnover = (
                float(total_traded_notional / avg_account_value)
                if avg_account_value > 0
                else np.nan
            )
            annualized_turnover = (
                float(turnover * (252.0 / periods))
                if (periods > 0) and np.isfinite(turnover)
                else np.nan
            )

            positions = (
                pd.to_numeric(actions_df["position_after_trade"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            if positions.size > 0:
                holding_mask = positions > 0.0
                runs: list[int] = []
                run_length = 0
                for is_holding in holding_mask:
                    if is_holding:
                        run_length += 1
                    elif run_length > 0:
                        runs.append(run_length)
                        run_length = 0
                if run_length > 0:
                    runs.append(run_length)
                average_holding_time_days = float(np.mean(runs)) if runs else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "annualized_turnover": annualized_turnover,
        "average_holding_time_days": average_holding_time_days,
    }


def save_portfolio_growth_plot(portfolio_curves_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a portfolio growth line chart similar to nonregime_baseline output.
    """

    if portfolio_curves_df.empty:
        raise ValueError("Cannot plot portfolio growth from an empty curve DataFrame.")

    # Imported lazily to keep training imports minimal when plotting is not needed.
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

    ax.set_title("SPY Portfolio Growth (DRL + Latent + Regime DQN)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_stitched_classic_baselines(
    full_df: pd.DataFrame,
    stitched_dates: pd.Series,
    initial_amount: float,
    momentum_lookback: int = 126,
) -> pd.DataFrame:
    """
    Build buy-and-hold and momentum equity curves over stitched OOS dates.
    """

    close = full_df["Close"].astype(float).copy()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    target_dates = pd.to_datetime(stitched_dates, errors="coerce")
    target_dates = (
        pd.Series(target_dates)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if target_dates.empty:
        raise ValueError("No stitched dates provided for classic baseline construction.")

    prices = close.reindex(target_dates).dropna()
    if prices.empty:
        raise ValueError("No close prices found on stitched dates.")

    # Buy-and-hold: fully invested from first stitched date.
    bh_shares = float(initial_amount / prices.iloc[0])
    bh_equity = bh_shares * prices

    # Momentum: all-in or all-cash based on lagged 126-day momentum signal.
    momentum = close.pct_change(momentum_lookback)
    signal = (momentum > 0).astype(int).shift(1).fillna(0).astype(int)
    signal_on_dates = signal.reindex(prices.index).fillna(0).astype(int)

    cash = float(initial_amount)
    shares = 0.0
    mom_equity: list[float] = []
    for dt, price in prices.items():
        day_signal = int(signal_on_dates.loc[dt])
        if day_signal == 1 and shares == 0.0 and cash > 0.0:
            shares = float(cash / price)
            cash = 0.0
        elif day_signal == 0 and shares > 0.0:
            cash = float(shares * price)
            shares = 0.0
        mom_equity.append(float(cash + shares * price))

    bh_df = pd.DataFrame(
        {
            "date": prices.index,
            "strategy": "buy_hold",
            "account_value": bh_equity.values,
        }
    )
    mom_df = pd.DataFrame(
        {
            "date": prices.index,
            "strategy": f"momentum_{momentum_lookback}",
            "account_value": mom_equity,
        }
    )
    return pd.concat([bh_df, mom_df], ignore_index=True)


def save_strategy_comparison_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save stitched equity comparison plot for RL vs classic baselines.
    """

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
    """
    Save a grid of subplots comparing DRL-Latent-Regime, Buy-and-Hold, and Momentum on each
    fold's test period independently â€” all three strategies start at the same initial
    capital on the same first test day, making fold-level performance directly comparable.
    """

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
        "drl_latent_regime": {"color": "tab:blue", "label": "DRL-Latent-Regime"},
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
        "Per-Fold Test Period: DRL-Latent-Regime vs Buy & Hold vs Momentum (each fold starts at $10k)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_walkforward_drl_latent_regime_baseline(
    config: RLBaselineConfig,
    selected_folds: str | None = None,
    max_folds: int | None = None,
) -> pd.DataFrame:
    """
    End-to-end walk-forward loop:
    1) split data by fold
    2) train on train split
    3) evaluate on val/test
    4) write fold artifacts and aggregate metrics
    """

    global FINRL_TECH_COLUMNS

    variant_root = Path(config.transformer_root) / config.variant
    if not variant_root.exists():
        raise FileNotFoundError(
            f"Missing transformer variant directory: {variant_root}. "
            "Expected one of: gating, no_gating, no_sentiment."
        )

    full_df = load_data(path=config.input_path)
    folds = _resolve_folds(selected_folds=selected_folds, max_folds=max_folds)

    output_root = Path(config.output_dir) / config.variant
    output_root.mkdir(parents=True, exist_ok=True)

    # Each row corresponds to one (fold, split) result.
    metrics_rows: list[dict[str, float | int | str]] = []
    portfolio_curves: list[pd.DataFrame] = []
    stitched_test_curves: list[pd.DataFrame] = []
    stitched_test_actions: list[pd.DataFrame] = []
    stitched_previous_state: list[float] | None = None
    per_fold_comparisons: list[pd.DataFrame] = []
    per_fold_test_actions: dict[str, pd.DataFrame] = {}
    for fold in folds:
        print(
            f"Running {fold['name']}: "
            f"train {fold['train_start']} to {fold['train_end']}, "
            f"val {fold['val_start']} to {fold['val_end']}, "
            f"test {fold['test_start']} to {fold['test_end']}"
        )

        # Fold boundaries are sourced from configs/walkforward_folds.py.
        train_raw, val_raw, test_raw = create_walk_forward_split(
            full_df,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )

        # Convert raw split data into FinRL schema, adding transformer latents
        # and regimes. Latent date files skip the first 19 days of each split.
        train_latents = load_fold_latents(variant_root, fold["name"], "train")
        val_latents = load_fold_latents(variant_root, fold["name"], "val")
        test_latents = load_fold_latents(variant_root, fold["name"], "test")
        train_regimes = load_fold_regimes(config.regime_root, fold["name"], "train")
        val_regimes = load_fold_regimes(config.regime_root, fold["name"], "val")
        test_regimes = load_fold_regimes(config.regime_root, fold["name"], "test")

        latent_columns = [col for col in train_latents.columns if col.startswith("latent_")]
        regime_columns = [
            col
            for col in train_regimes.columns
            if col == "regime_label" or col.startswith("regime_prob_")
        ]
        FINRL_TECH_COLUMNS = [*BASE_FINRL_TECH_COLUMNS, *latent_columns, *regime_columns]

        train_finrl = prepare_finrl_dataframe(
            train_raw,
            latent_df=train_latents,
            regime_df=train_regimes,
            ticker=config.ticker,
        )
        val_finrl = prepare_finrl_dataframe(
            val_raw,
            latent_df=val_latents,
            regime_df=val_regimes,
            ticker=config.ticker,
        )
        test_finrl = prepare_finrl_dataframe(
            test_raw,
            latent_df=test_latents,
            regime_df=test_regimes,
            ticker=config.ticker,
        )

        # Train one fresh model per fold (no leakage across folds).
        model, env_kwargs, stock_trading_env_cls, gym_module = train_fold_dqn(
            train_df=train_finrl,
            config=config,
        )

        fold_dir = output_root / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate only on val/test (train is used solely for fitting policy).
        fold_test_account_df: pd.DataFrame | None = None
        split_map = {"val": val_finrl, "test": test_finrl}
        for split_name, split_df in split_map.items():
            account_df, actions_df, _ = evaluate_dqn_on_split(
                model=model,
                split_df=split_df,
                env_kwargs=env_kwargs,
                stock_trading_env_cls=stock_trading_env_cls,
                gym_module=gym_module,
                action_grid=config.action_grid,
                initial_amount=config.initial_amount,
                downside_penalty_alpha=config.downside_penalty_alpha,
                trade_penalty_beta=config.trade_penalty_beta,
            )
            metrics = compute_performance_metrics(account_df, actions_df=actions_df)

            # Persist trajectories for plotting and post-hoc diagnostics.
            account_out = fold_dir / f"{split_name}_account_value.csv"
            actions_out = fold_dir / f"{split_name}_actions.csv"
            account_df.to_csv(account_out, index=False)
            actions_df.to_csv(actions_out, index=False)
            portfolio_curves.append(
                account_df.assign(fold=fold["name"], split=split_name)[
                    ["date", "account_value", "fold", "split"]
                ]
            )

            metrics_rows.append(
                {
                    "fold": fold["name"],
                    "split": split_name,
                    "train_rows": len(train_finrl),
                    "eval_rows": len(split_df),
                    "account_points": len(account_df),
                    **metrics,
                }
            )
            if split_name == "test":
                fold_test_account_df = account_df
                per_fold_test_actions[fold["name"]] = actions_df

        # Per-fold fair comparison: RL independent test vs B&H vs Momentum (all fresh $10k).
        if fold_test_account_df is not None:
            fold_test_dates = pd.to_datetime(fold_test_account_df["date"], errors="coerce").dropna()
            fold_classic_df = build_stitched_classic_baselines(
                full_df=full_df,
                stitched_dates=fold_test_dates,
                initial_amount=config.initial_amount,
                momentum_lookback=126,
            )
            fold_rl_curve = fold_test_account_df[["date", "account_value"]].assign(strategy="drl_latent_regime")
            fold_comparison = pd.concat([fold_rl_curve, fold_classic_df], ignore_index=True)
            fold_comparison["fold"] = fold["name"]
            per_fold_comparisons.append(fold_comparison)

        # Stitched out-of-sample test: carry terminal cash/shares across folds.
        stitched_account_df, stitched_actions_df, stitched_previous_state = evaluate_dqn_on_split(
            model=model,
            split_df=test_finrl,
            env_kwargs=env_kwargs,
            stock_trading_env_cls=stock_trading_env_cls,
            gym_module=gym_module,
            action_grid=config.action_grid,
            initial_amount=config.initial_amount,
            downside_penalty_alpha=config.downside_penalty_alpha,
            trade_penalty_beta=config.trade_penalty_beta,
            initial=stitched_previous_state is None,
            previous_state=stitched_previous_state,
        )
        stitched_account_out = fold_dir / "test_account_value_stitched.csv"
        stitched_actions_out = fold_dir / "test_actions_stitched.csv"
        stitched_account_df.to_csv(stitched_account_out, index=False)
        stitched_actions_df.to_csv(stitched_actions_out, index=False)
        stitched_test_curves.append(
            stitched_account_df.assign(fold=fold["name"])[["date", "account_value", "fold"]]
        )
        stitched_test_actions.append(
            stitched_actions_df.assign(fold=fold["name"])[
                [
                    "date",
                    "action_index",
                    "action_value",
                    "executed_shares",
                    "trade_price",
                    "trade_notional",
                    "position_after_trade",
                    "weight",
                    "pnl",
                    "drawdown",
                    "fold",
                ]
            ]
        )

    # Per-fold fair comparison: RL vs B&H vs Momentum, each fold starting fresh at $10k.
    if per_fold_comparisons:
        per_fold_df = pd.concat(per_fold_comparisons, ignore_index=True)
        per_fold_df["date"] = pd.to_datetime(per_fold_df["date"], errors="coerce")
        per_fold_df = (
            per_fold_df.dropna(subset=["date", "account_value", "strategy"])
            .sort_values(["fold", "strategy", "date"])
            .reset_index(drop=True)
        )
        per_fold_df.to_csv(output_root / "drl_latent_regime_per_fold_test_comparison.csv", index=False)
        save_per_fold_comparison_plot(
            per_fold_comparison_df=per_fold_df,
            output_path=output_root / "drl_latent_regime_per_fold_test_comparison.png",
        )
        per_fold_metrics_rows: list[dict[str, float | str]] = []
        for (fold_name, strategy_name), strat_df in per_fold_df.groupby(["fold", "strategy"]):
            strat_actions_df = per_fold_test_actions.get(fold_name) if strategy_name == "drl_latent_regime" else None
            strat_metrics = compute_performance_metrics(
                strat_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
                actions_df=strat_actions_df,
            )
            per_fold_metrics_rows.append({"fold": fold_name, "strategy": strategy_name, **strat_metrics})
        pd.DataFrame(per_fold_metrics_rows).to_csv(
            output_root / "drl_latent_regime_per_fold_test_metrics.csv", index=False
        )

    # Aggregate all fold/split metrics into one summary table.
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_root / "drl_latent_regime_fold_metrics.csv", index=False)
    if portfolio_curves:
        curves_df = pd.concat(portfolio_curves, ignore_index=True)
        curves_df.to_csv(output_root / "drl_latent_regime_portfolio_curves.csv", index=False)
        save_portfolio_growth_plot(
            portfolio_curves_df=curves_df,
            output_path=output_root / "drl_latent_regime_portfolio_growth.png",
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
        stitched_rl_df.to_csv(output_root / "drl_latent_regime_stitched_test_equity.csv", index=False)

        stitched_actions_df = pd.concat(stitched_test_actions, ignore_index=True)
        stitched_actions_df.to_csv(output_root / "drl_latent_regime_stitched_test_actions.csv", index=False)

        classic_df = build_stitched_classic_baselines(
            full_df=full_df,
            stitched_dates=stitched_rl_df["date"],
            initial_amount=config.initial_amount,
            momentum_lookback=126,
        )
        stitched_rl_long = stitched_rl_df[["date", "account_value"]].assign(
            strategy="drl_latent_regime_stitched_test"
        )
        comparison_df = pd.concat([stitched_rl_long, classic_df], ignore_index=True)
        comparison_df["date"] = pd.to_datetime(comparison_df["date"], errors="coerce")
        comparison_df = comparison_df.dropna(subset=["date", "account_value", "strategy"])
        comparison_df = comparison_df.sort_values(["date", "strategy"]).reset_index(drop=True)
        comparison_df.to_csv(output_root / "drl_latent_regime_stitched_comparison.csv", index=False)
        save_strategy_comparison_plot(
            comparison_df=comparison_df,
            output_path=output_root / "drl_latent_regime_stitched_comparison.png",
        )

        stitched_metrics_rows: list[dict[str, float | str]] = []
        for strategy_name, strategy_df in comparison_df.groupby("strategy"):
            strategy_actions_df = None
            if strategy_name == "drl_latent_regime_stitched_test":
                strategy_actions_df = stitched_actions_df
            strategy_metrics = compute_performance_metrics(
                strategy_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
                actions_df=strategy_actions_df,
            )
            stitched_metrics_rows.append({"strategy": strategy_name, **strategy_metrics})
        pd.DataFrame(stitched_metrics_rows).to_csv(
            output_root / "drl_latent_regime_stitched_comparison_metrics.csv",
            index=False,
        )

    # Save exact run config so results are reproducible.
    run_config = asdict(config)
    run_config["action_grid"] = list(config.action_grid)
    with open(output_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    return metrics_df


def parse_args() -> argparse.Namespace:
    """Define CLI arguments for running DRL + latent + regime experiments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a combined DRL + transformer latent + regime "
            "DQN model across walk-forward folds."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/spy_market_data.csv",
        help="Path to SPY market data CSV with engineered indicators.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/baselines/drl_latent_regime",
        help="Directory to store fold outputs and summary metrics.",
    )
    parser.add_argument(
        "--transformer-root",
        type=str,
        default="data/transformer_npy",
        help="Root containing variant directories: gating, no_gating, no_sentiment.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=("all", "gating", "no_gating", "no_sentiment"),
        help=(
            "Transformer latent variant to use. Default `all` runs gating, "
            "then no_gating, then no_sentiment."
        ),
    )
    parser.add_argument(
        "--regime-root",
        type=str,
        default="data/training",
        help="Root containing foldN/spy_{split}_labeled.csv regime files.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Ticker label used for FinRL input (single-asset baseline).",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=200_000,
        help="DQN training timesteps per fold.",
    )
    parser.add_argument(
        "--initial-amount",
        type=float,
        default=10_000.0,
        help="Initial portfolio value in USD.",
    )
    parser.add_argument(
        "--transaction-cost-pct",
        type=float,
        default=1e-3,
        help="Buy/sell transaction cost percentage per trade.",
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1e-4,
        help="Reward scaling factor passed to FinRL StockTradingEnv.",
    )
    parser.add_argument(
        "--hmax",
        type=int,
        default=100,
        help="Maximum shares FinRL can trade when the computed delta fills the full hmax budget.",
    )
    parser.add_argument(
        "--downside-penalty-alpha",
        type=float,
        default=1.0,
        help="Alpha in reward: R_t - alpha * max(0, -R_t)^2.",
    )
    parser.add_argument(
        "--trade-penalty-beta",
        type=float,
        default=1e-3,
        help="Beta penalty subtracted when any trade is executed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for DQN initialization.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Optional comma-separated fold names (e.g. fold1,fold2).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on number of folds to run from the selected list.",
    )
    parser.add_argument(
        "--action-grid",
        type=str,
        default="-1.0,-0.5,0.0,0.5,1.0",
        help=(
            "Comma-separated portfolio-weight deltas in [-1, 1] (must include 0 for hold). "
            "Each value is converted to the equivalent shares-to-trade at step time."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    variants = (
        ["gating", "no_gating", "no_sentiment"]
        if args.variant == "all"
        else [args.variant]
    )

    for variant in variants:
        print(f"Starting DRL + latent + regime training for variant: {variant}")
        config = RLBaselineConfig(
            input_path=args.input_path,
            transformer_root=args.transformer_root,
            variant=variant,
            regime_root=args.regime_root,
            output_dir=args.output_dir,
            ticker=args.ticker,
            train_timesteps=args.train_timesteps,
            initial_amount=args.initial_amount,
            transaction_cost_pct=args.transaction_cost_pct,
            reward_scaling=args.reward_scaling,
            hmax=args.hmax,
            downside_penalty_alpha=args.downside_penalty_alpha,
            trade_penalty_beta=args.trade_penalty_beta,
            seed=args.seed,
            action_grid=_parse_action_grid(args.action_grid),
        )
        metrics_df = run_walkforward_drl_latent_regime_baseline(
            config=config,
            selected_folds=args.folds,
            max_folds=args.max_folds,
        )
        print(f"DRL + latent + regime training complete for {variant}. Metrics:")
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()



