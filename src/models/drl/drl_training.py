"""
Market-State-Aware DRL using FinRL + DQN.

This model uses:
- SPY OHLCV
- Technical indicators
- Regime labels
- 32D latent state vector (from transformer)

Walk-forward training and evaluation.
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
from dataclasses import asdict, dataclass, field
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

# Allow running as `python src/baselines/rl_only_baseline.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.walkforward_folds import FOLDS
from spy.market_data_utils import create_walk_forward_split, load_data

LATENT_DIM = 32
LATENT_COLUMNS = [f"latent_{i}" for i in range(LATENT_DIM)]
REGIME_COLUMNS = ["regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"]

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

INDICATOR_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "rolling_vol_20",
    "ma_10",
    "ma_20",
    "ma_50",
    "weight",
    "PnL",
    "drawdown",
]

STATE_COLUMNS = [
    *OHLCV_COLUMNS,
    *INDICATOR_COLUMNS,
    *REGIME_COLUMNS,
    *LATENT_COLUMNS,
]

@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for DQN training."""

    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 500
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.02


@dataclass(frozen=True)
class DRLTrainingConfig:
    """Central config for market-state-aware DRL runs."""

    # Data / IO
    input_path: str = "data/spy_market_data.csv"
    output_dir: str = "data/drl_results"
    ticker: str = "SPY"

    # Training / trading
    train_timesteps: int = 50_000
    initial_amount: float = 10_000.0
    transaction_cost_pct: float = 1e-3
    reward_scaling: float = 1e-4
    hmax: int = 100
    seed: int = 42

    # Discrete action grid for DQN
    action_grid: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

    # Nested DQN hyperparameters
    dqn: DQNConfig = field(default_factory=DQNConfig)


def _parse_action_grid(raw: str) -> tuple[float, ...]:
    """Parse CLI action grid string and enforce valid discrete-action values."""

    # Example input: "-1.0,-0.5,0.0,0.5,1.0".
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise ValueError("Action grid needs at least 2 values (e.g. -1,0,1).")
    # Always keep an explicit hold action.
    if 0.0 not in values:
        raise ValueError("Action grid must include 0.0 (hold action).")
    # FinRL action semantics are in [-1, 1].
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


def prepare_finrl_dataframe(split_df: pd.DataFrame, ticker: str = "SPY") -> pd.DataFrame:
    """
    Convert a split into FinRL stock env format, including regime labels
    and 32D latent state vectors.
    """
    required_columns = [
        *OHLCV_COLUMNS,
        *INDICATOR_COLUMNS,
        *REGIME_COLUMNS,
        *LATENT_COLUMNS,
    ]
    missing_columns = [col for col in required_columns if col not in split_df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns for DRL training: {missing_columns}"
        )

    reset_df = split_df.reset_index()
    if "Date" not in reset_df.columns:
        raise KeyError("Expected `Date` column after reset_index().")

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
    finrl_df["tic"] = ticker

    selected_columns = ["date", "tic", "close", *STATE_COLUMNS]
    finrl_df = finrl_df[selected_columns].copy()

    finrl_df["date"] = pd.to_datetime(finrl_df["date"], errors="coerce")
    finrl_df = finrl_df.dropna(subset=["date"])

    feature_subset = [col for col in selected_columns if col not in {"date", "tic"}]
    finrl_df = finrl_df.dropna(subset=feature_subset)

    if finrl_df.empty:
        raise ValueError("Split becomes empty after dropping rows with NaN features.")

    finrl_df["date"] = finrl_df["date"].dt.strftime("%Y-%m-%d")
    finrl_df = finrl_df.sort_values(["date", "tic"]).reset_index(drop=True)
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


def _make_env_kwargs(config: DRLTrainingConfig) -> dict:
    """Build FinRL environment kwargs from config."""

    stock_dim = 1
    state_space = 1 + 2 * stock_dim + len(STATE_COLUMNS) * stock_dim

    return {
        "hmax": config.hmax,
        "initial_amount": config.initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [config.transaction_cost_pct] * stock_dim,
        "sell_cost_pct": [config.transaction_cost_pct] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": STATE_COLUMNS,
        "action_space": stock_dim,
        "reward_scaling": config.reward_scaling,
    }

def _build_discrete_env(
    stock_trading_env_cls,
    gym_module,
    split_df: pd.DataFrame,
    env_kwargs: dict,
    action_grid: tuple[float, ...],
):
    """
    Wrap continuous FinRL env with a discrete action interface for DQN.
    """

    class _DiscreteSingleAssetActionWrapper(gym_module.ActionWrapper):
        def __init__(self, env, grid: tuple[float, ...]):
            super().__init__(env)
            # Map integer action IDs to fixed continuous values in [-1, 1].
            self._grid = np.asarray(grid, dtype=np.float32)
            self.action_space = gym_module.spaces.Discrete(len(self._grid))
            # Keep history for fold diagnostics / CSV export.
            self.discrete_action_history: list[int] = []
            self.continuous_action_history: list[float] = []

        def action(self, action):
            # DQN emits an integer action ID.
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self._grid) - 1))
            continuous_value = float(self._grid[idx])
            self.discrete_action_history.append(idx)
            self.continuous_action_history.append(continuous_value)
            return np.asarray([continuous_value], dtype=np.float32)

    # Base FinRL env still handles portfolio accounting and reward logic.
    base_env = stock_trading_env_cls(df=split_df, **env_kwargs)
    wrapped_env = _DiscreteSingleAssetActionWrapper(base_env, action_grid)
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


def train_fold_dqn(train_df: pd.DataFrame, config: DRLTrainingConfig):
    """Train one DQN model on one fold's train split."""

    StockTradingEnv, DQN, gym = _load_rl_dependencies()
    env_kwargs = _make_env_kwargs(config)
    train_env, _ = _build_discrete_env(
        stock_trading_env_cls=StockTradingEnv,
        gym_module=gym,
        split_df=train_df,
        env_kwargs=env_kwargs,
        action_grid=config.action_grid,
    )

    dqn_cfg = config.dqn

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=dqn_cfg.learning_rate,
        batch_size=dqn_cfg.batch_size,
        buffer_size=dqn_cfg.buffer_size,
        learning_starts=dqn_cfg.learning_starts,
        train_freq=dqn_cfg.train_freq,
        target_update_interval=dqn_cfg.target_update_interval,
        exploration_fraction=dqn_cfg.exploration_fraction,
        exploration_final_eps=dqn_cfg.exploration_final_eps,
        seed=config.seed,
        verbose=0,
    )

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

    aligned_len = min(len(action_dates), len(executed_shares), len(trade_prices))
    action_dates = action_dates.iloc[:aligned_len].reset_index(drop=True)
    discrete_actions = actions[:aligned_len]
    continuous_actions = action_values[:aligned_len]
    executed_shares = executed_shares[:aligned_len]
    trade_prices = trade_prices.iloc[:aligned_len].reset_index(drop=True)
    position_after_trade = (np.cumsum(executed_shares) + initial_shares).tolist()
    actions_df = pd.DataFrame(
        {
            "date": action_dates,
            "action_index": discrete_actions,
            "action_value": continuous_actions,
            "executed_shares": executed_shares,
            "trade_price": trade_prices,
            "trade_notional": np.abs(np.asarray(executed_shares)) * trade_prices.to_numpy(),
            "position_after_trade": position_after_trade,
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

    ax.set_title("SPY Portfolio Growth (RL-Only DQN)")
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


def run_walkforward_drl_training(
    config: DRLTrainingConfig,
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

    full_df = load_data(path=config.input_path)
    folds = _resolve_folds(selected_folds=selected_folds, max_folds=max_folds)

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Each row corresponds to one (fold, split) result.
    metrics_rows: list[dict[str, float | int | str]] = []
    portfolio_curves: list[pd.DataFrame] = []
    stitched_test_curves: list[pd.DataFrame] = []
    stitched_test_actions: list[pd.DataFrame] = []
    stitched_previous_state: list[float] | None = None
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

        # Convert raw split data into FinRL schema and drop invalid feature rows.
        train_finrl = prepare_finrl_dataframe(train_raw, ticker=config.ticker)
        val_finrl = prepare_finrl_dataframe(val_raw, ticker=config.ticker)
        test_finrl = prepare_finrl_dataframe(test_raw, ticker=config.ticker)

        # Train one fresh model per fold (no leakage across folds).
        model, env_kwargs, stock_trading_env_cls, gym_module = train_fold_dqn(
            train_df=train_finrl,
            config=config,
        )

        fold_dir = output_root / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate only on val/test (train is used solely for fitting policy).
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

        # Stitched out-of-sample test: carry terminal cash/shares across folds.
        stitched_account_df, stitched_actions_df, stitched_previous_state = evaluate_dqn_on_split(
            model=model,
            split_df=test_finrl,
            env_kwargs=env_kwargs,
            stock_trading_env_cls=stock_trading_env_cls,
            gym_module=gym_module,
            action_grid=config.action_grid,
            initial_amount=config.initial_amount,
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
                    "fold",
                ]
            ]
        )

    # Aggregate all fold/split metrics into one summary table.
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_root / "rl_only_fold_metrics.csv", index=False)
    if portfolio_curves:
        curves_df = pd.concat(portfolio_curves, ignore_index=True)
        curves_df.to_csv(output_root / "rl_only_portfolio_curves.csv", index=False)
        save_portfolio_growth_plot(
            portfolio_curves_df=curves_df,
            output_path=output_root / "rl_only_portfolio_growth.png",
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
        stitched_rl_df.to_csv(output_root / "rl_only_stitched_test_equity.csv", index=False)

        stitched_actions_df = pd.concat(stitched_test_actions, ignore_index=True)
        stitched_actions_df.to_csv(output_root / "rl_only_stitched_test_actions.csv", index=False)

        classic_df = build_stitched_classic_baselines(
            full_df=full_df,
            stitched_dates=stitched_rl_df["date"],
            initial_amount=config.initial_amount,
            momentum_lookback=126,
        )
        stitched_rl_long = stitched_rl_df[["date", "account_value"]].assign(
            strategy="rl_stitched_test"
        )
        comparison_df = pd.concat([stitched_rl_long, classic_df], ignore_index=True)
        comparison_df["date"] = pd.to_datetime(comparison_df["date"], errors="coerce")
        comparison_df = comparison_df.dropna(subset=["date", "account_value", "strategy"])
        comparison_df = comparison_df.sort_values(["date", "strategy"]).reset_index(drop=True)
        comparison_df.to_csv(output_root / "rl_only_stitched_comparison.csv", index=False)
        save_strategy_comparison_plot(
            comparison_df=comparison_df,
            output_path=output_root / "rl_only_stitched_comparison.png",
        )

        stitched_metrics_rows: list[dict[str, float | str]] = []
        for strategy_name, strategy_df in comparison_df.groupby("strategy"):
            strategy_actions_df = None
            if strategy_name == "rl_stitched_test":
                strategy_actions_df = stitched_actions_df
            strategy_metrics = compute_performance_metrics(
                strategy_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
                actions_df=strategy_actions_df,
            )
            stitched_metrics_rows.append({"strategy": strategy_name, **strategy_metrics})
        pd.DataFrame(stitched_metrics_rows).to_csv(
            output_root / "rl_only_stitched_comparison_metrics.csv",
            index=False,
        )

    # Save exact run config so results are reproducible.
    run_config = asdict(config)
    run_config["action_grid"] = list(config.action_grid)
    with open(output_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    return metrics_df


def parse_args() -> argparse.Namespace:
    """Define CLI arguments for running DRL experiments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate an RL-only DQN baseline in FinRL across "
            "walk-forward folds using SPY OHLCV + technical indicators."
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
        default="data/baselines/rl_only",
        help="Directory to store fold outputs and summary metrics.",
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
        default=50_000,
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
        help="Maximum shares traded when continuous action magnitude is 1.0.",
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
            "Comma-separated discrete action grid in [-1, 1] mapped to FinRL "
            "continuous actions (must include 0)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()

    config = DRLTrainingConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        ticker=args.ticker,
        train_timesteps=args.train_timesteps,
        initial_amount=args.initial_amount,
        transaction_cost_pct=args.transaction_cost_pct,
        reward_scaling=args.reward_scaling,
        hmax=args.hmax,
        seed=args.seed,
        action_grid=_parse_action_grid(args.action_grid),
        dqn=DQNConfig(),
    )

    metrics_df = run_walkforward_drl_training(
        config=config,
        selected_folds=args.folds,
        max_folds=args.max_folds,
    )
    print("DRL training complete. Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
