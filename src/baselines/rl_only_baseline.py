"""
RL-only baseline using TDQN on SPY walk-forward folds.

Based on the position-based action space and precomputed lookahead reward
from: https://www.mdpi.com/2073-8994/18/1/112

Positions: H ∈ {-1=short, 0=flat, 1=long}.
Reward is precomputed from k-day lookahead during training only (offline label).
At eval time the trained policy runs deterministically on current observations only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure matplotlib has a writable cache path.
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

# Raw market columns expected from `spy_market_data.csv`.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
# Technical indicators requested for the RL-only baseline.
INDICATOR_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "rolling_vol_20",
    "ma_10",
    "ma_20",
    "ma_50",
]
# Observation feature columns sourced from the cleaned DataFrame.
ENV_FEATURE_COLUMNS = [
    "close",
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

# Detect gym at module level so SingleAssetTradingEnv can inherit the right base.
try:
    import gymnasium as _gym_module

    _USE_GYMNASIUM = True
except ImportError:
    try:
        import gym as _gym_module  # type: ignore[no-redef]

        _USE_GYMNASIUM = False
    except ImportError as _exc:
        raise ImportError("Could not import gymnasium or gym.") from _exc

_GymEnv = _gym_module.Env
_Spaces = _gym_module.spaces


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLBaselineConfig:
    """Central config for reproducible fold-wise TDQN baseline runs."""

    input_path: str = "data/spy_market_data.csv"
    output_dir: str = "data/baselines/rl_only"
    ticker: str = "SPY"

    train_timesteps: int = 200_000
    initial_amount: float = 10_000.0
    transaction_cost_pct: float = 1e-3
    # k-day lookahead window for precomputed training reward (Eq. 5-7 in paper).
    reward_window_k: int = 3
    seed: int = 42
    progress_interval_steps: int = 25_000

    # TDQN hyperparameters (DQN + Huber loss; SB3 uses SmoothL1Loss by default).
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_env_dataframe(split_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw split into a clean env DataFrame (no FinRL formatting)."""

    required = OHLCV_COLUMNS + INDICATOR_COLUMNS
    missing = [c for c in required if c not in split_df.columns]
    if missing:
        raise KeyError(f"Missing columns for env: {missing}")

    df = split_df.reset_index()
    if "Date" not in df.columns:
        raise KeyError("Expected `Date` column after reset_index().")

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = ["date"] + ENV_FEATURE_COLUMNS
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"] + ENV_FEATURE_COLUMNS)
    if df.empty:
        raise ValueError("Split is empty after dropping NaN feature rows.")
    return df.sort_values("date").reset_index(drop=True)


def compute_normalization_coeffs(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute per-feature (min, max) from training data for MinMax normalization."""
    coeffs: dict[str, tuple[float, float]] = {}
    for col in ENV_FEATURE_COLUMNS:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        coeffs[col] = (col_min, col_max)
    return coeffs


# ---------------------------------------------------------------------------
# Reward precomputation (offline label — training only)
# ---------------------------------------------------------------------------


def precompute_training_rewards(prices: np.ndarray, k: int) -> np.ndarray:
    """
    Compute R_signal[τ] for each training step using k-day lookahead (Eq. 5-7).

    R_signal is position-independent; multiply by the resulting position H to
    get the actual reward.  Last k rows get 0 because no future data exists.
    This label is only used during training; it never enters the observation.
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float32)
    for tau in range(n - k):
        p0 = prices[tau]
        if p0 <= 0.0:
            continue
        delta = (prices[tau + 1 : tau + k + 1] - p0) / p0 * 100.0
        r_max = float(np.max(delta)) if np.any(delta > 0) else 0.0
        r_min = float(np.min(delta)) if np.any(delta < 0) else 0.0
        if r_max > 0 or (r_max + r_min) > 0:
            signals[tau] = r_max
        elif r_min < 0 or (r_max + r_min) < 0:
            signals[tau] = r_min
        # else: both zero → signal stays 0 (no clear trend)
    return signals


# ---------------------------------------------------------------------------
# Position transition logic (Table 1)
# ---------------------------------------------------------------------------


def _position_transition(old: int, signal: int) -> tuple[int, int]:
    """Return (actual_action, new_position) from Table 1 of the paper."""
    if old == 0:
        if signal == 1:
            return (1, 1)    # open long
        if signal == -1:
            return (-1, -1)  # open short
        return (0, 0)        # hold cash
    if old == 1:
        if signal == -1:
            return (-1, 0)   # close long → flat
        return (0, 1)        # hold long (signal 0 or 1)
    # old == -1
    if signal == 1:
        return (1, 0)        # close short → flat
    return (0, -1)           # hold short (signal 0 or -1)


# ---------------------------------------------------------------------------
# Custom Gym environment
# ---------------------------------------------------------------------------

# Observation: account_value + 12 market features + 3 portfolio diagnostics.
_OBS_DIM = 1 + len(ENV_FEATURE_COLUMNS) + 3  # 16


class SingleAssetTradingEnv(_GymEnv):
    """
    Single-asset long/short/flat trading environment.

    Action space: Discrete(3) → index 0=signal -1, 1=signal 0, 2=signal +1.
    Observation (16-dim): [account_value, close, open, high, low, volume,
                           rsi_14, macd, macd_signal, rolling_vol_20,
                           ma_10, ma_20, ma_50,
                           current_position, step_return, frac_drawdown]

    Modes:
      "train" — returns precomputed lookahead reward (H_new * R_signal[τ]).
      "eval"  — returns actual portfolio return as reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float,
        transaction_cost_pct: float,
        reward_window_k: int = 3,
        mode: str = "train",
        precomputed_rewards: np.ndarray | None = None,
        initial_state: dict | None = None,
        norm_coeffs: dict | None = None,
    ):
        super().__init__()
        self._df = df.reset_index(drop=True)
        self._prices = self._df["close"].to_numpy(dtype=np.float64)
        self._n = len(self._df)
        self._initial_amount = float(initial_amount)
        self._tc = float(transaction_cost_pct)
        self._k = reward_window_k
        self._mode = mode
        self._initial_state = initial_state  # {"account_value": float, "position": int}
        self._norm_coeffs = norm_coeffs or {}
        self._norm_initial = float(initial_amount)

        if mode == "train":
            if precomputed_rewards is None:
                precomputed_rewards = precompute_training_rewards(self._prices, reward_window_k)
            self._reward_signals = precomputed_rewards.astype(np.float32)
        else:
            self._reward_signals = np.zeros(self._n, dtype=np.float32)

        # With normalization, most dims sit in [0, 1]; set tight bounds for known-range ones.
        finfo = np.finfo(np.float32)
        low = np.zeros(_OBS_DIM, dtype=np.float32)
        high = np.ones(_OBS_DIM, dtype=np.float32)
        high[0] = finfo.max          # norm_account_value: positive, unbounded above
        low[13] = -1.0               # current_position: {-1, 0, 1}
        low[14] = -1.0               # step_return: can be negative
        # idx 15 frac_drawdown: [0, 1] already set by zeros/ones

        self.observation_space = _Spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = _Spaces.Discrete(3)

        # Populated during rollout for export.
        self.date_history: list = []
        self.signal_history: list[int] = []
        self.actual_action_history: list[int] = []
        self.position_history: list[int] = []
        self.price_history: list[float] = []
        self.account_value_history: list[float] = []
        self.trade_notional_history: list[float] = []

        self._step_idx = 0
        self._position = 0
        self._account_value = self._initial_amount
        self._running_peak = self._initial_amount
        self._step_return = 0.0

    def _build_obs(self) -> np.ndarray:
        row = self._df.iloc[self._step_idx]

        norm_account_value = self._account_value / self._norm_initial

        frac_drawdown = (
            (self._running_peak - self._account_value) / self._running_peak
            if self._running_peak > 0.0
            else 0.0
        )

        # MinMax normalization using training-data coefficients (clipped to [0, 1]).
        def _mm(col: str) -> float:
            lo, hi = self._norm_coeffs.get(col, (0.0, 1.0))
            span = hi - lo
            if span < 1e-8:
                return 0.0
            return float(np.clip((float(row[col]) - lo) / span, 0.0, 1.0))

        return np.array(
            [
                norm_account_value,      # 0: account_value / initial_amount
                _mm("close"),            # 1
                _mm("open"),             # 2
                _mm("high"),             # 3
                _mm("low"),              # 4
                _mm("volume"),           # 5
                _mm("rsi_14"),           # 6
                _mm("macd"),             # 7
                _mm("macd_signal"),      # 8
                _mm("rolling_vol_20"),   # 9
                _mm("ma_10"),            # 10
                _mm("ma_20"),            # 11
                _mm("ma_50"),            # 12
                float(self._position),   # 13: {-1, 0, 1}
                self._step_return,       # 14: last portfolio return (small float)
                max(0.0, frac_drawdown), # 15: [0, 1]
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._step_return = 0.0

        if options is not None and isinstance(options, dict):
            # Stitched eval: carry terminal state from previous fold.
            self._account_value = float(options.get("account_value", self._initial_amount))
            self._position = int(options.get("position", 0))
        elif self._initial_state is not None:
            self._account_value = float(self._initial_state.get("account_value", self._initial_amount))
            self._position = int(self._initial_state.get("position", 0))
        else:
            self._account_value = self._initial_amount
            self._position = 0

        self._running_peak = self._account_value

        self.date_history.clear()
        self.signal_history.clear()
        self.actual_action_history.clear()
        self.position_history.clear()
        self.price_history.clear()
        self.account_value_history.clear()
        self.trade_notional_history.clear()

        obs = self._build_obs()
        if _USE_GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        # Map DQN index → signal ∈ {-1, 0, 1}.
        signal = int(action) - 1

        old_position = self._position
        actual_action, new_position = _position_transition(old_position, signal)

        current_price = float(self._prices[self._step_idx])
        next_price = (
            float(self._prices[self._step_idx + 1])
            if self._step_idx + 1 < self._n
            else current_price
        )

        # Apply transaction cost on position change, before P&L.
        trade_notional = 0.0
        if actual_action != 0:
            trade_notional = self._account_value          # full position value traded
            tc_cost = trade_notional * self._tc
            self._account_value -= tc_cost

        # P&L: long profits from up moves, short profits from down moves.
        asset_return = (next_price - current_price) / current_price if current_price > 0.0 else 0.0
        portfolio_return = float(new_position) * asset_return
        self._account_value *= 1.0 + portfolio_return
        self._step_return = portfolio_return

        self._running_peak = max(self._running_peak, self._account_value)
        self._position = new_position

        # Record history for post-episode export.
        self.date_history.append(self._df.iloc[self._step_idx]["date"])
        self.signal_history.append(signal)
        self.actual_action_history.append(actual_action)
        self.position_history.append(new_position)
        self.price_history.append(current_price)
        self.account_value_history.append(self._account_value)
        self.trade_notional_history.append(trade_notional)

        # Advance step.
        self._step_idx += 1
        done = self._step_idx >= self._n - 1

        obs = self._build_obs()

        if self._mode == "train":
            R = float(self._reward_signals[self._step_idx - 1])
            reward = float(np.clip(float(new_position) * R, -1.0, 1.0))
            # Counterfactual rewards for all 3 action indices (Theate & Ernst exploration trick).
            cf_rewards: dict[int, float] = {}
            for cf_idx in range(3):
                cf_sig = cf_idx - 1
                _, cf_pos = _position_transition(old_position, cf_sig)
                cf_rewards[cf_idx] = float(np.clip(float(cf_pos) * R, -1.0, 1.0))
            info: dict = {"counterfactual_rewards": cf_rewards}
        else:
            reward = portfolio_return
            info = {}

        if _USE_GYMNASIUM:
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Dependency loading
# ---------------------------------------------------------------------------


def _load_tdqn_dependencies():
    """Lazy-load stable_baselines3 and torch so import never fails on import."""
    try:
        import torch
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise ImportError(
            "Could not import stable_baselines3 or torch. "
            "Install stable-baselines3 and PyTorch."
        ) from exc
    return DQN, torch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_fold_tdqn(
    train_df: pd.DataFrame, config: RLBaselineConfig, val_df: pd.DataFrame | None = None
) -> tuple:
    """
    Train one TDQN model on one fold's training split.

    Returns (model, norm_coeffs) so the caller can pass norm_coeffs to evaluation.
    """
    DQN, torch = _load_tdqn_dependencies()
    from stable_baselines3.common.callbacks import BaseCallback

    norm_coeffs = compute_normalization_coeffs(train_df)
    prices = train_df["close"].to_numpy(dtype=np.float64)
    precomputed = precompute_training_rewards(prices, config.reward_window_k)

    env = SingleAssetTradingEnv(
        df=train_df,
        initial_amount=config.initial_amount,
        transaction_cost_pct=config.transaction_cost_pct,
        reward_window_k=config.reward_window_k,
        mode="train",
        precomputed_rewards=precomputed,
        norm_coeffs=norm_coeffs,
    )

    class _CounterfactualProgressCallback(BaseCallback):
        """
        After each env step, push counterfactual transitions for the two
        non-taken actions into the replay buffer (Theate & Ernst, 2020).
        Both the actual and counterfactual outcomes are observable from the
        same pre-step state, doubling the useful signal per env interaction.

        Also prints periodic progress so long SB3 learn() calls do not look
        stalled while they are still doing useful work.
        """

        def __init__(self, total_timesteps: int, interval_steps: int, val_df=None, norm_coeffs=None, config=None) -> None:
            super().__init__()
            self.total_timesteps = int(total_timesteps)
            self.interval_steps = max(0, int(interval_steps))
            self._start_time = 0.0
            self._last_report_timestep = 0
            self._val_df = val_df
            self._val_norm_coeffs = norm_coeffs
            self._val_config = config

        @staticmethod
        def _format_seconds(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            hours, rem = divmod(int(seconds), 3600)
            minutes, secs = divmod(rem, 60)
            if hours:
                return f"{hours}h {minutes:02d}m {secs:02d}s"
            return f"{minutes}m {secs:02d}s"

        def _on_training_start(self) -> None:
            self._start_time = time.perf_counter()
            self._last_report_timestep = 0
            print(
                f"  TDQN learn start: {self.total_timesteps:,} timesteps "
                f"(progress every {self.interval_steps:,} steps)",
                flush=True,
            )

        def _maybe_report_progress(self) -> None:
            if self.interval_steps <= 0:
                return
            current = int(self.num_timesteps)
            is_done = current >= self.total_timesteps
            if not is_done and current - self._last_report_timestep < self.interval_steps:
                return

            elapsed = time.perf_counter() - self._start_time
            steps_per_sec = current / elapsed if elapsed > 0.0 else 0.0
            remaining_steps = max(0, self.total_timesteps - current)
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0.0 else 0.0
            pct = min(100.0, 100.0 * current / max(1, self.total_timesteps))
            print(
                "  TDQN progress: "
                f"{current:,}/{self.total_timesteps:,} ({pct:5.1f}%) | "
                f"{steps_per_sec:,.1f} steps/s | "
                f"elapsed {self._format_seconds(elapsed)} | "
                f"eta {self._format_seconds(eta)}",
                flush=True,
            )
            self._last_report_timestep = current

            if self._val_df is not None and self._val_config is not None:
                try:
                    val_account_df, _, _ = evaluate_tdqn_on_split(
                        model=self.model,
                        split_df=self._val_df,
                        config=self._val_config,
                        norm_coeffs=self._val_norm_coeffs,
                    )
                    val_metrics = compute_performance_metrics(val_account_df)
                    print(
                        f"  Val: total_return={val_metrics['total_return']:.4f} "
                        f"sharpe={val_metrics['sharpe_ratio']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"  [WARN] Val eval failed: {exc}", flush=True)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [{}])
            actions = self.locals.get("actions", [])
            new_obs = self.locals.get("new_obs")
            dones = self.locals.get("dones")
            if new_obs is None or dones is None:
                self._maybe_report_progress()
                return True
            last_obs = self.model._last_obs  # pre-step; updated after _on_step
            for i, info in enumerate(infos):
                cf_rewards = info.get("counterfactual_rewards", {})
                if not cf_rewards:
                    continue
                actual_action = int(np.asarray(actions[i]).reshape(-1)[0])
                for cf_action_idx, cf_reward in cf_rewards.items():
                    if cf_action_idx == actual_action:
                        continue
                    self.model.replay_buffer.add(
                        last_obs[i : i + 1],
                        new_obs[i : i + 1],
                        np.array([[cf_action_idx]], dtype=np.int64),
                        np.array([cf_reward], dtype=np.float32),
                        np.array([bool(dones[i])]),
                        [{}],
                    )
            self._maybe_report_progress()
            return True

    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
    )

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
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        verbose=0,
    )
    model.learn(
        total_timesteps=config.train_timesteps,
        progress_bar=False,
        callback=_CounterfactualProgressCallback(
            total_timesteps=config.train_timesteps,
            interval_steps=config.progress_interval_steps,
            val_df=val_df,
            norm_coeffs=norm_coeffs,
            config=config,
        ),
    )
    return model, norm_coeffs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_tdqn_on_split(
    model,
    split_df: pd.DataFrame,
    config: RLBaselineConfig,
    initial_state: dict | None = None,
    norm_coeffs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run deterministic policy rollout and return trajectory DataFrames."""

    env = SingleAssetTradingEnv(
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
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).reshape(-1)[0])
        step_out = env.step(action_int)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, _ = step_out
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError("Evaluation exceeded expected number of steps.")

    # Build account_value_df: row 0 = starting state, then one row per step.
    # Use split_df dates directly: date[0]=start, date[k]=value after step k-1's P&L.
    # P&L at step τ spans price[τ]→price[τ+1], so account_value[τ] belongs on date[τ+1].
    # split_df has N rows; env runs N-1 steps → N values total, perfectly aligned.
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

    # Build actions_df — one row per step.
    n = len(env.date_history)
    account_arr = np.array(env.account_value_history, dtype=float)
    prev_account = np.concatenate([[start_val], account_arr[:-1]]) if n > 1 else np.array([start_val])
    step_pnl = account_arr - prev_account

    running_peak = np.maximum.accumulate(account_arr)
    drawdown = running_peak - account_arr

    positions = np.array(env.position_history, dtype=float)

    actions_df = pd.DataFrame(
        {
            "date": env.date_history,
            "signal": env.signal_history,
            "actual_action": env.actual_action_history,
            "position": env.position_history,
            "position_after_trade": env.position_history,   # alias for metrics compat
            "weight": positions,                             # -1/0/1 as float
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
# Performance metrics
# ---------------------------------------------------------------------------


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

    daily_returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)

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
                # Count both long (+1) and short (-1) as active.
                holding_mask = positions != 0.0
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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


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

    ax.set_title("SPY Portfolio Growth (RL-Only TDQN)")
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
    """Build buy-and-hold and momentum equity curves over stitched OOS dates."""

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

    bh_shares = float(initial_amount / prices.iloc[0])
    bh_equity = bh_shares * prices

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
        {"date": prices.index, "strategy": "buy_hold", "account_value": bh_equity.values}
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
    Grid of subplots comparing RL-TDQN, Buy-and-Hold, and Momentum on each
    fold's test period — all three strategies start at the same initial capital
    on the same first test day.
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
        "rl_tdqn": {"color": "tab:blue", "label": "RL-TDQN"},
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
        "Per-Fold Test Period: RL-TDQN vs Buy & Hold vs Momentum (each fold starts at $10k)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fold resolver
# ---------------------------------------------------------------------------


def _resolve_folds(selected_folds: str | None, max_folds: int | None) -> list[dict[str, str]]:
    if selected_folds:
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


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------


def run_walkforward_rl_only_baseline(
    config: RLBaselineConfig,
    selected_folds: str | None = None,
    max_folds: int | None = None,
) -> pd.DataFrame:
    """
    End-to-end walk-forward loop:
    1) split data by fold
    2) train TDQN on train split with precomputed rewards
    3) evaluate on val/test (deterministic policy, actual P&L)
    4) write fold artifacts and aggregate metrics
    """

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
            f"Running {fold['name']}: "
            f"train {fold['train_start']} to {fold['train_end']}, "
            f"val {fold['val_start']} to {fold['val_end']}, "
            f"test {fold['test_start']} to {fold['test_end']}"
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

        train_df = prepare_env_dataframe(train_raw)
        val_df = prepare_env_dataframe(val_raw)
        test_df = prepare_env_dataframe(test_raw)

        model, norm_coeffs = train_fold_tdqn(train_df=train_df, config=config, val_df=val_df)

        fold_dir = output_root / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_test_account_df: pd.DataFrame | None = None
        split_map = {"val": val_df, "test": test_df}
        for split_name, split_df in split_map.items():
            account_df, actions_df, _ = evaluate_tdqn_on_split(
                model=model,
                split_df=split_df,
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

        # Per-fold fair comparison: all strategies start at same $10k on same day.
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

        # Stitched eval: carry terminal position/cash across folds.
        stitched_account_df, stitched_actions_df, stitched_previous_state = evaluate_tdqn_on_split(
            model=model,
            split_df=test_df,
            config=config,
            initial_state=stitched_previous_state,
            norm_coeffs=norm_coeffs,
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

    # Per-fold comparison output.
    if per_fold_comparisons:
        per_fold_df = pd.concat(per_fold_comparisons, ignore_index=True)
        per_fold_df["date"] = pd.to_datetime(per_fold_df["date"], errors="coerce")
        per_fold_df = (
            per_fold_df.dropna(subset=["date", "account_value", "strategy"])
            .sort_values(["fold", "strategy", "date"])
            .reset_index(drop=True)
        )
        per_fold_df.to_csv(output_root / "rl_only_per_fold_test_comparison.csv", index=False)
        save_per_fold_comparison_plot(
            per_fold_comparison_df=per_fold_df,
            output_path=output_root / "rl_only_per_fold_test_comparison.png",
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
            output_root / "rl_only_per_fold_test_metrics.csv", index=False
        )

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

        all_stitched_actions = pd.concat(stitched_test_actions, ignore_index=True)
        all_stitched_actions.to_csv(
            output_root / "rl_only_stitched_test_actions.csv", index=False
        )

        classic_df = build_stitched_classic_baselines(
            full_df=full_df,
            stitched_dates=stitched_rl_df["date"],
            initial_amount=config.initial_amount,
            momentum_lookback=126,
        )
        stitched_rl_long = stitched_rl_df[["date", "account_value"]].assign(
            strategy="rl_tdqn_stitched"
        )
        comparison_df = pd.concat([stitched_rl_long, classic_df], ignore_index=True)
        comparison_df["date"] = pd.to_datetime(comparison_df["date"], errors="coerce")
        comparison_df = (
            comparison_df.dropna(subset=["date", "account_value", "strategy"])
            .sort_values(["date", "strategy"])
            .reset_index(drop=True)
        )
        comparison_df.to_csv(output_root / "rl_only_stitched_comparison.csv", index=False)
        save_strategy_comparison_plot(
            comparison_df=comparison_df,
            output_path=output_root / "rl_only_stitched_comparison.png",
        )

        stitched_metrics_rows: list[dict] = []
        for strategy_name, strategy_df in comparison_df.groupby("strategy"):
            strategy_actions_df = (
                all_stitched_actions if strategy_name == "rl_tdqn_stitched" else None
            )
            strategy_metrics = compute_performance_metrics(
                strategy_df[["date", "account_value"]].sort_values("date").reset_index(drop=True),
                actions_df=strategy_actions_df,
            )
            stitched_metrics_rows.append({"strategy": strategy_name, **strategy_metrics})
        pd.DataFrame(stitched_metrics_rows).to_csv(
            output_root / "rl_only_stitched_comparison_metrics.csv", index=False
        )

    with open(output_root / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    return metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a TDQN baseline across walk-forward folds "
            "using SPY OHLCV + technical indicators."
        )
    )
    parser.add_argument("--input-path", type=str, default="data/spy_market_data.csv")
    parser.add_argument("--output-dir", type=str, default="data/baselines/rl_only")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--train-timesteps", type=int, default=200_000)
    parser.add_argument("--initial-amount", type=float, default=10_000.0)
    parser.add_argument(
        "--transaction-cost-pct",
        type=float,
        default=1e-3,
        help="Flat fee applied to account value on every position change.",
    )
    parser.add_argument(
        "--reward-window-k",
        type=int,
        default=3,
        help="Lookahead window k (trading days) for precomputed training rewards.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        help="Optional cap on number of folds to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RLBaselineConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        ticker=args.ticker,
        train_timesteps=args.train_timesteps,
        initial_amount=args.initial_amount,
        transaction_cost_pct=args.transaction_cost_pct,
        reward_window_k=args.reward_window_k,
        seed=args.seed,
    )
    metrics_df = run_walkforward_rl_only_baseline(
        config=config,
        selected_folds=args.folds,
        max_folds=args.max_folds,
    )
    print("TDQN baseline complete. Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
