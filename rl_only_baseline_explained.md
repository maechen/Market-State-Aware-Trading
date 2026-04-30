# RL-Only Baseline: Complete Walkthrough

This document explains every part of `src/baselines/rl_only_baseline.py` in plain language, including how all the pieces connect, how money moves, what each concept means, and what to expect from the results.

The architecture is based on the position-based TDQN approach described in:
> "Trend-following and Mean-reversion in Financial Markets" — MDPI Symmetry 2025 (https://www.mdpi.com/2073-8994/18/1/112)

---

## Table of Contents

1. [What This File Is Trying To Do](#1-what-this-file-is-trying-to-do)
2. [The Big Picture: How All Pieces Fit Together](#2-the-big-picture-how-all-pieces-fit-together)
3. [Key Vocabulary Before We Start](#3-key-vocabulary-before-we-start)
4. [What is Gym / Gymnasium?](#4-what-is-gym--gymnasium)
5. [What is SingleAssetTradingEnv?](#5-what-is-singleassettradingenv)
6. [The Walk-Forward Folds](#6-the-walk-forward-folds)
7. [Configuration Parameters (RLBaselineConfig)](#7-configuration-parameters-rlbaselineconfig)
8. [How Funding Works — The Portfolio Mechanics](#8-how-funding-works--the-portfolio-mechanics)
9. [The Action Space: Position Signals (Table 1)](#9-the-action-space-position-signals-table-1)
10. [The Observation Space: What the Agent Sees](#10-the-observation-space-what-the-agent-sees)
11. [The Reward Function: k-Step Lookahead vs Actual P&L](#11-the-reward-function-k-step-lookahead-vs-actual-pl)
12. [What is TDQN?](#12-what-is-tdqn)
13. [Function-by-Function Walkthrough](#13-function-by-function-walkthrough)
14. [End-to-End Data Flow: What Happens When You Run the Script](#14-end-to-end-data-flow-what-happens-when-you-run-the-script)
15. [The Two Evaluation Modes: Independent vs Stitched](#15-the-two-evaluation-modes-independent-vs-stitched)
16. [The Classic Baselines: Buy-and-Hold and Momentum](#16-the-classic-baselines-buy-and-hold-and-momentum)
17. [Is the Comparison Fair? What Should We Expect?](#17-is-the-comparison-fair-what-should-we-expect)
18. [Output Files](#18-output-files)

---

## 1. What This File Is Trying To Do

This file trains and evaluates a trading algorithm for SPY (the S&P 500 ETF). The algorithm is a **TDQN agent** (Trend-following Deep Q-Network). Its job is to look at today's market data and decide whether to go **Long** (own SPY), **Flat** (hold cash), or **Short** (profit from SPY falling).

The algorithm is not told "buy when RSI is low" or "sell when volatility spikes." Instead, it **learns by trial and error** using offline reward labels that tell it whether going long or short over the next few days would have been profitable.

Once trained, it is compared to two simple strategies: Buy-and-Hold (just buy SPY and never touch it) and a Momentum strategy (buy when the 6-month trend is up, exit otherwise).

This is called a **baseline** because it will later be compared to the full model that adds regime detection and sentiment analysis. If this RL agent already beats Buy-and-Hold, that's noteworthy. If the advanced model beats this, we can quantify exactly how much the extra signals help.

---

## 2. The Big Picture: How All Pieces Fit Together

```
spy_market_data.csv
        |
        v
   load_data()              ← loads raw daily OHLCV + technical indicators
        |
        v
 create_walk_forward_split()  ← splits data into train / val / test per fold
        |
        v
 prepare_env_dataframe()     ← renames columns, drops NaNs, sorts by date
        |
        v
 precompute_training_rewards()  ← offline k-day lookahead labels (train only)
        |
        +-----------+
        |           |
  [train data]  [val/test data]
        |
        v
 train_fold_tdqn()
        |
        ├── SingleAssetTradingEnv(mode="train")   ← custom Gym env
        ├── DQN("MlpPolicy", [128,128], ReLU)     ← TDQN architecture
        └── model.learn(total_timesteps=50_000)   ← agent learns from reward labels
              |
              v
           trained_model
                |
                v
  evaluate_tdqn_on_split(mode="eval")
        |
        ├── deterministic policy rollout
        ├── actual P&L tracked per step
        └── terminal_state saved for stitched eval
                |
                v
  compute_performance_metrics()
                |
                v
  build_stitched_classic_baselines()  ← buy-hold + momentum curves
                |
                v
  save plots + CSVs
```

Everything revolves around `SingleAssetTradingEnv` — a custom stock market simulator. The agent interacts with it like a game: it sees today's market data, picks a direction signal {-1, 0, +1}, the environment transitions to the appropriate position and computes P&L, and the agent gets a reward. This repeats for every trading day.

---

## 3. Key Vocabulary Before We Start

| Term | Plain English |
|------|---------------|
| **Episode** | One full pass through a dataset (e.g. all of 2007–2011 for fold1 train) |
| **Step** | One trading day — the agent sees today's data, acts, moves to tomorrow |
| **Observation / State** | The 17 numbers the agent sees at each step (prices, indicators, position info) |
| **Signal** | The agent's raw output: -1 (go short/stay short), 0 (go flat/stay), +1 (go long/stay long) |
| **Actual Action** | What physically happens: -1 (close or open short), 0 (hold), +1 (close or open long) |
| **Position** | Current portfolio stance: -1 = short, 0 = flat/cash, +1 = long |
| **Reward** | During training: precomputed lookahead label. During eval: actual portfolio return. |
| **Policy** | The learned function: given an observation → output a signal |
| **Fold** | One train/val/test time window (e.g. train 2007–2011, val 2012, test 2013) |
| **Walk-forward** | Testing on data the model never saw, one fold at a time |
| **Stitched** | Evaluation where terminal cash/position carry over from one fold's test into the next |
| **TC** | Transaction cost — a flat 0.1% of account value charged when the position changes |

---

## 4. What is Gym / Gymnasium?

**Gym** (now called Gymnasium) is a Python library made by OpenAI. It defines a standard interface for training RL agents.

Any RL environment — a video game, a robot, or a stock simulator — exposes:
- `env.reset()` → start a new episode, get the initial observation
- `env.step(action)` → take an action, get back `(new_observation, reward, done?, info)`
- `env.observation_space` → describes what the observation looks like (shape, min/max)
- `env.action_space` → describes what actions are valid

This standardization means SB3's DQN can work with our stock environment without knowing anything about finance.

The code detects which version is installed at module level:
```python
try:
    import gymnasium as _gym_module     # newer name
    _USE_GYMNASIUM = True
except ImportError:
    import gym as _gym_module           # older name
    _USE_GYMNASIUM = False
```

This matters because `gymnasium.reset()` returns `(obs, info)` while `gym.reset()` returns just `obs`. The code handles both branches.

---

## 5. What is SingleAssetTradingEnv?

`SingleAssetTradingEnv` is a custom Gym environment written from scratch (no FinRL). It handles:

- Tracking `account_value` as cash equivalent (all P&L expressed in dollar terms)
- Applying transaction costs **before** the P&L for each position change
- Computing P&L from 100% long or 100% short exposure
- Maintaining a `running_peak` for drawdown calculation
- Recording full trade history for post-episode analysis
- Supporting `initial_state` injection so stitched evaluation can carry cash/position forward

**Why not FinRL?**
FinRL only supports long/flat positions (cannot short). It also couples share-count bookkeeping with its action space in a way that would require complex patching to support short selling. Building from scratch gave us clean control over every assumption.

**At reset:**
- `account_value` is set to `initial_amount` (or carried from previous fold in stitched mode)
- `position` is set to 0 (or carried from previous fold)
- All history lists are cleared

**At each step:**
1. Map DQN output index (0,1,2) → signal ∈ {-1, 0, +1}
2. Look up (old_position, signal) → (actual_action, new_position) using Table 1
3. If `actual_action ≠ 0`: deduct TC from `account_value` first
4. Compute asset return `(next_price - current_price) / current_price`
5. Apply P&L: `account_value *= 1 + position * asset_return`
6. Update `running_peak = max(running_peak, account_value)`
7. Record history, advance `step_idx`
8. Return reward (lookahead label in train mode, portfolio return in eval mode)

---

## 6. The Walk-Forward Folds

The SPY dataset (~2007 to 2020) is split into 8 overlapping time windows called **folds**. Each fold has three non-overlapping segments:

```
Fold 1:
  Train: 2007-01-01 → 2011-12-31  (5 years, ~1260 trading days)
  Val:   2012-01-01 → 2012-12-31  (1 year,  ~252 trading days)
  Test:  2013-01-01 → 2013-12-31  (1 year,  ~252 trading days)

Fold 2:
  Train: 2008-01-01 → 2012-12-31
  Val:   2013-01-01 → 2013-12-31
  Test:  2014-01-01 → 2014-12-31
...and so on, sliding one year forward each fold.
```

**Why walk-forward?**
Simulates real-world deployment: train up to date X, then test on the unseen year X+1. The agent never touches test data during training. This is far more realistic than a single train/test split.

**Why 8 folds?**
It gives 8 different out-of-sample test years (2013–2020), spanning bull markets, corrections, elevated volatility (2018), and the COVID crash onset (2020). This makes conclusions more robust.

**What val is used for:**
In this baseline, val metrics are recorded for diagnostics but no automatic hyperparameter tuning is done. Hyperparameters are fixed in `RLBaselineConfig`.

---

## 7. Configuration Parameters (RLBaselineConfig)

The `RLBaselineConfig` dataclass is frozen (immutable). All hyperparameters live here.

### Data & Output
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `input_path` | `data/spy_market_data.csv` | Where to load market data from |
| `output_dir` | `data/baselines/rl_only` | Where to save results |
| `ticker` | `"SPY"` | Label used in file names |

### Portfolio Setup
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `initial_amount` | `10,000.0` | Starting portfolio value in dollars |
| `transaction_cost_pct` | `0.001` (0.1%) | Flat fee applied to account value on every position change |

### Reward
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `reward_window_k` | `3` | Days of lookahead for precomputed reward labels (train mode only) |

### TDQN Hyperparameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `dqn_learning_rate` | `1e-4` | Adam optimizer step size |
| `dqn_batch_size` | `64` | Transitions per gradient update |
| `dqn_buffer_size` | `50,000` | Replay buffer capacity |
| `dqn_learning_starts` | `1,000` | Steps before any gradient updates begin |
| `dqn_train_freq` | `4` | Collect 4 transitions before each gradient update |
| `dqn_gradient_steps` | `1` | Gradient steps per training update |
| `dqn_target_update_interval` | `500` | Steps between copying online network → target network |
| `dqn_tau` | `1.0` | Hard target update (tau=1 = copy weights fully, not a soft blend) |
| `dqn_gamma` | `0.99` | Discount factor for future rewards |
| `dqn_exploration_fraction` | `0.1` | Fraction of training spent decaying epsilon |
| `dqn_exploration_final_eps` | `0.02` | Final epsilon (2% random actions) |
| `seed` | `42` | Reproducibility seed for numpy, torch, and gym |

---

## 8. How Funding Works — The Portfolio Mechanics

The agent always controls 100% of `account_value`. There are no fractional positions — the agent is either fully long, fully flat (cash), or fully short.

**Going Long (position = +1):**
- The full `account_value` is used to buy SPY
- If SPY goes up 1%, `account_value` grows by 1%
- If SPY goes down 1%, `account_value` shrinks by 1%

**Going Flat (position = 0):**
- All money sits as cash
- No P&L regardless of SPY moves

**Going Short (position = -1):**
- The agent borrows and sells SPY
- If SPY goes down 1%, `account_value` grows by 1% (profits from the fall)
- If SPY goes up 1%, `account_value` shrinks by 1%
- Borrow costs are ignored in this baseline

**Transaction Costs:**
When a position change actually occurs (actual_action ≠ 0), a flat 0.1% fee is deducted **before** the P&L is applied:

```
tc_cost = account_value × 0.001
account_value -= tc_cost
# then P&L is computed on the post-TC account value
```

Charging TC before P&L means the fee comes directly out of capital, which is slightly more conservative than charging after.

---

## 9. The Action Space: Position Signals (Table 1)

This is the core design from the paper. The agent does **not** directly pick "long" or "short." Instead it emits a **signal** ∈ {-1, 0, +1} and the environment computes the actual action based on the current position:

| Old Position | Signal | Actual Action | New Position |
|:---:|:---:|:---:|:---:|
| 0 (flat) | +1 | +1 (open long) | +1 |
| 0 (flat) | 0 | 0 (stay flat) | 0 |
| 0 (flat) | -1 | -1 (open short) | -1 |
| +1 (long) | +1 | 0 (hold long) | +1 |
| +1 (long) | 0 | 0 (hold long) | +1 |
| +1 (long) | -1 | -1 (close long → flat) | 0 |
| -1 (short) | -1 | 0 (hold short) | -1 |
| -1 (short) | 0 | 0 (hold short) | -1 |
| -1 (short) | +1 | +1 (close short → flat) | 0 |

**Why this design solves the old dead-zone problem:**
In the previous architecture (weight deltas), from a starting weight of 0%, actions -1.0 and -0.5 both resulted in negative weights that were clamped to 0% — meaning 3 out of 5 actions did nothing. The agent learned to do nothing.

With Table 1, from any starting position, every signal triggers a meaningful transition. There is no dead zone.

**DQN uses indices 0, 1, 2:**
SB3's `Discrete(3)` action space uses non-negative integers. The mapping is:
- DQN output 0 → signal -1 (short/exit)
- DQN output 1 → signal 0 (hold/do nothing)
- DQN output 2 → signal +1 (long/enter)

In code: `signal = int(action) - 1`

---

## 10. The Observation Space: What the Agent Sees

At every step, the environment returns a 17-dimensional float32 vector:

| Index | Feature | What It Represents |
|:---:|---|---|
| 0 | `account_value` | Current portfolio value in dollars |
| 1 | `close` | Today's SPY closing price |
| 2 | `signed_exposure` | `position × account_value` — positive=long dollars, negative=short dollars, 0=flat |
| 3 | `open` | Today's opening price |
| 4 | `high` | Today's high |
| 5 | `low` | Today's low |
| 6 | `volume` | Today's volume |
| 7 | `rsi_14` | RSI over 14 days — momentum oscillator (0–100) |
| 8 | `macd` | MACD line — trend indicator |
| 9 | `macd_signal` | MACD signal line — smoothed MACD |
| 10 | `rolling_vol_20` | 20-day rolling volatility |
| 11 | `ma_10` | 10-day moving average |
| 12 | `ma_20` | 20-day moving average |
| 13 | `ma_50` | 50-day moving average |
| 14 | `current_position` | Current position ∈ {-1, 0, +1} |
| 15 | `step_return` | Portfolio return from the previous step |
| 16 | `frac_drawdown` | `(running_peak - account_value) / running_peak` clipped to [0, 1] |

**Why `signed_exposure` (index 2)?**
It tells the agent the dollar amount at risk. When long with $15,000 it's +15,000. When short it's negative. This lets the agent reason about magnitude, not just direction.

**Why `current_position` (index 14) separately from `signed_exposure` (index 2)?**
When `account_value` is very small (near-zero after losses), `signed_exposure` would be near zero for both long and flat. `current_position` gives the agent a clean categorical signal of its stance regardless of account size.

---

## 11. The Reward Function: k-Step Lookahead vs Actual P&L

The reward function works differently in training vs evaluation — this is deliberate and important.

### Training Reward (mode="train"): Precomputed k-Step Lookahead Label

During training, the environment uses an **offline label** precomputed from the training data:

```
For each training day τ:
    delta[i] = (price[τ+i] - price[τ]) / price[τ] × 100   for i = 1..k

    R_signal[τ] = max(delta)    if any delta > 0 (upside available)
                = min(delta)    if any delta < 0 (downside available)
                = 0             otherwise (flat market)
```

This is Equations 5–7 from the paper. The reward the agent receives is:

```
reward = H_new × R_signal[τ]
```

Where `H_new` is the new position (+1 = long, 0 = flat, -1 = short).

**Example:** If k=3 and SPY is about to go up 2.1% over the next 3 days:
- `R_signal = +2.1`
- Long position (+1) → reward = +2.1 ✓ (correct direction)
- Flat position (0) → reward = 0 (missed opportunity, no penalty)
- Short position (-1) → reward = -2.1 ✗ (wrong direction)

**Why this is better than just observing P&L:**
Real trading rewards are very noisy — a correct bet can lose money on day 1 but win by day 3. The lookahead reward tells the agent the "right answer" over the next k days, giving a clear training signal even when short-term noise would be misleading.

**No look-ahead bias:**
The reward labels are computed from training data only. They are never part of the observation (the agent doesn't see future prices). At test time, the labels are unused — the agent acts from its learned policy with only current observations as input.

### Evaluation Reward (mode="eval"): Actual Portfolio Return

At val/test time, the reward is simply:
```
reward = position × asset_return
```

This is the true portfolio P&L for one day. Metrics are computed from this.

---

## 12. What is TDQN?

**DQN (Deep Q-Network)** is a reinforcement learning algorithm. At its core, it learns a function `Q(state, action) → expected_future_reward`. The agent always picks the action with the highest Q-value (exploitation), or a random action with probability epsilon (exploration).

**The Q-network** is updated via:
```
target = reward + γ × max_a' Q_target(next_state, a')
loss = Huber(Q_online(state, action) - target)
```

The **target network** is a copy of the Q-network that updates slowly. This prevents the target values from chasing a moving network and destabilizes training.

**TDQN** is DQN configured specifically for financial time series (from the paper):
- **Architecture:** `[128, 128]` MLP with ReLU activations (not the default `[64, 64]`)
- **Loss:** Huber / SmoothL1 (SB3's default — robust to outlier rewards)
- **Target update:** `tau=1.0` — hard copy every 500 steps, not a soft blend. This gives more stable targets in financial data where reward distributions shift over time.
- **Gamma:** `0.99` — values rewards up to ~100 steps in the future

**Why DQN and not PPO or A2C?**
DQN is specifically designed for discrete action spaces. Our action space is exactly `Discrete(3)` — short/flat/long — which is a natural fit. DQN's experience replay also helps with the non-stationarity of financial data by breaking correlations between consecutive trading days.

---

## 13. Function-by-Function Walkthrough

### `prepare_env_dataframe(split_df)`
Converts a raw split (uppercase column names from CSV) to the environment format (lowercase, sorted by date). Drops rows with any NaN in the feature columns. Validates all required OHLCV and indicator columns are present.

### `precompute_training_rewards(prices, k)`
Runs once per fold before training. For each price index τ (except the last k), computes the max upside and max downside over the next k days as percentage returns. Returns `R_signal[τ]` — a 1D float32 array of the same length as the price array. Last k entries are 0 (no future data).

### `_position_transition(old, signal)`
Pure function. Implements Table 1 from the paper. Takes `old_position ∈ {-1, 0, 1}` and `signal ∈ {-1, 0, 1}`, returns `(actual_action, new_position)`.

### `SingleAssetTradingEnv.__init__`
Sets up observation and action spaces, initializes all history lists, precomputes reward signals if in train mode.

### `SingleAssetTradingEnv._build_obs`
Assembles the 17-dim observation vector from current env state and the current DataFrame row.

### `SingleAssetTradingEnv.reset`
Restores to start of episode. If `options` or `initial_state` is provided (stitched eval), restores that account_value and position. Otherwise starts fresh at `initial_amount` and position 0.

### `SingleAssetTradingEnv.step`
Core simulation step (described in Section 5). Returns `(obs, reward, done, info)` or `(obs, reward, terminated, truncated, info)` depending on gym version.

### `train_fold_tdqn(train_df, config)`
Instantiates the env in train mode, builds the DQN with `[128, 128]` MLP and ReLU, calls `model.learn()`.

### `evaluate_tdqn_on_split(model, split_df, config, initial_state)`
Runs the trained model deterministically on one split. Builds two DataFrames:
- `account_value_df`: date + account_value (row 0 = starting state, then one per step)
- `actions_df`: date, signal, actual_action, position, trade_price, trade_notional, pnl, drawdown

Also returns `terminal_state = {"account_value": float, "position": int}` for stitched evaluation.

### `compute_performance_metrics(account_value_df, actions_df)`
Standard backtest metrics from the equity curve:
- `total_return`, `annualized_return`, `annualized_volatility`
- `sharpe_ratio` = annualized_return / annualized_volatility
- `sortino_ratio` = annualized_return / downside_volatility
- `calmar_ratio` = annualized_return / |max_drawdown|
- `max_drawdown`, `turnover`, `annualized_turnover`, `average_holding_time_days`

### `build_stitched_classic_baselines(full_df, stitched_dates, initial_amount)`
Builds equity curves for Buy-and-Hold and 126-day Momentum over the exact same dates as the stitched RL curve, starting from the same `initial_amount`. Used for fair apples-to-apples comparison.

### `run_walkforward_rl_only_baseline(config, selected_folds, max_folds)`
Outer loop over all folds. For each fold: splits data, trains TDQN, evaluates on val/test independently (each fold restarts at $10k), evaluates on test stitched (cash/position carry over), saves all artifacts.

---

## 14. End-to-End Data Flow: What Happens When You Run the Script

```
python src/baselines/rl_only_baseline.py
```

1. **`load_data()`** — loads `data/spy_market_data.csv`, which has OHLCV + indicators
2. **Loop over 8 folds:**
   a. `create_walk_forward_split()` cuts the data into train/val/test windows
   b. `prepare_env_dataframe()` cleans each split into environment format
   c. `precompute_training_rewards()` builds k-day lookahead labels for the train split
   d. `train_fold_tdqn()` trains the TDQN for 50,000 timesteps
   e. `evaluate_tdqn_on_split()` runs the policy on val and test independently
   f. `evaluate_tdqn_on_split()` runs stitched test (carrying terminal state from fold n-1)
   g. Saves per-fold CSVs and per-fold comparison plots
3. After all folds, writes aggregate CSVs: fold metrics, stitched equity curve, stitched comparison metrics
4. Writes `run_config.json` — a snapshot of all hyperparameters used

---

## 15. The Two Evaluation Modes: Independent vs Stitched

### Independent (per-fold)
Each fold's val and test evaluation starts fresh: `account_value = 10,000`, `position = 0`. This is used for per-fold diagnostics and the per-fold comparison plots against B&H and Momentum. It answers: "In this isolated year, how did the agent do?"

### Stitched (cross-fold)
The test evaluation chains folds together. Fold 1 test ends with some `account_value` and `position`. Fold 2 test starts from those values. This simulates continuous live deployment — the agent's wealth compounds (or decays) across the full out-of-sample period. It answers: "If we had deployed this from 2013 to 2020 without stopping, what would have happened?"

The stitched result is what appears in `rl_only_stitched_comparison_metrics.csv` and is the primary headline number.

---

## 16. The Classic Baselines: Buy-and-Hold and Momentum

### Buy-and-Hold
At the start of the evaluation period, all capital is spent buying SPY shares. No further trades. The equity curve is purely driven by SPY's price.

```
shares = initial_amount / price_on_day_0
equity[t] = shares × price[t]
```

This is the hardest baseline to beat because it has zero transaction costs, perfect market exposure from day 1, and never pays for any signal.

### Momentum (126-day)
Holds SPY when the 126-day (approximately 6-month) price return is positive. Exits to cash when it turns negative. Uses yesterday's signal to avoid look-ahead bias (`signal.shift(1)`).

This is a classic trend-following strategy. It typically avoids the worst crash periods at the cost of missing early recoveries.

Both baselines use the same starting capital as the RL agent for each comparison (either per-fold at $10k, or stitched using the first stitched fold's starting capital).

---

## 17. Is the Comparison Fair? What Should We Expect?

**Is it fair?**
Reasonably so. All three strategies see the same prices over the same time period. The RL agent has higher transaction costs (it trades more actively, so it pays TC more often). The classic baselines have zero costs in the stitched version (the implementation applies no TC).

One asymmetry: the RL agent trains on 5 years of historical data before each test year. The classic baselines have no "training" — they use simple rules that work the same everywhere.

**What should we realistically expect?**
- **Bull market years:** Buy-and-Hold is very hard to beat. Long-only exposure + no costs = hard to outperform.
- **Bear or sideways years:** The agent's short position and flat option become valuable.
- **Crash periods:** If the agent learned to go short before a crash, it could significantly outperform B&H.
- **Choppy markets:** Transaction costs hurt — the agent should learn to hold longer.

**The key metric is Sharpe ratio**, not raw return. A lower return with much lower drawdown is often more useful than higher return with bigger crashes.

---

## 18. Output Files

All files are saved to `data/baselines/rl_only/`.

### Per-fold (in `fold1/`, `fold2/`, ...):
| File | Contents |
|------|----------|
| `val_account_value.csv` | date, account_value — equity curve on validation split |
| `val_actions.csv` | Full trade log on validation split |
| `test_account_value.csv` | Equity curve on test split (independent, starts at $10k) |
| `test_actions.csv` | Trade log on test split |
| `test_account_value_stitched.csv` | Stitched equity curve for this fold's test period |
| `test_actions_stitched.csv` | Stitched trade log |

### Aggregate (at `data/baselines/rl_only/`):
| File | Contents |
|------|----------|
| `rl_only_fold_metrics.csv` | Performance metrics for every fold × split combination |
| `rl_only_portfolio_curves.csv` | All fold equity curves combined |
| `rl_only_portfolio_growth.png` | Plot of all fold curves |
| `rl_only_stitched_test_equity.csv` | Single continuous equity curve across all test folds |
| `rl_only_stitched_test_actions.csv` | All stitched trades |
| `rl_only_stitched_comparison.csv` | RL vs B&H vs Momentum on stitched test period |
| `rl_only_stitched_comparison.png` | Plot of stitched comparison |
| `rl_only_stitched_comparison_metrics.csv` | Summary metrics for each strategy on stitched test |
| `rl_only_per_fold_test_comparison.csv` | Per-fold comparison of all three strategies |
| `rl_only_per_fold_test_comparison.png` | Grid plot: each fold's test period, all 3 strategies |
| `rl_only_per_fold_test_metrics.csv` | Metrics per fold × strategy |
| `run_config.json` | Snapshot of all hyperparameters used for this run |

### Key columns in actions files:
| Column | Meaning |
|--------|---------|
| `date` | Trading date |
| `signal` | Agent's raw output ∈ {-1, 0, +1} |
| `actual_action` | What physically happened ∈ {-1, 0, +1} (from Table 1) |
| `position` | Position after the step ∈ {-1, 0, +1} |
| `weight` | Same as position, as float (for metrics compatibility) |
| `trade_price` | SPY price when this step occurred |
| `trade_notional` | Dollar amount on which TC was charged (= account_value when position changed, else 0) |
| `pnl` | Dollar P&L for this step |
| `drawdown` | Running drawdown in dollars from peak |
