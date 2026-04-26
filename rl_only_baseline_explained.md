# RL-Only Baseline: Complete Walkthrough

This document explains every part of `src/baselines/rl_only_baseline.py` in plain language, including how all the pieces connect, how money moves, what each concept means, and what to expect from the results.

---

## Table of Contents

1. [What This File Is Trying To Do](#1-what-this-file-is-trying-to-do)
2. [The Big Picture: How All Pieces Fit Together](#2-the-big-picture-how-all-pieces-fit-together)
3. [Key Vocabulary Before We Start](#3-key-vocabulary-before-we-start)
4. [What is Gym / Gymnasium?](#4-what-is-gym--gymnasium)
5. [What is FinRL?](#5-what-is-finrl)
6. [What is StockTradingEnv?](#6-what-is-stocktradingenv)
7. [The Walk-Forward Folds](#7-the-walk-forward-folds)
8. [Configuration Parameters (RLBaselineConfig)](#8-configuration-parameters-rlbaselineconfig)
9. [How Funding Works — The Portfolio Mechanics](#9-how-funding-works--the-portfolio-mechanics)
10. [The Action Space: Weight Deltas](#10-the-action-space-weight-deltas)
11. [The Observation Space: What the Agent Sees](#11-the-observation-space-what-the-agent-sees)
12. [What is Weight, PnL, and Drawdown?](#12-what-is-weight-pnl-and-drawdown)
13. [The Reward Function](#13-the-reward-function)
14. [What is DQN?](#14-what-is-dqn)
15. [The _DiscreteSingleAssetActionWrapper — The Core of the System](#15-the-_discretesingleassetactionwrapper--the-core-of-the-system)
16. [Function-by-Function Walkthrough](#16-function-by-function-walkthrough)
17. [End-to-End Data Flow: What Happens When You Run the Script](#17-end-to-end-data-flow-what-happens-when-you-run-the-script)
18. [The Two Evaluation Modes: Independent vs Stitched](#18-the-two-evaluation-modes-independent-vs-stitched)
19. [The Classic Baselines: Buy-and-Hold and Momentum](#19-the-classic-baselines-buy-and-hold-and-momentum)
20. [Is the Comparison Fair? What Should We Expect?](#20-is-the-comparison-fair-what-should-we-expect)
21. [Output Files](#21-output-files)

---

## 1. What This File Is Trying To Do

This file trains and evaluates a trading algorithm for SPY (the S&P 500 ETF). The algorithm is called a **DRL agent** (Deep Reinforcement Learning agent). Its job is to look at today's market data and decide how much of its money to put into SPY.

The algorithm is not told in advance "buy when the market goes up" or "sell when RSI is high." Instead, it **learns by trial and error** — it tries different actions, sees whether it made money or lost money, and gradually gets better.

Once trained, it is compared to two simple strategies: Buy-and-Hold (just buy SPY and never touch it) and a Momentum strategy (buy when the trend is up, sell when it's down).

This is called a **baseline** because it will later be compared to the full, more advanced model that uses regime detection and sentiment analysis. If this simple RL agent already beats Buy-and-Hold, that's interesting. If the advanced model beats this, we can quantify how much the extra information helps.

---

## 2. The Big Picture: How All Pieces Fit Together

```
spy_market_data.csv
        |
        v
   load_data()              ← loads the raw daily OHLCV + indicators
        |
        v
  create_walk_forward_split()  ← splits data into train / val / test
        |
        v
  prepare_finrl_dataframe()   ← converts our format → FinRL format
        |
        +-----------+
        |           |
  [train data]  [val/test data]
        |
        v
  train_fold_dqn()
        |
        ├── _make_env_kwargs()          ← configure StockTradingEnv
        ├── StockTradingEnv(...)        ← FinRL builds the base env
        ├── _build_discrete_env()       ← we wrap it with our custom logic
        └── DQN.learn()                 ← DQN trains by playing in the env
              |
              v
           model  (a trained DQN policy)
                |
                v
      evaluate_dqn_on_split()   ← run the trained model on val or test data
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

Everything revolves around an **environment** — a simulated stock market. The agent interacts with it like a game: the environment shows the agent today's market data, the agent picks an action, the environment moves to the next day and tells the agent what happened (reward). This repeats for every trading day in the training period.

---

## 3. Key Vocabulary Before We Start

| Term | Plain English |
|------|---------------|
| **Episode** | One full pass through a dataset (e.g. all of 2007–2011 for fold1 train) |
| **Step** | One trading day — the agent sees today's data, acts, moves to tomorrow |
| **Observation / State** | The numbers the agent sees at each step (price, indicators, position info) |
| **Action** | What the agent decides to do (how much to change its SPY position) |
| **Reward** | The score the agent gets after each action (did it make money?) |
| **Policy** | The learned function: given an observation → pick an action |
| **Weight** | What fraction of total money is invested in SPY (0 = all cash, 1 = fully invested) |
| **PnL** | Profit and Loss — how much money was made or lost |
| **Drawdown** | How far the portfolio has fallen from its all-time high |
| **Fold** | One train/val/test time window (e.g. train 2007–2011, val 2012, test 2013) |
| **Walk-forward** | Testing on data the model has never seen, one fold at a time |

---

## 4. What is Gym / Gymnasium?

**Gym** (now called Gymnasium) is a Python library made by OpenAI. It defines a standard interface for training reinforcement learning agents.

The idea is: any RL environment — whether it's a video game, a robot simulation, or a stock trading simulator — should look the same to the learning algorithm. They all have:

- `env.reset()` → start a new episode, get the initial observation
- `env.step(action)` → take an action, get back (new observation, reward, done?, info)
- `env.observation_space` → describes what the observation looks like (dimensions, min/max values)
- `env.action_space` → describes what actions are valid

This standardization means DQN from the `stable_baselines3` library can work with any Gym-compatible environment, including our stock trading environment, without needing to know anything about trading.

In the code, `gym_module` is either `gymnasium` (the newer name) or `gym` (the older name) — we try both because different versions of FinRL use different ones. The line:

```python
gym = importlib.import_module("gymnasium")
```

loads whichever version is installed.

The **ActionWrapper** class (which our custom wrapper inherits from) is a Gym utility that lets you intercept and transform actions before they reach the base environment. We use it to translate our weight-delta actions into the share quantities that FinRL understands.

---

## 5. What is FinRL?

**FinRL** is a Python library built specifically for financial reinforcement learning. It provides:

1. **A ready-made trading environment** (`StockTradingEnv`) that simulates buying and selling stocks with transaction costs, tracks your portfolio value, handles share counting, etc. Building this correctly from scratch is hard and error-prone.

2. **Data connectors** for pulling market data (we don't use this — we bring our own data).

3. **Integration with standard RL libraries** (like stable-baselines3) via the Gym interface.

**What we cannot easily do without FinRL:**
- Correctly simulating transaction costs being deducted from cash on each trade
- Tracking how many shares you own after each buy/sell
- Computing portfolio value (cash + shares × price) at each step
- Maintaining `asset_memory` (portfolio value history), `date_memory`, `actions_memory` — the logs we use to reconstruct the equity curve after training
- Handling the accounting when you can only buy as many shares as your cash allows
- Starting the next fold's evaluation where the previous fold ended (via `previous_state`)

In short, FinRL handles all the realistic portfolio bookkeeping so we can focus on the learning algorithm and reward design.

---

## 6. What is StockTradingEnv?

`StockTradingEnv` is the specific class inside FinRL that does the portfolio simulation. You give it a DataFrame of market data and a set of parameters, and it becomes a Gym-compatible environment.

**How it works internally:**

- At `reset()`: it sets cash = `initial_amount`, shares = 0, moves to day 0
- At each `step(action)`:
  - Reads today's price
  - Interprets action as: if positive → buy `action × hmax` shares; if negative → sell `|action| × hmax` shares
  - Deducts transaction costs (`buy_cost_pct` or `sell_cost_pct`) from the trade value
  - Updates cash and share count
  - Records portfolio value in `asset_memory`
  - Moves to the next day
  - Returns the new state (cash + all prices + all shares + all tech indicators), reward, done flag

**What it stores:**
- `env.state` — the current state vector: `[cash, price, shares, open, high, low, volume, rsi_14, macd, ...]`
- `env.asset_memory` — list of portfolio values over time (used to build the equity curve)
- `env.date_memory` — list of dates corresponding to each value
- `env.actions_memory` — list of what trades were actually executed each day

**What we override:**
- We replace FinRL's reward with our own (see Section 13)
- We intercept the action to convert weight deltas → share quantities
- We append extra features to the observation (weight, step_return, drawdown)
- We expose a discrete action space instead of FinRL's continuous one

The key insight: StockTradingEnv handles the "plumbing" (money management, share tracking, transaction costs) and we handle the "intelligence" (what the agent learns to do).

---

## 7. The Walk-Forward Folds

The dataset (SPY from ~2007 to 2020) is split into 8 overlapping time windows called **folds**. Each fold has three non-overlapping segments:

```
Fold 1:
  Train: 2007-01-01 → 2011-12-31  (5 years, ~1260 trading days)
  Val:   2012-01-01 → 2012-12-31  (1 year,  ~252 trading days)
  Test:  2013-01-01 → 2013-12-31  (1 year,  ~252 trading days)

Fold 2:
  Train: 2008-01-01 → 2012-12-31
  Val:   2013-01-01 → 2013-12-31
  Test:  2014-01-01 → 2014-12-31
...
```

Each fold slides forward by one year. This is called **walk-forward testing** and it mimics how real strategies are validated:

- **Train**: the model learns here. It sees these prices and learns what to do.
- **Val**: used to check if the model is overfitting. We can use this to tune hyperparameters. (In this baseline, we just record val metrics; no hyperparameter tuning happens automatically.)
- **Test**: the final, untouched measure of performance. The model was never exposed to this data.

**Why not just train on everything and test on the same data?**
Because the model would memorize the training data and look great on it, but fail in the real world. Walk-forward testing simulates the real-world scenario where you train up to some date and then deploy going forward.

**Why 8 folds?**
It gives us 8 different out-of-sample test years (2013–2020), spanning different market conditions: recovery years, bull market, volatility (2018), the COVID crash (2020 partial fold). This makes the results more robust than a single train/test split.

---

## 8. Configuration Parameters (RLBaselineConfig)

The `RLBaselineConfig` dataclass is a frozen (immutable) container for all hyperparameters. Here is what each one does:

### Data & Output
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `input_path` | `data/spy_market_data.csv` | Where to load the market data from |
| `output_dir` | `data/baselines/rl_only` | Where to save results (CSVs, plots) |
| `ticker` | `"SPY"` | Just a label FinRL uses to identify the stock |

### Portfolio Setup
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `initial_amount` | `10,000.0` | Starting money in dollars. Each fold starts with $10,000 cash. |
| `transaction_cost_pct` | `0.001` (0.1%) | Fraction of trade value charged as cost. You buy $1,000 of SPY → $1 fee. |
| `hmax` | `100` | Maximum shares FinRL can move in one step. This is a scaling budget — see Section 10 for how it interacts with weight deltas. |

### Reward Shaping
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `reward_scaling` | `0.0001` | FinRL's internal scaling, passed to the base env but effectively unused since we override the reward. |
| `downside_penalty_alpha` | `1.0` | How hard to penalize losses. Higher = more risk-averse. |
| `trade_penalty_beta` | `0.001` | Fixed penalty for each trade executed. Higher = fewer trades. |

### Action Space
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `action_grid` | `(-1.0, -0.5, 0.0, 0.5, 1.0)` | The five discrete decisions the agent can make (weight deltas, see Section 10) |

### Training
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `train_timesteps` | `50,000` | Total number of environment steps DQN gets to practice during training |
| `seed` | `42` | Random seed for reproducibility |

### DQN Hyperparameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `dqn_learning_rate` | `0.0001` | How fast the neural network updates. Too high → unstable. Too low → slow learning. |
| `dqn_batch_size` | `64` | How many past experiences DQN samples at once to learn from |
| `dqn_buffer_size` | `100,000` | How many past experiences (steps) to keep in memory for sampling |
| `dqn_learning_starts` | `1,000` | DQN waits until it has 1,000 experiences stored before starting to learn |
| `dqn_train_freq` | `4` | DQN learns every 4 environment steps |
| `dqn_target_update_interval` | `500` | How often to copy the main neural network to a "target" network (a stabilization technique) |
| `dqn_exploration_fraction` | `0.1` | First 10% of training steps use random actions (exploration) |
| `dqn_exploration_final_eps` | `0.02` | After exploration phase, 2% of actions remain random (prevents getting stuck) |

---

## 9. How Funding Works — The Portfolio Mechanics

This is crucial to understand, so let's be very explicit.

### Starting State
Every fold starts with: **$10,000 cash, 0 shares of SPY**.

The portfolio is split into two buckets:
- **Cash**: money sitting idle, earning nothing
- **SPY shares**: how many shares you currently hold

**Total account value = cash + (shares × current SPY price)**

### What Happens When You Buy

Say the agent decides to increase its SPY weight by 0.5 (half its total portfolio).

Current state: $10,000 cash, 0 shares, SPY price = $400.
- Current weight = 0 / $10,000 = 0%
- Target weight = 0% + 50% = 50%
- Dollar amount to invest = 50% × $10,000 = $5,000
- Shares to buy = $5,000 / $400 = 12.5 shares

After the trade:
- Cash = $10,000 - $5,000 - ($5,000 × 0.001 transaction cost) = $4,995
- Shares = 12.5
- Account value ≈ $4,995 + 12.5 × $400 = $9,995 (the $5 was eaten by fees)

### What Happens When the Market Moves

Next day, SPY goes up to $404 (+1%).
- Account value = $4,995 + 12.5 × $404 = $4,995 + $5,050 = $10,045
- No trade, no fee. The gains are already in the portfolio.

The 12.5 shares you own are now worth more. That's PnL.

### What Happens When You Sell

Agent decides to reduce weight by 0.5 (sell half the portfolio).
- Current weight = $5,050 / $10,045 = 50.3%
- Target weight = 50.3% - 50% = 0.3% ≈ 0 (capped at 0 since the delta pushes below 0)

Actually, target weight = max(0, 0.503 - 0.5) = 0.003, meaning nearly all shares are sold.

After selling:
- Cash increases by the sale proceeds minus fees
- Shares decrease to near zero

### Do Profits Get Reinvested?

**Yes, automatically.** The portfolio value at any time is cash + shares × price. When you next buy, you compute target allocation as a fraction of this current (possibly larger) total value. So if you turned $10,000 into $12,000 by holding SPY through a rally, your next buy/sell decision is based on the full $12,000.

### Can You Go "Over" the Initial Fund?

No, you can't borrow money or buy on leverage. The weight is always clipped to [0, 1], meaning you can't invest more than 100% of what you have. The code does:

```python
target_weight = float(np.clip(current_weight + weight_delta, 0.0, 1.0))
```

So if you're 80% invested and try to increase by 50%, the target is clipped to 100% — you just buy as much as your remaining cash allows.

You also can't short (go negative). Weight is always ≥ 0.

### The hmax Parameter

When we compute how many shares to trade (delta_shares), we divide by `hmax` to produce a value in [-1, 1] for FinRL's API:

```python
finrl_action = delta_shares / hmax
```

If `delta_shares` is huge (e.g. you're trying to buy 200 shares) but `hmax = 100`, the FinRL action gets clipped to 1.0. This means FinRL only moves `1.0 × hmax = 100` shares per step. So `hmax` acts as a **per-step trade size cap**. In practice, for SPY with $10,000 and prices around $250–$450, you'd typically hold fewer than 100 shares total, so this cap rarely bites in the single-fold independent evaluation. In the stitched evaluation where profits compound across 8 folds, it could start to matter.

### Summary

- You start with all cash, no shares
- You can invest 0% to 100% of your total portfolio in SPY at any time
- Profits stay in the portfolio and compound naturally
- Transaction costs (0.1% per trade) are deducted from cash
- You cannot borrow or short

---

## 10. The Action Space: Weight Deltas

The agent has **5 possible actions** at each step:

| Action Index | Weight Delta | What It Means |
|---|---|---|
| 0 | -1.0 | Sell everything (reduce SPY weight by 100%) |
| 1 | -0.5 | Sell half (reduce SPY weight by 50 percentage points) |
| 2 | 0.0 | Hold (do nothing) |
| 3 | +0.5 | Buy more (increase SPY weight by 50 percentage points) |
| 4 | +1.0 | Buy everything possible (go fully invested) |

These are **changes** in what fraction of your portfolio is in SPY — not absolute targets.

**Example sequence:**
- Day 1: weight = 0%, action = +0.5 → weight becomes 50%
- Day 2: weight = 50%, action = +0.5 → weight becomes 100%
- Day 3: weight = 100%, action = 0.0 → weight stays 100%
- Day 4: weight = 100%, action = -0.5 → weight becomes 50%
- Day 5: weight = 50%, action = -1.0 → weight becomes 0%

Notice how on Day 1 the agent can't go from 0% to 100% in one step — it would take two steps. But going from 100% to 0% in one step is possible because -1.0 reduces any weight to 0%.

**Why deltas instead of absolute targets?**
The description specifies deltas. It makes the policy shift-invariant: "increase my position" is always the same signal regardless of where you currently are.

**How deltas become FinRL actions:**

FinRL's API expects "how many shares to move" expressed as a fraction of hmax. Our wrapper converts:

```python
current_weight = (shares × price) / account_value
target_weight = clip(current_weight + weight_delta, 0, 1)
target_shares = (target_weight × account_value) / price
delta_shares = target_shares - shares_currently_held
finrl_action = clip(delta_shares / hmax, -1, 1)
```

FinRL then executes the trade and deducts costs.

---

## 11. The Observation Space: What the Agent Sees

At each trading day, the agent receives a vector of numbers describing the current state of the world and its own position. This is called the **observation**.

The observation has two parts:

### Part 1: FinRL's Built-in State (14 numbers)
Automatically constructed by StockTradingEnv from the data:

| Index | Value | What it is |
|-------|-------|------------|
| 0 | Cash | How much uninvested money you have (dollars) |
| 1 | Close price | Today's SPY closing price |
| 2 | Shares held | How many SPY shares you currently own |
| 3 | Open | Today's open price |
| 4 | High | Today's high price |
| 5 | Low | Today's low price |
| 6 | Volume | Trading volume today |
| 7 | RSI (14-day) | Relative Strength Index — momentum oscillator |
| 8 | MACD | Moving Average Convergence Divergence — trend indicator |
| 9 | MACD Signal | Smoothed version of MACD |
| 10 | Rolling vol (20-day) | How volatile the stock has been recently |
| 11 | MA (10-day) | 10-day moving average of price |
| 12 | MA (20-day) | 20-day moving average of price |
| 13 | MA (50-day) | 50-day moving average of price |

### Part 2: Our Custom Augmentation (3 numbers)
Added by `_augment_observation()` in our wrapper:

| Index | Value | What it is |
|-------|-------|------------|
| 14 | Current weight | What fraction of the portfolio is currently in SPY (0 to 1) |
| 15 | Step return | Percentage return from yesterday to today (e.g., 0.01 means +1%) |
| 16 | Fractional drawdown | How far you are below your all-time peak, as a fraction (e.g., 0.05 means 5% below peak) |

**Why add these three?**
- **Weight**: Without this, the agent doesn't know its own position. It might try to "sell" when it has nothing to sell.
- **Step return**: Tells the agent whether its current position made money today, providing immediate feedback context.
- **Fractional drawdown**: Tells the agent how "in the hole" it currently is. A large drawdown might signal the agent to reduce risk.

**Why percentage/fractional instead of dollar amounts?**
Dollar values depend on the absolute portfolio size, which changes over time and across folds. Using percentages and fractions means the same signal means the same thing regardless of whether the portfolio is at $8,000 or $15,000.

---

## 12. What is Weight, PnL, and Drawdown?

### Weight
Weight = what fraction of your total money is invested in SPY.

```
Weight = (shares_held × current_price) / total_account_value
```

- Weight = 0.0: All cash, no SPY. Your money is safe but doesn't grow with the market.
- Weight = 1.0: Fully invested, no cash. You gain and lose exactly with SPY.
- Weight = 0.5: Half in SPY, half cash.

The agent's job is fundamentally to decide what weight to hold each day.

### PnL (Profit and Loss)
In the CSV exports (`actions_df`), PnL is the **dollar change** in account value from one day to the next:

```
PnL on day t = account_value(t) - account_value(t-1)
```

- Positive PnL: the portfolio grew that day
- Negative PnL: the portfolio shrank that day

This is a raw dollar number. A $50 gain when your portfolio is $10,000 is very different from a $50 gain when it's $1,000. That's why the **observation** uses the percentage return (`step_return`) instead.

In the observation (what the agent sees), PnL is expressed as:
```
step_return = account_value(t) / account_value(t-1) - 1
```
This is a percentage: 0.01 = +1%, -0.02 = -2%. Scale-invariant across different portfolio sizes.

### Drawdown
Drawdown measures how far the portfolio has fallen from its **all-time peak during this episode**.

In the observation (the `drawdown_frac` the agent sees):
```
running_peak = max(account_value seen so far)
fractional_drawdown = (running_peak - current_value) / running_peak
```

- 0.0: You are at or above your all-time high
- 0.05: You are 5% below your best-ever value
- 0.20: You are 20% below your peak — a significant drawdown

In the CSV exports, drawdown is the raw dollar amount below the running peak (for readability).

In `compute_performance_metrics()`, `max_drawdown` is the worst fractional drawdown over the entire period — the single largest percentage drop from peak. Investors care deeply about this because large drawdowns are psychologically painful and can force selling at the worst time.

---

## 13. The Reward Function

The reward is what the DQN is trained to maximize. It is computed inside the wrapper's `step()` method after every trading day.

### The Formula

```
If a trade was executed (Δw ≠ 0):
    reward = R_t - α × max(0, -R_t)² - β

If no trade was executed (hold):
    reward = R_t - α × max(0, -R_t)²
```

Where:
- **R_t** = `step_return` = today's percentage return (e.g., 0.01 for +1%)
- **α** = `downside_penalty_alpha` (default 1.0)
- **β** = `trade_penalty_beta` (default 0.001)

### Breaking It Down

**R_t (net daily return):** The core signal. If the portfolio grew by 1% today, R_t = 0.01. If it fell by 2%, R_t = -0.02.

**α × max(0, -R_t)²  (downside penalty):** This term only activates when R_t is negative (a loss day). The squaring makes the penalty grow much faster for large losses than small ones:
- -0.5% loss: penalty = 1.0 × (0.005)² = 0.000025 (tiny)
- -2% loss: penalty = 1.0 × (0.02)² = 0.0004 (larger)
- -5% loss: penalty = 1.0 × (0.05)² = 0.0025 (much larger)

This makes the agent more fearful of big crashes than small dips — a desirable property for risk management.

**β (trade penalty):** A fixed cost subtracted whenever any trade is executed (shares actually changed hands, as recorded in `actions_memory`). This discourages churning — excessive buying and selling that racks up transaction costs without improving returns.

### Why Not Just Use R_t?

If the reward were just R_t (daily return), the agent would be indifferent between:
- A strategy that returns +0.5% every day steadily
- A strategy that alternates +10% one day, -9.5% the next (net similar but very volatile)

The downside penalty makes the agent prefer the stable strategy. The trade penalty makes the agent prefer holding over constantly reshuffling.

### What Does the Agent Learn?

Over 50,000 training steps (individual trading days), the DQN sees the consequence of its decisions and adjusts its policy. It learns patterns like:
- "When RSI is high and I'm fully invested, staying invested was bad — I should reduce"
- "When MA_10 is above MA_50 and I'm in cash, I'm missing out"
- "Trading frequently costs me the β penalty — I should hold unless there's a good reason to trade"

---

## 14. What is DQN?

**DQN (Deep Q-Network)** is a reinforcement learning algorithm invented by DeepMind in 2015 (originally used to play Atari games). It uses a neural network to learn which action is best in each situation.

### The Core Idea

DQN learns a function called the **Q-function** (quality function):

```
Q(observation, action) → expected total future reward
```

"If I am in state `observation` and take `action`, what total reward can I expect to collect from now until the end of the episode?"

Over training, the neural network gets better and better at predicting this. Once well-trained, the agent's policy is simple: given an observation, compute Q for all 5 actions, and pick the one with the highest Q value.

### How DQN Learns

1. The agent plays in the environment, collecting experiences: `(observation, action, reward, next_observation)`
2. These experiences are stored in a **replay buffer** (up to `dqn_buffer_size = 100,000` entries)
3. Every `dqn_train_freq = 4` steps, DQN samples a **batch** of 64 random past experiences
4. It uses these to update the neural network weights — trying to make Q predictions more accurate
5. Every `dqn_target_update_interval = 500` steps, a "target network" (a slowly updated copy) is synced to stabilize learning

### Exploration vs Exploitation

During training, DQN uses an ε-greedy strategy:
- First 10% of steps (`dqn_exploration_fraction = 0.1`): 100% random actions — the agent explores without knowing what it's doing
- Gradually, random action probability decays
- After the exploration phase: 2% random (`dqn_exploration_final_eps = 0.02`), 98% greedy (best known action)

This prevents the agent from getting stuck in a local optimum (e.g., always holding) early on.

### Why DQN for This Problem?

Our action space is **discrete** — exactly 5 possible actions (weight deltas). DQN is designed precisely for discrete action spaces. If the action space were continuous (any weight between 0 and 1), we'd use PPO or SAC instead.

### DQN in the Code

```python
model = DQN(
    policy="MlpPolicy",   # "MlpPolicy" = a standard multi-layer neural network
    env=train_env,         # the wrapped FinRL environment
    learning_rate=1e-4,
    batch_size=64,
    buffer_size=100_000,
    ...
)
model.learn(total_timesteps=50_000)
```

After `model.learn()`, the neural network has been updated from 50,000 days of simulated trading experience. During evaluation:

```python
action, _ = model.predict(obs, deterministic=True)
```

`deterministic=True` means no random actions — pure greedy policy. The model picks the single best action it knows.

---

## 15. The `_DiscreteSingleAssetActionWrapper` — The Core of the System

This is the most important class in the file. It sits between FinRL (which handles portfolio mechanics) and DQN (which handles learning), and it does three critical things:

1. **Translates actions**: converts DQN's integer action IDs → weight deltas → share quantities for FinRL
2. **Replaces the reward**: throws away FinRL's default reward and substitutes our custom shaped reward
3. **Augments the observation**: appends weight, step_return, and fractional_drawdown to what the agent sees

Think of it as a translator and augmenter that sits in the middle of the pipeline.

### The Class Hierarchy

```
gym.ActionWrapper
    └── _DiscreteSingleAssetActionWrapper
            └── wraps → StockTradingEnv (the base FinRL env)
```

`gym.ActionWrapper` is a Gym utility that intercepts actions before they reach the base environment. When DQN calls `wrapped_env.step(2)` (action index 2 = hold), the wrapper:
1. Calls `self.action(2)` → converts to a FinRL-compatible float array
2. Calls `self.env.step(that_float_array)` → FinRL executes the trade
3. Receives FinRL's output (obs, reward, done, info)
4. **Ignores** FinRL's reward, computes our custom reward
5. Augments the observation with our three extra features
6. Returns the augmented (obs, custom_reward, done, info) to DQN

### Internal State

The wrapper tracks two values across steps:
- `_previous_account_value`: what the portfolio was worth yesterday (for computing step_return)
- `_running_peak_account_value`: the highest portfolio value ever seen this episode (for computing drawdown_frac)

Both are reset at `reset()` and updated at every `step()`.

### The `action()` Method (Weight Delta → FinRL Share Quantity)

```python
def action(self, action):
    idx = clip(action, 0, 4)           # e.g., index 3 ("+0.5" action)
    weight_delta = grid[idx]            # e.g., 0.5
    
    # Read current portfolio state from FinRL's internal state vector
    cash, price, shares, account_value = self._extract_portfolio_state()
    
    # Compute current and target weight
    current_weight = (shares × price) / account_value
    target_weight = clip(current_weight + weight_delta, 0, 1)
    
    # How many shares do we need to reach target_weight?
    target_shares = (target_weight × account_value) / price
    delta_shares = target_shares - shares  # positive=buy, negative=sell
    
    # Convert to FinRL's [-1, 1] scale
    finrl_action = clip(delta_shares / hmax, -1, 1)
    
    # Save for logging
    self.discrete_action_history.append(idx)
    self.continuous_action_history.append(weight_delta)  # logs the delta, not the FinRL action
    
    return [finrl_action]
```

The weight delta (what the agent decided) is stored in `continuous_action_history` for later export to CSV. The converted finrl_action is what actually goes to StockTradingEnv.

### The `_trade_executed()` Method

```python
def _trade_executed(self):
    actions_memory = self.env.actions_memory
    if actions_memory:
        latest_trade = actions_memory[-1]  # what FinRL actually did
        return any(|trade| > 0)            # True if any shares changed hands
    return False
```

This checks FinRL's `actions_memory` — the record of what was **actually** traded (after accounting for cash constraints, rounding, etc.). It returns False if no trade happened, meaning the β penalty is not applied. Importantly, we do not use the agent's *intended* action (the weight delta) — an agent that wanted to buy but had no cash should not be penalized.

### The `step()` Method

```python
def step(self, action):
    prev_val = self._previous_account_value
    
    # 1. Let FinRL execute the trade (via super().step() which calls self.action(action) first)
    obs, finrl_reward, done, info = super().step(action)
    
    # 2. Get the new portfolio value
    curr_val = self._current_account_value()
    
    # 3. Compute step return (percentage)
    step_return = curr_val / prev_val - 1.0
    
    # 4. Compute our custom reward (ignore finrl_reward)
    downside_penalty = alpha × max(0, -step_return)²
    custom_reward = step_return - downside_penalty
    if self._trade_executed():
        custom_reward -= beta
    
    # 5. Update running peak and compute fractional drawdown
    self._running_peak = max(self._running_peak, curr_val)
    drawdown_frac = (self._running_peak - curr_val) / self._running_peak
    
    # 6. Update previous value for next step
    self._previous_account_value = curr_val
    
    # 7. Return augmented observation + custom reward
    aug_obs = concatenate(obs, [current_weight, step_return, drawdown_frac])
    return aug_obs, custom_reward, done, info
```

---

## 16. Function-by-Function Walkthrough

### `prepare_finrl_dataframe(split_df, ticker)`

**Input**: A DataFrame from `load_data()` with Date as index, columns like Open, High, Low, Close, Volume, rsi_14, macd, etc.

**What it does**:
1. Validates that all required columns exist
2. Resets the date index into a column called "date"
3. Renames columns to lowercase (FinRL requires: open, high, low, close, volume)
4. Adds a "tic" column with value "SPY" (FinRL needs this even for single-asset)
5. Drops rows where any feature is NaN (indicators need warmup periods at the start)
6. Converts dates to strings "YYYY-MM-DD"
7. Sets the integer index as `factorize()[0]` — FinRL uses this to group rows by trading day

**Output**: A clean DataFrame in FinRL format, ready for `StockTradingEnv`

**Why this is necessary**: Our data and FinRL use different column naming conventions. This bridge function ensures compatibility.

---

### `_load_rl_dependencies()`

**What it does**: Lazily imports FinRL, stable-baselines3, and gym/gymnasium. These are only loaded when training actually runs, not when the module is imported.

**Why lazy loading**: FinRL has complex optional dependencies (broker APIs, etc.) that sometimes fail to import even if the core library is installed. The fallback path directly loads the environment file from the filesystem, bypassing the broken import chain.

**Returns**: `(StockTradingEnv class, DQN class, gym module)`

---

### `_make_env_kwargs(config)`

**What it does**: Builds the keyword arguments dictionary for `StockTradingEnv`. This specifies:
- How much starting money (`initial_amount`)
- What the state space dimension is (14 numbers from FinRL's built-in state — the wrapper will add 3 more later)
- Which technical indicators to include
- Transaction cost percentages
- The `hmax` share limit per step

**Important detail about state_space**: The value `14` computed here is what the **base FinRL env** expects. Our wrapper adds 3 more dimensions (weight, step_return, drawdown_frac) on top, making the actual observation the agent sees 17-dimensional. DQN learns from the 17-dimensional space.

---

### `_build_discrete_env(stock_trading_env_cls, gym_module, split_df, env_kwargs, action_grid, ...)`

**What it does**: Creates the full trading environment:
1. Instantiates `StockTradingEnv` (the base FinRL env) with `split_df` and `env_kwargs`
2. Wraps it with `_DiscreteSingleAssetActionWrapper`, which adds discrete actions, custom reward, and augmented observation
3. Returns both `(wrapped_env, base_env)` — the base env is kept separately for extracting `asset_memory` and `date_memory` after evaluation

**Why return both?** The wrapper obscures FinRL's internal state. After an evaluation run, we need to read `base_env.asset_memory` (portfolio values) and `base_env.date_memory` to reconstruct the equity curve. The wrapper doesn't expose these directly.

---

### `_reset_env(env)` and `_step_env(env, action)`

**What they do**: Compatibility helpers. Older gym versions return `obs` from reset; newer gymnasium returns `(obs, info)`. Similarly, older gym's step returns 4 values; gymnasium returns 5. These helpers normalize to 4-return format regardless of which version is installed.

---

### `train_fold_dqn(train_df, config)`

**Input**: A FinRL-formatted DataFrame for the training period, and the config

**What it does**:
1. Loads the RL dependencies
2. Builds the wrapped environment from `train_df`
3. Creates a DQN model with all hyperparameters
4. Calls `model.learn(total_timesteps=50_000)` — this is where the actual training happens

**What training looks like internally**:
- DQN resets the environment (start of 2007 for fold1)
- Steps through days: observe, pick action, get reward, repeat
- Stores each (obs, action, reward, next_obs) in the replay buffer
- Every 4 steps, samples 64 random past experiences and updates the neural network
- When it reaches Dec 2011, the episode ends; it resets to Jan 2007 and starts again
- 50,000 steps ÷ ~1260 trading days per year = roughly 40 complete passes through the training data

**Returns**: `(model, env_kwargs, StockTradingEnv class, gym module)` — everything needed to rebuild an evaluation environment later

---

### `_extract_account_value_frame(base_env, split_df, initial_amount)`

**What it does**: After a completed evaluation run, extracts the portfolio value history from FinRL's memory buffers.

**Primary path**: Read `base_env.date_memory` and `base_env.asset_memory` → zip into a DataFrame → clean duplicates → sort by date

**Fallback path**: If FinRL's memory buffers are empty or missing (shouldn't happen in normal operation), construct a flat DataFrame where account_value = initial_amount for all dates. This is a safeguard, not normal behavior.

**Returns**: A `(date, account_value)` DataFrame — the equity curve

---

### `evaluate_dqn_on_split(model, split_df, env_kwargs, ..., initial, previous_state)`

This is the evaluation function and it has **two modes**.

**Mode 1: `initial=True` (independent evaluation)**
- Starts fresh: cash = $10,000, shares = 0
- Used for the per-fold val and test evaluation in `split_map`
- Each fold is evaluated independently — fold 3's test doesn't know what happened in fold 2

**Mode 2: `initial=False` (stitched evaluation)**
- Starts where the previous fold left off
- `previous_state` contains the terminal state vector from the previous fold (cash balance, shares held, etc.)
- FinRL uses `previous_state` to initialize the portfolio at the start of this fold's data
- This simulates what would happen if you ran the strategy continuously from 2013 to 2020

**Common to both modes**:
1. Build the evaluation environment
2. Reset the environment to get the initial observation
3. Loop: `obs → model.predict(obs, deterministic=True) → action → env.step(action) → new obs`
4. Continue until `done = True` (all trading days exhausted)
5. Extract `account_value_df` (the equity curve)
6. Build `actions_df` with per-day trading detail

**Building actions_df**:
- `discrete_action_history`: the integer action index (0-4) chosen each day
- `continuous_action_history`: the weight delta chosen each day (logged in `action()`)
- `executed_shares`: what FinRL actually traded (from `base_env.actions_memory`)
- `trade_price`: the SPY close price on each trading day
- `trade_notional`: abs(executed_shares) × price (dollar size of each trade)
- `position_after_trade`: cumulative shares after each trade (running total)
- `weight`: position market value / account value
- `pnl`: dollar change in account value (current - previous)
- `drawdown`: dollar distance below the running peak

**Returns**: `(account_value_df, actions_df, terminal_state)` — the equity curve, detailed trade log, and final state to pass to the next fold

---

### `compute_performance_metrics(account_value_df, actions_df)`

**Input**: The equity curve and trade log

**What it computes**:

| Metric | Formula | What it means |
|--------|---------|---------------|
| `total_return` | (final/initial) - 1 | Total gain or loss over the period |
| `annualized_return` | (1 + total_return)^(252/days) - 1 | Return normalized to a yearly basis |
| `annualized_volatility` | std(daily_returns) × √252 | How wildly the returns fluctuate |
| `sharpe_ratio` | annualized_return / annualized_volatility | Return per unit of risk. Higher is better. |
| `sortino_ratio` | annualized_return / downside_volatility | Like Sharpe but only penalizes downside swings |
| `calmar_ratio` | annualized_return / |max_drawdown| | Return per unit of worst drawdown. Higher = better. |
| `max_drawdown` | max(peak/current - 1) | Worst peak-to-trough loss over the period |
| `turnover` | total_traded_notional / avg_account_value | How much the portfolio was turned over |
| `annualized_turnover` | turnover × (252/days) | Yearly trading activity |
| `average_holding_time_days` | avg length of continuous holding runs | How long the agent holds SPY before selling |

---

### `build_stitched_classic_baselines(full_df, stitched_dates, initial_amount, momentum_lookback)`

**What it does**: Constructs two simple comparison strategies over the same dates as the stitched RL curve.

**Buy-and-Hold**:
- Day 1: spend all $10,000 on SPY at the opening price
- Never trade again
- Equity curve = (initial_shares) × (price each day)

**Momentum (126-day)**:
- Check: is today's price higher than it was 126 trading days (≈6 months) ago?
- If yes and you're in cash → buy all in
- If no and you're invested → sell all
- This is a classic trend-following strategy

---

## 17. End-to-End Data Flow: What Happens When You Run the Script

Let's trace a complete execution for fold1 only (`--max-folds 1`):

### Step 1: Parse Arguments and Build Config
```
python src/baselines/rl_only_baseline.py --max-folds 1
```
→ `parse_args()` reads CLI args → `RLBaselineConfig(...)` is created with defaults

### Step 2: Load Data
```python
full_df = load_data(path="data/spy_market_data.csv")
```
→ Returns a DataFrame indexed by Date, with columns: Open, High, Low, Close, Volume, rsi_14, macd, macd_signal, rolling_vol_20, ma_10, ma_20, ma_50

### Step 3: Select Folds
```python
folds = _resolve_folds(None, max_folds=1)
```
→ Returns `[fold1]` only

### Step 4: Split Data for Fold1
```python
train_raw, val_raw, test_raw = create_walk_forward_split(full_df, "2007-01-01", "2011-12-31", ...)
```
→ `train_raw`: SPY data from 2007 to 2011 (~1260 rows)
→ `val_raw`: SPY data for 2012 (~252 rows)
→ `test_raw`: SPY data for 2013 (~252 rows)

### Step 5: Convert to FinRL Format
```python
train_finrl = prepare_finrl_dataframe(train_raw)
val_finrl   = prepare_finrl_dataframe(val_raw)
test_finrl  = prepare_finrl_dataframe(test_raw)
```
→ Lowercase columns, "tic" column added, NaN rows dropped, string dates

### Step 6: Train DQN
```python
model, env_kwargs, StockTradingEnv, gym = train_fold_dqn(train_finrl, config)
```

Inside `train_fold_dqn`:
- `StockTradingEnv(df=train_finrl, initial_amount=10000, hmax=100, ...)` is created
- `_DiscreteSingleAssetActionWrapper` wraps it
- `DQN("MlpPolicy", env=wrapped_env, ...)` is created
- `model.learn(50_000)` runs:
  - Episode 1: Jan 2007 → Dec 2011 (~1260 steps)
  - Episode 2: Jan 2007 → Dec 2011 (~1260 steps again, fresh reset)
  - ... ~40 episodes total
  - At each step: observe 17 numbers → DQN picks action → wrapper translates → FinRL executes → custom reward → DQN learns
- After 50,000 steps, `model` contains a trained neural network

→ Returns the trained model

### Step 7: Evaluate on Val (Independent)
```python
account_df, actions_df, _ = evaluate_dqn_on_split(
    model, val_finrl, env_kwargs, ..., initial=True
)
```
→ Starts with $10,000, 0 shares
→ Runs through all of 2012, one step per trading day
→ Returns equity curve for 2012 (e.g., starts at $10,000, ends at $9,800 or $10,500)
→ Returns actions taken each day

### Step 8: Evaluate on Test (Independent)
Same as above but for 2013. Starts fresh with $10,000.

### Step 9: Compute Metrics
```python
metrics = compute_performance_metrics(account_df, actions_df)
```
→ Returns Sharpe ratio, max drawdown, total return, etc.

### Step 10: Evaluate on Test (Stitched)
```python
stitched_account_df, stitched_actions_df, stitched_state = evaluate_dqn_on_split(
    model, test_finrl, ..., initial=True  # first fold, so initial=True
)
```
→ Also starts with $10,000 for fold1's stitched test
→ `stitched_state` = terminal state at end of 2013

*(For fold2, fold3... `initial=False` and `previous_state=stitched_state` are passed)*

### Step 11: Build Classic Baselines
```python
classic_df = build_stitched_classic_baselines(full_df, stitched_rl_df["date"], 10000)
```
→ Computes Buy-and-Hold and Momentum curves over the same test dates

### Step 12: Save Everything
- `data/baselines/rl_only/rl_only_fold_metrics.csv`: One row per (fold, split) with all metrics
- `data/baselines/rl_only/fold1/val_account_value.csv`: Daily portfolio values for fold1 val
- `data/baselines/rl_only/fold1/val_actions.csv`: Daily trades and positions for fold1 val
- `data/baselines/rl_only/fold1/test_account_value.csv`: Same for test (independent)
- `data/baselines/rl_only/fold1/test_account_value_stitched.csv`: Test (stitched)
- `data/baselines/rl_only/rl_only_stitched_test_equity.csv`: Combined stitched equity across folds
- `data/baselines/rl_only/rl_only_stitched_comparison.csv`: RL vs Buy-Hold vs Momentum
- `data/baselines/rl_only/rl_only_stitched_comparison_metrics.csv`: Metrics for all three
- `data/baselines/rl_only/rl_only_portfolio_growth.png`: Equity curve plot per fold
- `data/baselines/rl_only/rl_only_stitched_comparison.png`: Side-by-side comparison plot
- `data/baselines/rl_only/run_config.json`: The exact config used (for reproducibility)

---

## 18. The Two Evaluation Modes: Independent vs Stitched

### Independent Evaluation (`initial=True`)

Every fold's val and test starts **fresh**: $10,000 cash, 0 shares.

```
Fold1 test: Jan 2013 → Dec 2013, starts at $10,000
Fold2 test: Jan 2014 → Dec 2014, starts at $10,000
Fold3 test: Jan 2015 → Dec 2015, starts at $10,000
...
```

This answers: "How well does this fold's model perform in isolation?"

You can compare folds directly because they all start from the same baseline. The `metrics_df` table summarizes these per-fold results.

**Limitation**: Each fold's test is independent. We can't see how the strategy performs over a multi-year continuous run, because we reset funding each year.

### Stitched Evaluation (`initial=False`)

The test periods are chained together. Fold1's terminal state (how much cash and how many shares remain at end of 2013) becomes the starting point for fold2's evaluation in 2014.

```
Fold1 test: Jan 2013 → Dec 2013, starts at $10,000
  → ends with $11,200 (example), holding 15 shares at $440
Fold2 test: Jan 2014 → Dec 2014, starts with $11,200 and 15 shares
  → ends with $10,900
Fold3 test: Jan 2015 → Dec 2015, starts with $10,900
...
```

This answers: "If we deployed this strategy continuously from 2013 to 2020, switching to a newly trained model each year, what would our portfolio look like?"

The stitched equity curve is what's shown in the comparison plot against Buy-and-Hold and Momentum.

**Important note**: The model changes each year (fold1's model trains on 2007-2011, fold2's trains on 2008-2012, etc.). The stitched evaluation uses each fold's model for exactly one year of test data, which is realistic — in practice, you'd retrain periodically.

---

## 19. The Classic Baselines: Buy-and-Hold and Momentum

### Buy-and-Hold

Buy SPY on the first stitched test date, never sell.

```python
bh_shares = initial_amount / prices.iloc[0]
bh_equity = bh_shares × prices  # for every date
```

This is the **gold standard** that professional fund managers struggle to beat. Historically, SPY returns ~10% per year. Any strategy that underperforms this with more complexity is probably not worth it.

### Momentum (126-day)

A simple trend-following rule: if SPY's price is higher today than it was 126 trading days (~6 months) ago, be invested. Otherwise, be in cash.

```python
signal_today = (price_today > price_126_days_ago)
if signal_today and in_cash:   → buy all in
if not signal_today and invested:  → sell all
```

The signal is computed with a 1-day lag (we observe yesterday's signal and act today) to avoid look-ahead bias.

Momentum captures broad market trends and tends to avoid the worst of bear markets. It's more active than Buy-and-Hold but less active than the RL agent.

---

## 20. Is the Comparison Fair? What Should We Expect?

### Is It Fair?

**Mostly yes, with one important caveat.**

The comparison is fair because:
- All three strategies start with the same $10,000 on the same dates
- All three are evaluated over the same periods (the stitched 2013–2020 test window)
- Buy-and-Hold and Momentum have no access to future data (both use past prices only)
- The RL agent was trained on earlier data (2007–2012 for fold1, 2008–2013 for fold2, etc.) that does not overlap with the test period

The **one caveat**: The RL agent pays transaction costs (0.1% per trade), while our Buy-and-Hold only trades once (negligible cost) and Momentum trades a handful of times per year (also low cost). An actively trading RL agent faces a cost drag that passive strategies don't.

### What Should We Expect?

**Be realistic — this is a baseline, not the final model.** Here's what's likely:

**Against Buy-and-Hold**:
- 2013–2019 was a historic bull market. Buy-and-Hold performed exceptionally well.
- The RL agent will struggle to beat it consistently unless it successfully stays fully invested during rallies and reduces exposure before crashes.
- It will almost certainly **underperform** in terms of raw return because the bull market rewarded continuous exposure, and transaction costs add up.

**Against Momentum**:
- Momentum is a well-known and effective systematic strategy.
- It naturally sits out crash periods (2018 volatility, early COVID) which benefits its Sharpe ratio.
- The RL agent might beat Momentum if it learns to make more nuanced decisions, but momentum's simplicity is a strength in trending markets.

**What "success" looks like for this baseline**:
- Not catastrophically losing money (no ruin)
- A Sharpe ratio > 0.5 (some positive risk-adjusted return)
- Some evidence of market timing (lower weight during bad periods)
- Reasonable turnover (not trading every day)

The **purpose** of this baseline is not to be the best strategy. It's to establish how much DRL alone can do, so we can later quantify the added value of regime detection and sentiment signals in the full model.

**When compared to the full model later**:
- The full model adds regime awareness (hidden Markov model identifying market states) and potentially sentiment
- We expect the full model to have better Sharpe and smaller drawdowns, because it uses more information
- If the full model does NOT outperform this RL baseline, it suggests the extra signals are not being used effectively

### Realistic Performance Expectations (SPY, 2013–2020)

| Metric | Buy-and-Hold (approximate) | Momentum (approximate) | RL Agent (rough expectation) |
|--------|---------------------------|----------------------|------------------------------|
| Total Return (7 yrs) | ~180% | ~100-150% | ~50-150% (wide uncertainty) |
| Sharpe Ratio | ~0.9 | ~0.7-0.9 | ~0.4-0.8 |
| Max Drawdown | ~-34% (COVID) | ~-15% | ~-20 to -35% |
| Turnover | Very low | Low | Moderate to High |

These are rough estimates. RL performance is highly sensitive to hyperparameters and the specific training periods. The first run is unlikely to beat Buy-and-Hold; this is normal and expected for a simple baseline without regime or sentiment features.

---

## 21. Output Files

After a full run, the following files are written to `data/baselines/rl_only/`:

### Per-Fold Files (inside `fold1/`, `fold2/`, etc.)
| File | Contents |
|------|---------|
| `val_account_value.csv` | Date + portfolio value for validation period (independent) |
| `val_actions.csv` | Per-day: action chosen, shares traded, price, position, weight, pnl, drawdown |
| `test_account_value.csv` | Same for test period (independent, fresh $10k start) |
| `test_actions.csv` | Same detailed log for test period |
| `test_account_value_stitched.csv` | Portfolio values for test period (stitched, carries over from previous fold) |
| `test_actions_stitched.csv` | Detailed log for stitched test |

### Aggregate Files (root of output dir)
| File | Contents |
|------|---------|
| `rl_only_fold_metrics.csv` | All metrics for every (fold, split) combination |
| `rl_only_portfolio_curves.csv` | Combined equity curves for all folds/splits |
| `rl_only_portfolio_growth.png` | Line chart: portfolio value per fold |
| `rl_only_stitched_test_equity.csv` | The continuous RL equity curve (2013–2020) |
| `rl_only_stitched_test_actions.csv` | All daily actions across all folds (stitched) |
| `rl_only_stitched_comparison.csv` | RL + Buy-and-Hold + Momentum equity curves |
| `rl_only_stitched_comparison.png` | Side-by-side comparison plot |
| `rl_only_stitched_comparison_metrics.csv` | Final metrics for all three strategies |
| `run_config.json` | The exact config that produced this run (for reproducibility) |
