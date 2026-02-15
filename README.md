# Market-State-Aware-Trading

Class project for Georgia Tech's CS 4644/7643: Sentiment-Guided Regime-Aware Market Transformer for Alpha Generation

## Project Overview

This project generates alpha on SPY by:

1. Extracting market regimes (Bull, Bear, Volatile) using a Gaussian Hidden Markov Model on daily returns and volatility.
2. Building a sentiment‑guided transformer with cross‑attention gating, trained to predict next‑day direction and HMM regimes, producing a 32‑dimensional market state.
3. Training a deep reinforcement learning agent (DQN) in FinRL with discrete delta actions, a downside‑penalized reward, and transaction costs to dynamically adjust position sizes.
4. Evaluating risk‑adjusted performance via walk‑forward validation, comparing against buy‑and‑hold, momentum, and ablations (no sentiment, no HMM).

## Pace Ice Quick Start

### 0. Load the Anaconda Module
PACE provides Anaconda as a preinstalled module.

```bash
module load anaconda3
```

> Check available versions with:
> ```bash
> module avail anaconda
> ```

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate env_name

# Or use pip (if requirements.txt exists)
pip install -r requirements.txt
```

## Team

Mae Chen, Xiangming Huang, Roopjeet Singh, Aarav Shah
