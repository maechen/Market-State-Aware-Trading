# Market State Aware Trading

Class project for Georgia Tech's CS 4644/7643: Sentiment-Guided Regime-Aware Market Transformer for Alpha Generation

## Project Overview

This project generates alpha on SPY by:

1. Extracting market regimes (Bull, Bear, Volatile) using a Gaussian Hidden Markov Model on daily returns and volatility.
2. Building a sentiment‑guided transformer with cross‑attention gating, trained to predict next‑day direction and HMM regimes, producing a 32‑dimensional market state.
3. Training a deep reinforcement learning agent (DQN) in FinRL with discrete delta actions, a downside‑penalized reward, and transaction costs to dynamically adjust position sizes.
4. Evaluating risk‑adjusted performance via walk‑forward validation, comparing against buy‑and‑hold, momentum, and ablations (no sentiment, no HMM).

## Project Structure

```bash
Market-State-Aware-Trading/
├── data/
│   └── spy_market_data.csv # SPY dataset with OHLCV and features
├── src/
│   ├── spy/
│   │   ├── fetch_spy_data.py # Downloads raw SPY OHLCV from yfinance
│   │   ├── market_data_utils.py # Loads `spy_market_data.csv`
│   │   └── __init__.py
│   ├── regimes/
│   │   ├── ghmm_selection.py # GHMM model selection using BIC with a persistence filter
│   │   └── __init__.py
│   ├── baselines/
│   │   └── nonregime_baseline.py # Simple buy‑and‑hold and momentum baselines for SPY
│   |── sentiment/
│   |   └── sent_analysis.py # Extract sentiment score via FinBert
│   └── __init__.py
├── scripts/
│   └── label_regimes.py # Multi‑fold walk‑forward GHMM training and labelling on SPY
├── configs/
│   └── walkforward_folds.py # Defines the walk‑forward folds.
├── environment.yml
└── README.md
```

## Pace Ice Quick Start

### 0. Load the Anaconda Module

PACE provides Anaconda as a preinstalled module.

```bash
module load anaconda3
```

> Check available versions with:
>
> ```bash
> module avail anaconda
> ```

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dl-project
```

### 2. Train GHMM and Label Regimes

```bash
python -m scripts.label_regimes.py
```

### Troubleshooting
If it's your first time on [Pace Ice](https://discord.com/channels/1466105862912348493/1466115198011052215/1481762866637050041)

If you're running into disk quota issues:
```bash
conda config --add pkgs_dirs /home/hice1/<gt_username>/scratch/conda_pkgs
rm -rf ~/.conda/pkgs/*
```

## Team

Mae Chen, Xiangming Huang, Roopjeet Singh, Aarav Shah
