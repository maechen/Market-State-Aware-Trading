"""
Train a GHMM on SPY log returns and 20-day rolling volatility (train-only)
and label regimes for train/validation/test splits.

This script wires together:
- src.spy.load_market_data: loading and splitting the engineered dataset
- src.regimes.selection: BIC + persistence-based GHMM fitting

Outputs labeled CSVs with regime labels and posterior probabilities.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.spy import load_data, create_walk_forward_split, get_hmm_features
from src.regimes import select_and_fit_ghmm
from configs.walkforward_folds import FOLDS


def _add_regime_columns(
    df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    feature_index: pd.Index,
) -> pd.DataFrame:
    """
    Attach regime labels and posteriors to a DataFrame using the feature index.
    :param df: original DataFrame indexed by Date
    :param labels: integer labels (n_samples,)
    :param probs: posterior probabilities (n_samples, n_components)
    :param feature_index: index of rows used to build X (after dropping NaNs)
    :return: copy of df with new regime columns
    """
    out = df.copy()
    out.loc[feature_index, "regime_label"] = labels
    n_components = probs.shape[1]
    for k in range(n_components):
        out.loc[feature_index, f"regime_prob_{k}"] = probs[:, k]
    return out


def label_splits(
    full_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load splits, fit GHMM on train-only features, and label train/val/test.
    :param full_df: full SPY market DataFrame indexed by Date
    :param train_start: train period start date (YYYY-MM-DD)
    :param train_end: train period end date (YYYY-MM-DD)
    :param val_start: validation period start date
    :param val_end: validation period end date
    :param test_start: test period start date
    :param test_end: test period end date
    :param output_dir: directory where labeled CSVs will be saved
    :return: (train_df_labeled, val_df_labeled, test_df_labeled)
    """
    train_df, val_df, test_df = create_walk_forward_split(
        full_df,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )

    # Build features and fit GHMM on train-only data using get_hmm_features,
    # which applies the standard log_return/rolling_vol_20 dropna pattern.
    X_train = get_hmm_features(train_df)
    if X_train.shape[0] == 0:
        raise ValueError("No valid HMM features in train period after dropping NaNs.")

    model, best_n, best_bic = select_and_fit_ghmm(
        X_train,
        hidden_states=(2, 3, 4),
        min_self_transition=0.8,
        restarts=5,
        random_state=42,
        max_iter=1000,
        tolerance=1e-4,
    )

    transmat = model.transmat_
    diag = np.diag(transmat)
    avg_diag = float(diag.mean())
    min_diag = float(diag.min())
    max_diag = float(diag.max())

    print(
        f"Fitted GHMM with {best_n} states, BIC={best_bic:.2f}, "
        f"self-transition min/avg/max = {min_diag:.3f}/{avg_diag:.3f}/{max_diag:.3f}"
    )

    # Label each split with the same fitted model.
    labeled_splits = []
    split_names = ["train", "val", "test"]
    for name, split_df in zip(split_names, [train_df, val_df, test_df]):
        feature_df = split_df[["log_return", "rolling_vol_20"]].dropna(
            subset=["log_return", "rolling_vol_20"]
        )
        if feature_df.empty:
            print(f"Warning: no valid HMM features in {name} split after dropping NaNs.")
            labeled_splits.append(split_df.copy())
            continue

        X_split = get_hmm_features(split_df)
        labels_split = model.predict(X_split)
        probs_split = model.predict_proba(X_split)
        labeled_df = _add_regime_columns(
            split_df, labels_split, probs_split, feature_df.index
        )
        labeled_splits.append(labeled_df)

    train_labeled, val_labeled, test_labeled = labeled_splits

    # Persist labeled datasets.
    os.makedirs(output_dir, exist_ok=True)
    train_labeled.to_csv(os.path.join(output_dir, "spy_train_labeled.csv"))
    val_labeled.to_csv(os.path.join(output_dir, "spy_val_labeled.csv"))
    test_labeled.to_csv(os.path.join(output_dir, "spy_test_labeled.csv"))

    return (
        train_labeled,
        val_labeled,
        test_labeled,
        best_n,
        best_bic,
        avg_diag,
        min_diag,
        max_diag,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit GHMM on SPY log returns + 20d vol (train-only) and label "
            "regimes for multiple 5y/1y/1y walk-forward folds."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/spy_market_data.csv",
        help="Path to engineered SPY CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training/",
        help="Directory for labeled CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_df = load_data(path=args.input_path)
    fold_stats: list[dict[str, float | str]] = []
    for fold in FOLDS:
        fold_dir = os.path.join(args.output_dir, fold["name"])
        print(
            f"Running {fold['name']}: "
            f"train {fold['train_start']} to {fold['train_end']}, "
            f"val {fold['val_start']} to {fold['val_end']}, "
            f"test {fold['test_start']} to {fold['test_end']}"
        )
        (
            train_labeled,
            val_labeled,
            test_labeled,
            best_n,
            best_bic,
            avg_diag,
            min_diag,
            max_diag,
        ) = label_splits(
            full_df=full_df,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
            output_dir=fold_dir,
        )

        fold_stats.append(
            {
                "fold": fold["name"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "best_n": best_n,
                "best_bic": best_bic,
                "avg_self_transition": avg_diag,
                "min_self_transition": min_diag,
                "max_self_transition": max_diag,
            }
        )

    # Write central summary of fold-level GHMM stats.
    if fold_stats:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "ghmm_fold_summary.csv")
        pd.DataFrame(fold_stats).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()

