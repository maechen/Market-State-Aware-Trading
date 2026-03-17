"""
Produce GHMM diagnostics per fold: state occupancy, emission params, switching
behavior, BIC gaps across K, stability across folds, transition matrix, and
per-state summary. Outputs CSVs to a dedicated directory.
"""

import argparse
import os

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM

from src.spy import load_data, create_walk_forward_split, get_hmm_features
from src.regimes import select_and_fit_ghmm
from configs.walkforward_folds import FOLDS

# Match label_regimes fit settings
BIC_K_CANDIDATES = (2, 3, 4, 5)
CHOSEN_K_CANDIDATES = (2, 3, 4)
MIN_SELF_TRANSITION = 0.8
RESTARTS = 5
RANDOM_STATE = 42
MAX_ITER = 1000
TOLERANCE = 1e-4
MIN_COVAR = 1e-2
COVARIANCE_TYPE = "full"


def _run_length_encode(z: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (state, duration) for contiguous runs in z."""
    if len(z) == 0:
        return []
    runs = []
    state, count = int(z[0]), 1
    for t in range(1, len(z)):
        if z[t] == state:
            count += 1
        else:
            runs.append((state, count))
            state, count = int(z[t]), 1
    runs.append((state, count))
    return runs


def _unscale_mean(means_scaled: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray) -> np.ndarray:
    """Unscale per-state means: (K, 2) -> original units."""
    return means_scaled * train_std + train_mean


def _unscale_var(covars_scaled: np.ndarray, train_std: np.ndarray) -> np.ndarray:
    """Unscale diagonal variances from full covars: (K, 2, 2) -> (K, 2) var per feature."""
    # Var in original = covars_scaled[i, j, j] * train_std[j]**2
    var_return = covars_scaled[:, 0, 0] * (train_std[0] ** 2)
    var_vol = covars_scaled[:, 1, 1] * (train_std[1] ** 2)
    return np.column_stack([var_return, var_vol])


def _empirical_mean_var(X: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and weighted variance per state; X (T, 2), gamma (T, K). Returns (K, 2) means, (K, 2) vars."""
    K = gamma.shape[1]
    means = np.zeros((K, 2))
    vars_ = np.zeros((K, 2))
    for i in range(K):
        w = gamma[:, i]
        sw = w.sum()
        if sw < 1e-10:
            continue
        means[i] = np.average(X, axis=0, weights=w)
        vars_[i] = np.average((X - means[i]) ** 2, axis=0, weights=w)
    return means, vars_


def _bic_per_k(
    X: np.ndarray,
    k_candidates: tuple[int, ...],
    restarts: int,
    random_state: int,
) -> dict[int, float]:
    """Fit GHMM for each K (no persistence filter), return best BIC per K."""
    rng = np.random.default_rng(random_state)
    best_bic: dict[int, float] = {k: np.inf for k in k_candidates}
    for n in k_candidates:
        for _ in range(restarts):
            seed = rng.integers(0, 2**31)
            ghmm = GaussianHMM(
                n_components=n,
                covariance_type=COVARIANCE_TYPE,
                min_covar=MIN_COVAR,
                n_iter=MAX_ITER,
                tol=TOLERANCE,
                random_state=seed,
            )
            try:
                ghmm.fit(X)
                bic = ghmm.bic(X)
                if bic < best_bic[n]:
                    best_bic[n] = bic
            except Exception:
                continue
    return best_bic


def run_diagnostics_for_fold(
    fold_name: str,
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    X_train_scaled: np.ndarray,
    export_transition_matrix: bool,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    pd.DataFrame | None,
]:
    """
    Run all diagnostics for one fold. Returns rows for state_occupancy,
    emission_params, switching, bic_by_k, stability_by_vol_rank, duration_histogram,
    per_state_summary; and optional transition_matrix DataFrame for the fold.
    """
    # BIC per K (no persistence)
    bic_per_k = _bic_per_k(X_train_scaled, BIC_K_CANDIDATES, RESTARTS, RANDOM_STATE)
    min_bic = min(bic_per_k.values())
    bic_rows = [
        {"fold": fold_name, "K": k, "BIC": bic_per_k[k], "BIC_gap": bic_per_k[k] - min_bic}
        for k in BIC_K_CANDIDATES
    ]

    # Chosen model (same as label_regimes)
    model, best_n, _ = select_and_fit_ghmm(
        X_train_scaled,
        hidden_states=CHOSEN_K_CANDIDATES,
        min_self_transition=MIN_SELF_TRANSITION,
        restarts=RESTARTS,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
        tolerance=TOLERANCE,
        min_covar=MIN_COVAR,
    )
    K = best_n

    z_vit = model.predict(X_train_scaled)
    gamma = model.predict_proba(X_train_scaled)
    T = len(z_vit)

    # 1) State occupancy
    occ_vit = np.array([(z_vit == i).mean() for i in range(K)])
    occ_post = gamma.mean(axis=0)
    occupancy_rows = [
        {
            "fold": fold_name,
            "state_id": i,
            "occupancy_viterbi": occ_vit[i],
            "occupancy_posterior": occ_post[i],
            "T": T,
        }
        for i in range(K)
    ]

    # 2) Emission parameters (model unscaled + empirical)
    means_scaled = model.means_
    covars_scaled = model.covars_
    mean_orig = _unscale_mean(means_scaled, train_mean, train_std)
    var_orig = _unscale_var(covars_scaled, train_std)
    mean_emp, var_emp = _empirical_mean_var(X_train, gamma)

    emission_rows = [
        {
            "fold": fold_name,
            "state_id": i,
            "mean_return_model": mean_orig[i, 0],
            "mean_vol_model": mean_orig[i, 1],
            "var_return_model": var_orig[i, 0],
            "var_vol_model": var_orig[i, 1],
            "mean_return_emp": mean_emp[i, 0],
            "mean_vol_emp": mean_emp[i, 1],
            "var_return_emp": var_emp[i, 0],
            "var_vol_emp": var_emp[i, 1],
        }
        for i in range(K)
    ]

    # 3) Switching behavior (years covered = date range of rows that have valid HMM features)
    switches = int(np.sum(z_vit[1:] != z_vit[:-1]))
    valid_idx = train_df[["log_return", "rolling_vol_20"]].dropna().index
    dates = valid_idx
    if len(dates) >= 2:
        years_covered = (dates.max() - dates.min()).days / 365.25
    else:
        years_covered = 0.0
    switches_per_year = switches / years_covered if years_covered > 0 else 0.0
    switching_rows = [
        {
            "fold": fold_name,
            "switches": switches,
            "years_covered": years_covered,
            "switches_per_year": switches_per_year,
        }
    ]

    # Duration histogram: run-length encode and list (state_id, duration)
    runs = _run_length_encode(z_vit)
    duration_hist_rows = [
        {"fold": fold_name, "state_id": s, "duration": d} for s, d in runs
    ]

    # 5) Stability: sort states by mean_vol (original), then report rank, mean_vol, mean_return, state_id
    order = np.argsort(mean_orig[:, 1])
    stability_rows = [
        {
            "fold": fold_name,
            "state_rank_by_vol": r,
            "mean_vol": mean_orig[order[r], 1],
            "mean_return": mean_orig[order[r], 0],
            "state_id": int(order[r]),
        }
        for r in range(K)
    ]

    # 6) Transition matrix for this fold (if requested)
    transmat_df = None
    if export_transition_matrix:
        A = model.transmat_
        transmat_df = pd.DataFrame(
            A,
            index=[f"state_{i}" for i in range(K)],
            columns=[f"state_{j}" for j in range(K)],
        )

    # 7) Per-state summary (raw state_id order: occupancy, means, vars, expected_duration)
    transmat = model.transmat_
    expected_duration = np.array([1.0 / (1.0 - transmat[i, i]) if transmat[i, i] < 1.0 else np.nan for i in range(K)])
    per_state_rows = [
        {
            "fold": fold_name,
            "state_id": i,
            "occupancy_viterbi": occ_vit[i],
            "occupancy_posterior": occ_post[i],
            "mean_return": mean_orig[i, 0],
            "mean_vol": mean_orig[i, 1],
            "var_return": var_orig[i, 0],
            "var_vol": var_orig[i, 1],
            "expected_duration": expected_duration[i],
        }
        for i in range(K)
    ]

    return (
        occupancy_rows,
        emission_rows,
        switching_rows,
        bic_rows,
        stability_rows,
        duration_hist_rows,
        per_state_rows,
        transmat_df,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce GHMM diagnostics (occupancy, emissions, switching, BIC, stability, transition matrix, per-state summary)."
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
        default="data/training/ghmm_diagnostics",
        help="Directory for diagnostic CSV outputs.",
    )
    parser.add_argument(
        "--transition-matrix-fold",
        type=int,
        default=1,
        help="Fold number (1-based) for which to export the full KxK transition matrix (e.g. 1 = fold1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_df = load_data(path=args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)

    all_occupancy: list[dict] = []
    all_emission: list[dict] = []
    all_switching: list[dict] = []
    all_bic: list[dict] = []
    all_stability: list[dict] = []
    all_duration_hist: list[dict] = []
    all_per_state: list[dict] = []

    for idx, fold in enumerate(FOLDS):
        fold_name = fold["name"]
        train_df, val_df, test_df = create_walk_forward_split(
            full_df,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )
        X_train = get_hmm_features(train_df)
        if X_train.shape[0] == 0:
            print(f"Warning: no valid HMM features in {fold_name} train, skipping.")
            continue

        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0) + 1e-8
        X_train_scaled = (X_train - train_mean) / train_std

        export_trans = (idx + 1) == args.transition_matrix_fold
        (
            occ_rows,
            em_rows,
            sw_rows,
            bic_rows,
            stab_rows,
            dur_rows,
            per_state_rows,
            transmat_df,
        ) = run_diagnostics_for_fold(
            fold_name=fold_name,
            train_df=train_df,
            X_train=X_train,
            train_mean=train_mean,
            train_std=train_std,
            X_train_scaled=X_train_scaled,
            export_transition_matrix=export_trans,
        )

        all_occupancy.extend(occ_rows)
        all_emission.extend(em_rows)
        all_switching.extend(sw_rows)
        all_bic.extend(bic_rows)
        all_stability.extend(stab_rows)
        all_duration_hist.extend(dur_rows)
        all_per_state.extend(per_state_rows)

        if transmat_df is not None:
            path = os.path.join(args.output_dir, f"transition_matrix_{fold_name}.csv")
            transmat_df.to_csv(path)
            print(f"Wrote {path}")

    pd.DataFrame(all_occupancy).to_csv(
        os.path.join(args.output_dir, "state_occupancy.csv"), index=False
    )
    pd.DataFrame(all_emission).to_csv(
        os.path.join(args.output_dir, "emission_params.csv"), index=False
    )
    pd.DataFrame(all_switching).to_csv(
        os.path.join(args.output_dir, "switching.csv"), index=False
    )
    pd.DataFrame(all_bic).to_csv(
        os.path.join(args.output_dir, "bic_by_k.csv"), index=False
    )
    pd.DataFrame(all_stability).to_csv(
        os.path.join(args.output_dir, "stability_by_vol_rank.csv"), index=False
    )
    pd.DataFrame(all_duration_hist).to_csv(
        os.path.join(args.output_dir, "duration_histogram.csv"), index=False
    )
    pd.DataFrame(all_per_state).to_csv(
        os.path.join(args.output_dir, "per_state_summary.csv"), index=False
    )
    print(f"Wrote all diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
