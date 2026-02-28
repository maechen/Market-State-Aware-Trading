"""
BIC-based model selection and persistence filtering for regime GHMM.

Selects number of hidden states via BIC and enforces high self-transition probabilities to avoid rapid regime flipping.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


def satisfies_persistence(
    model: object,
    min_self_transition: float = 0.8,
) -> bool:
    """
    Return True iff all self-transition probabilities are >= min_self_transition.
    :param model: fitted HMM or any object with a transmat_ attribute
    :param min_self_transition: minimum required diagonal value in the transition matrix
    :return: True if min(diag(transmat_)) >= min_self_transition
    """
    transmat = getattr(model, "transmat_", None)

    if transmat is None:
        return False

    return bool(np.min(np.diag(transmat)) >= min_self_transition)


def select_and_fit_ghmm(
    X: np.ndarray,
    hidden_states_candidates: tuple[int, ...] = (2, 3, 4),
    min_self_transition: float = 0.8,
    restarts: int = 5,
    random_state: int | np.random.RandomState | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
    covariance_type: str = "full",
) -> tuple[GaussianHMM, int, float]:
    """
    Fit a GHMM using BIC to choose hidden_states and a persistence filter.
    :param X: observation matrix (n_samples, 2)
    :param hidden_states_candidates: candidate numbers of hidden states to try. 2-4 states are typical (bull, bear, volatile/crash, sideways).
    :param min_self_transition: minimum transmat_ diagonal required to accept a model
    :param restarts: number of random restarts per hidden_states
    :param random_state: random state or seed for reproducibility
    :param max_iter: max EM iterations per fit
    :param tolerance: convergence tolerance for EM
    :param covariance_type: covariance type for GHMM
    :return:
      model: fitted GHMM (best BIC among those passing persistence, or best for 3 states)
      best_n: chosen hidden_states
      best_bic: BIC of the returned model on X
    """
    rng = np.random.default_rng(random_state)
    best_model: GaussianHMM | None = None
    best_n: int = 3
    best_bic: float = np.inf

    for n in hidden_states_candidates:
        # EM can converge to different local optima depending on the starting parameters.
        # trying multiple random initializations to increase the chance of finding a good optimum for each candidate
        for _ in range(restarts):
            seed = rng.integers(0, 2**31)
            ghmm = GaussianHMM(
                hidden_states=n,
                covariance_type=covariance_type,
                max_iter=max_iter,
                tolerance=tolerance,
                random_state=seed,
            )
            ghmm.fit(X)

            # apply the persistence filter 
            if not satisfies_persistence(ghmm, min_self_transition):
                continue

            # BIC penalises the number of parameters for regularization.
            # lower BIC is a better model.
            bic = ghmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
                best_model = ghmm

    if best_model is None:
        # no model passed persistence because the data is very noisy
        # fit with default hidden_states=3 and return the one with the best BIC (no persistence)
        for _ in range(restarts):
            seed = rng.integers(0, 2**31)
            ghmm = GaussianHMM(
                hidden_states=3,
                covariance_type=covariance_type,
                max_iter=max_iter,
                tolerance=tolerance,
                random_state=seed,
            )
            ghmm.fit(X)
            bic = ghmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = 3
                best_model = ghmm

    if best_model is None:
        raise RuntimeError("fit_regime_ghmm: no model could be fitted.")

    return best_model, best_n, best_bic
