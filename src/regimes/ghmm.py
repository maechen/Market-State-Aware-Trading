"""
Gaussian HMM for market regime detection.
Fitted on daily log returns and 20-day rolling volatility.
Outputs regime labels and posteriors.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


def build_observation_matrix(
    log_returns: np.ndarray,
    rolling_vol: np.ndarray
) -> np.ndarray:
    """
    Build the observation matrix X used by the GHMM.
    :param log_returns: shape (n,) or (n, 1)
    :param rolling_vol: rolling std of returns, shape (n,) or (n, 1)
    :return: X, observation matrix of shape (n_samples, 2) with columns [log_return, rolling_vol]
    """
    log_returns = np.asarray(log_returns).ravel()
    rolling_vol = np.asarray(rolling_vol).ravel()

    # alignment
    n = min(len(log_returns), len(rolling_vol))
    log_returns = log_returns[:n]
    rolling_vol = rolling_vol[:n]

    X = np.column_stack([log_returns, rolling_vol])
    # log returns will have a NaN for the first day, rolling volatility will have a NaN for the first 19 days
    # drop NaNs
    X = X[~np.any(np.isnan(X), axis=1)]

    return X


class RegimeGHMM:
    """
    Wrapper around hmmlearn's GaussianHMM for regime detection.
    Expects observation matrix X (n_samples, 2).
    Supports an optional fixed transition matrix to enforce temporal persistence.
    """

    def __init__(
        self,
        hidden_states: int = 3,
        min_self_transition: float = 0.8,
        random_state: int | np.random.RandomState | None = None,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        covariance_type: str = "full"
    ):
        """
        Construct a regime GHMM wrapper.
        :param hidden_states: number of hidden states (e.g. 3 for Bull/Bear/Volatile)
        :param min_self_transition: minimum diagonal value used when fixing the transition matrix
        :param random_state: random state or seed passed to GaussianHMM
        :param max_iter: maximum number of EM iterations
        :param tolerance: convergence tolerance for EM
        :param covariance_type: covariance type for GaussianHMM (e.g. \"full\")
        """
        self.hidden_states = hidden_states
        self.min_self_transition = min_self_transition
        self.random_state = random_state
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.covariance_type = covariance_type
        self._model: GaussianHMM | None = None

    def _build_model(self) -> GaussianHMM:
        """
        Build and constrain the underlying GaussianHMM.
        :return: an unfitted GaussianHMM instance with the configured parameters
        """
        params = "smc" # startprob, means, covars
        
        model = GaussianHMM(
            hidden_states=self.hidden_states,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            random_state=self.random_state,
            params=params
        )

        # initialize with high self‑transition probabilities
        d = self.min_self_transition
        off = (1.0 - d) / (self.n_components - 1) if self.n_components > 1 else 0.0
        model.transmat_ = np.full(
            (self.n_components, self.n_components), off, dtype=float
        )
        np.fill_diagonal(model.transmat_, d)

        return model

    def fit(
        self,
        X: np.ndarray,
        lengths: list[int] | None = None
    ) -> "RegimeGHMM":
        """
        Fit the GHMM to X.
        :param X: observation matrix (n_samples, 2)
        :param lengths: optional list of sequence lengths if X is a concatenation of sequences
        :return: self, the fitted RegimeGHMM instance
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n_samples, 2).")
        self._model = self._build_model()
        self._model.fit(X, lengths=lengths)
        return self

    def predict(
        self,
        X: np.ndarray,
        lengths: list[int] | None = None
    ) -> np.ndarray:
        """
        Predict the most likely regime label for each observation.
        :param X: observation matrix (n_samples, 2)
        :param lengths: optional list of sequence lengths if X is concatenated
        :return: integer regime labels (n_samples,)
        """
        self._check_fitted()
        return self._model.predict(X, lengths=lengths)

    def predict_proba(
        self,
        X: np.ndarray,
        lengths: list[int] | None = None
    ) -> np.ndarray:
        """
        Compute probabilities of regimes for each observation.
        :param X: observation matrix (n_samples, 2)
        :param lengths: optional list of sequence lengths if X is concatenated
        :return: probabilities (n_samples, n_components)
        """
        self._check_fitted()
        return self._model.predict_proba(X, lengths=lengths)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise ValueError("Model not fitted. Call fit(X) first.")

    @property
    def transmat_(self) -> np.ndarray:
        """
        Transition matrix of the GHMM.
        :return: transition matrix (n_components, n_components)
        """
        self._check_fitted()
        return self._model.transmat_

    @classmethod
    def from_fitted(cls, model: GaussianHMM) -> "RegimeGHMM":
        """
        Wrap an already-fitted GaussianHMM instance.
        :param model: fitted GaussianHMM to wrap
        :return: RegimeGHMM wrapper that delegates to the provided model
        """
        wrapper = cls(hidden_states=model.n_components)
        wrapper._model = model
        return wrapper
