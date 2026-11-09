"""Naive observational difference-in-means baseline estimator.

This estimator reflects what a decision-maker observes without causal adjustment.
It highlights the magnitude of confounding / selection bias in observational data.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np

from causallens.models.base import BaseCausalEstimator


class NaiveEstimator(BaseCausalEstimator):
    """Unadjusted Difference-in-Means Estimator.

    Computes the raw difference in sample means between treated and control groups:
        tau_naive = mean(Y | T=1) - mean(Y | T=0)

    Warning
    -------
    This estimator assumes complete random assignment without confounding.
    In observational datasets (such as IHDP), this assumption is strongly violated,
    producing substantial selection bias.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean_y0_: float = 0.0
        self.mean_y1_: float = 0.0
        self.diff_in_means_: float = 0.0

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": (
                "Assumes treatment assignment is completely randomized (T _|_ (Y(0), Y(1))), "
                "ignoring all confounders X. Known to fail drastically in observational data."
            ),
            "positivity": "Assumes at least one treated and one control unit are observed.",
            "sutva": "No interference between units; single treatment version.",
            "heterogeneity": "Assumes homogeneous treatment effect (constant across all units).",
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "NaiveEstimator":
        """Compute sample means on treated and control units."""
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        self.mean_y1_ = float(np.mean(Y_arr[treated_mask]))
        self.mean_y0_ = float(np.mean(Y_arr[control_mask]))
        self.diff_in_means_ = self.mean_y1_ - self.mean_y0_

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = len(X_arr)
        self.training_treated_count_ = int(np.sum(treated_mask))
        self.training_control_count_ = int(np.sum(control_mask))
        self.training_ate_ = self.diff_in_means_

        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        """Return constant treatment effect for all units."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        return np.full(X_arr.shape[0], self.diff_in_means_, dtype=np.float64)

    def estimate_ate(self, X: Optional[Any] = None) -> float:
        """Return raw difference in means."""
        self._check_is_fitted()
        if X is not None:
            self._validate_inputs(X)
        return self.diff_in_means_

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Predict constant potential outcomes based on arm sample means."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        n = X_arr.shape[0]
        y0 = np.full(n, self.mean_y0_, dtype=np.float64)
        y1 = np.full(n, self.mean_y1_, dtype=np.float64)
        return y0, y1
