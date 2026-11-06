"""Inverse Probability of Treatment Weighting (IPW / Propensity Score Weighting).

References:
    - Rosenbaum, P. R., & Rubin, D. B. (1983).
      The central role of the propensity score in observational studies for causal effects.
      Biometrika, 70(1), 41-55.
    - Imbens, G. W., & Rubin, D. B. (2015).
      Causal Inference for Statistics, Social, and Biomedical Sciences.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV

from causallens.models.base import BaseCausalEstimator


class IPWEstimator(BaseCausalEstimator):
    """Inverse Probability of Treatment Weighting (IPW) Estimator.

    Adjusts for confounding by weighting observations inversely proportional
    to their probability of receiving the assigned treatment:
        w_i = T_i / e(X_i) + (1 - T_i) / (1 - e(X_i))

    Includes propensity clipping to prevent variance inflation from extreme weights,
    and uses the normalized Hájek estimator for stable ATE calculation.
    """

    def __init__(
        self,
        propensity_estimator: Optional[Any] = None,
        outcome_estimator: Optional[Any] = None,
        clip_range: Tuple[float, float] = (0.01, 0.99),
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.propensity_estimator = (
            propensity_estimator
            if propensity_estimator is not None
            else LogisticRegression(max_iter=1000)
        )
        self.outcome_estimator = (
            outcome_estimator if outcome_estimator is not None else RidgeCV()
        )
        self.clip_range = clip_range
        self.normalize_weights = normalize_weights

        self.propensity_model_: Any = None
        self.model_y0_: Any = None
        self.model_y1_: Any = None
        self.training_propensities_: Optional[np.ndarray] = None
        self.effective_sample_size_: Optional[float] = None

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. Treatment assignment is conditionally unconfounded.",
            "positivity": f"0 < e(X) < 1 for all X. Clipped to {self.clip_range} to ensure stable weights.",
            "propensity_specification": "Assumes propensity score model P(T=1|X) is correctly specified.",
            "sutva": "No interference between units; single treatment version.",
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "IPWEstimator":
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        # Fit propensity model e(X) = P(T=1 | X)
        self.propensity_model_ = clone(self.propensity_estimator).fit(X_arr, T_arr)
        if hasattr(self.propensity_model_, "predict_proba"):
            e = self.propensity_model_.predict_proba(X_arr)[:, 1]
        else:
            e = self.propensity_model_.predict(X_arr)

        low, high = self.clip_range
        e = np.clip(e, low, high)
        self.training_propensities_ = e

        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        # Weights
        w1 = treated_mask / e
        w0 = control_mask / (1.0 - e)

        # Compute ATE (Normalized Hájek estimator vs Horvitz-Thompson)
        if self.normalize_weights:
            sum_w1 = np.sum(w1)
            sum_w0 = np.sum(w0)
            y1_ate = np.sum(w1 * Y_arr) / sum_w1 if sum_w1 > 0 else 0.0
            y0_ate = np.sum(w0 * Y_arr) / sum_w0 if sum_w0 > 0 else 0.0
        else:
            n = len(X_arr)
            y1_ate = np.sum(w1 * Y_arr) / n
            y0_ate = np.sum(w0 * Y_arr) / n

        self.training_ate_ = float(y1_ate - y0_ate)

        # Effective sample size: (sum w)^2 / sum(w^2)
        total_w = np.where(treated_mask, w1, w0)
        self.effective_sample_size_ = float((np.sum(total_w) ** 2) / np.sum(total_w ** 2))

        # Fit propensity-weighted outcome models for ITE estimation
        w_treated = w1[treated_mask]
        w_control = w0[control_mask]

        self.model_y1_ = clone(self.outcome_estimator)
        self.model_y0_ = clone(self.outcome_estimator)

        # Some regressors accept sample_weight, others do not
        try:
            self.model_y1_.fit(X_arr[treated_mask], Y_arr[treated_mask], sample_weight=w_treated)
        except TypeError:
            self.model_y1_.fit(X_arr[treated_mask], Y_arr[treated_mask])

        try:
            self.model_y0_.fit(X_arr[control_mask], Y_arr[control_mask], sample_weight=w_control)
        except TypeError:
            self.model_y0_.fit(X_arr[control_mask], Y_arr[control_mask])

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = len(X_arr)
        self.training_treated_count_ = int(np.sum(treated_mask))
        self.training_control_count_ = int(np.sum(control_mask))

        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = self.model_y0_.predict(X_arr)
        y1_pred = self.model_y1_.predict(X_arr)
        return np.asarray(y1_pred - y0_pred, dtype=np.float64)

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = np.asarray(self.model_y0_.predict(X_arr), dtype=np.float64)
        y1_pred = np.asarray(self.model_y1_.predict(X_arr), dtype=np.float64)
        return y0_pred, y1_pred

    def get_propensity_scores(self, X: Any) -> np.ndarray:
        """Predict propensity scores P(T=1 | X) for given covariate matrix."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        if hasattr(self.propensity_model_, "predict_proba"):
            e = self.propensity_model_.predict_proba(X_arr)[:, 1]
        else:
            e = self.propensity_model_.predict(X_arr)
        low, high = self.clip_range
        return np.clip(e, low, high)
