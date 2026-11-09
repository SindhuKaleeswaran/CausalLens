"""Inverse Probability of Treatment Weighting (IPW) + Weighted Outcome Regression.

Architecture Notice:
    This estimator employs a hybrid causal architecture:
    1. Population ATE: Estimated via normalized Hájek inverse probability weighting
       over the empirical propensity score distribution e(X).
    2. Individual Treatment Effects (CATE/ITE): Estimated via propensity-weighted outcome
       regressions mu1(X; w1) and mu0(X; w0). Standard IPW alone does not predict
       individual counterfactuals without such an outcome modeling layer.

References:
    - Rosenbaum, P. R., & Rubin, D. B. (1983).
      The central role of the propensity score in observational studies for causal effects.
      Biometrika, 70(1), 41-55.
    - Imbens, G. W., & Rubin, D. B. (2015).
      Causal Inference for Statistics, Social, and Biomedical Sciences.
"""
from typing import Any, Dict, Optional, Tuple
from inspect import Parameter, signature
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV

from causallens.models.base import BaseCausalEstimator


class IPWEstimator(BaseCausalEstimator):
    """IPW + Weighted Outcome Regression Estimator.

    Adjusts for confounding by weighting observations inversely proportional
    to their probability of receiving the assigned treatment:
        w_i = T_i / e(X_i) + (1 - T_i) / (1 - e(X_i))

    Methodological Separation:
    - Population ATE: Computed directly via the normalized Hájek estimator:
        tau_ATE_Hajek = sum(w1 * Y) / sum(w1) - sum(w0 * Y) / sum(w0)
    - Heterogeneous CATE: Computed via propensity-weighted outcome regressions
        tau_hat(x) = mu1_hat(x; w1) - mu0_hat(x; w0)
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
        self.overlap_diagnostics_: Optional[Dict[str, Any]] = None
        self.weights_supported_: bool = True

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. Assumes no unobserved confounders.",
            "positivity": f"0 < e(X) < 1 for all X. Clipped to {self.clip_range} to ensure stable weights.",
            "propensity_specification": "Assumes propensity score model P(T=1|X) is correctly specified.",
            "sutva": "No interference between units; single treatment version.",
            "hybrid_cate_assumption": "Individual CATE estimates rely on the correctly weighted outcome regression specification.",
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "IPWEstimator":
        low, high = self.clip_range
        if not 0 < low < high < 1:
            raise ValueError("clip_range must satisfy 0 < low < high < 1")
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        # Fit propensity model e(X) = P(T=1 | X)
        self.propensity_model_ = clone(self.propensity_estimator).fit(X_arr, T_arr)
        if hasattr(self.propensity_model_, "predict_proba"):
            raw_e = self.propensity_model_.predict_proba(X_arr)[:, 1]
        else:
            raw_e = self.propensity_model_.predict(X_arr)

        low, high = self.clip_range
        e = np.clip(raw_e, low, high)
        self.training_propensities_ = e

        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        e_treated = e[treated_mask]
        e_control = e[control_mask]

        # Inverse probability weights
        w1 = treated_mask / e
        w0 = control_mask / (1.0 - e)

        # 1. Population ATE: Normalized Hájek estimator vs Horvitz-Thompson
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

        # Comprehensive overlap diagnostics
        common_low = max(float(np.min(e_treated)), float(np.min(e_control)))
        common_high = min(float(np.max(e_treated)), float(np.max(e_control)))
        overlap_range = max(0.0, common_high - common_low)
        fraction_clipped = float(np.mean((raw_e <= low) | (raw_e >= high)))

        self.overlap_diagnostics_ = {
            "treated_propensity_min": float(np.min(e_treated)),
            "treated_propensity_median": float(np.median(e_treated)),
            "treated_propensity_max": float(np.max(e_treated)),
            "control_propensity_min": float(np.min(e_control)),
            "control_propensity_median": float(np.median(e_control)),
            "control_propensity_max": float(np.max(e_control)),
            "common_support_range": (float(common_low), float(common_high)),
            "fraction_clipped": fraction_clipped,
            "effective_sample_size": self.effective_sample_size_,
            "positivity_warning": bool(fraction_clipped > 0.05 or overlap_range < 0.3),
        }

        # 2. CATE: Fit propensity-weighted outcome regression models for ITE estimation
        w_treated = w1[treated_mask]
        w_control = w0[control_mask]

        self.model_y1_ = clone(self.outcome_estimator)
        self.model_y0_ = clone(self.outcome_estimator)
        params = signature(self.model_y1_.fit).parameters
        self.weights_supported_ = "sample_weight" in params or any(
            p.kind == Parameter.VAR_KEYWORD for p in params.values()
        )
        if not self.weights_supported_:
            warnings.warn(
                f"{type(self.outcome_estimator).__name__} does not accept sample_weight; "
                "falling back to unweighted outcome regressions.", UserWarning, stacklevel=2,
            )
        for regressor, mask, weights in ((self.model_y1_, treated_mask, w_treated),
                                        (self.model_y0_, control_mask, w_control)):
            kwargs = {"sample_weight": weights} if self.weights_supported_ else {}
            regressor.fit(X_arr[mask], Y_arr[mask], **kwargs)

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = len(X_arr)
        self.training_treated_count_ = int(np.sum(treated_mask))
        self.training_control_count_ = int(np.sum(control_mask))

        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        """Estimate Individual Treatment Effect (CATE) using the propensity-weighted outcome regressions."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = self.model_y0_.predict(X_arr)
        y1_pred = self.model_y1_.predict(X_arr)
        return np.asarray(y1_pred - y0_pred, dtype=np.float64)

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Predict counterfactual potential outcomes via propensity-weighted outcome regressions."""
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

    def summary(self) -> Dict[str, Any]:
        base_s = super().summary()
        base_s.update({
            "architecture": "Hybrid: Hájek weighting for ATE + Weighted outcome regression for CATE",
            "ate_estimator": "Normalized Hájek inverse probability weighting",
            "cate_estimator": "Propensity-weighted outcome surface regression",
            "weights_supported": getattr(self, "weights_supported_", True),
            "effective_sample_size": self.effective_sample_size_,
            "overlap_diagnostics": self.overlap_diagnostics_,
        })
        return base_s
