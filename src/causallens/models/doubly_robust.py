"""Doubly Robust Estimator / Augmented Inverse Probability Weighting (AIPW).

References:
    - Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994).
      Estimation of regression coefficients when some regressors are not always observed.
      Journal of the American Statistical Association, 89(427), 846-866.
    - Chernozhukov, V., et al. (2018).
      Double/debiased machine learning for treatment and structural parameters.
      The Econometrics Journal, 21(1), C1-C68.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV

from causallens.models.base import BaseCausalEstimator


class DoublyRobustEstimator(BaseCausalEstimator):
    """Doubly Robust / Augmented Inverse Probability Weighting (AIPW) Estimator.

    Combines outcome regression models mu0(X), mu1(X) with a propensity score model e(X).
    The doubly robust pseudo-outcome (uncentered influence function) for unit i is:
        Gamma_i = (mu1(Xi) - mu0(Xi))
                + Ti * (Yi - mu1(Xi)) / e(Xi)
                - (1 - Ti) * (Yi - mu0(Xi)) / (1 - e(Xi))

    Double Robustness Guarantee
    ---------------------------
    The ATE estimator is asymptotically unbiased and consistent if EITHER:
    1. The outcome models mu0, mu1 are correctly specified, OR
    2. The propensity score model e(X) is correctly specified.
    """

    def __init__(
        self,
        outcome_estimator: Optional[Any] = None,
        propensity_estimator: Optional[Any] = None,
        effect_estimator: Optional[Any] = None,
        clip_range: Tuple[float, float] = (0.01, 0.99),
    ) -> None:
        super().__init__()
        self.outcome_estimator = (
            outcome_estimator if outcome_estimator is not None else RidgeCV()
        )
        self.propensity_estimator = (
            propensity_estimator
            if propensity_estimator is not None
            else LogisticRegression(max_iter=1000)
        )
        self.effect_estimator = (
            effect_estimator if effect_estimator is not None else RidgeCV()
        )
        self.clip_range = clip_range

        self.mu0_: Any = None
        self.mu1_: Any = None
        self.propensity_model_: Any = None
        self.effect_model_: Any = None
        self.gamma_scores_: Optional[np.ndarray] = None
        self.ate_std_error_: Optional[float] = None

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. No unobserved confounders.",
            "positivity": f"0 < e(X) < 1 for all X. Clipped to {self.clip_range}.",
            "sutva": "No interference between units; single treatment version.",
            "double_robustness": (
                "Consistent if either the outcome models OR the propensity model is correctly specified."
            ),
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "DoublyRobustEstimator":
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        # Step 1: Fit outcome response models mu0(X) and mu1(X)
        self.mu0_ = clone(self.outcome_estimator).fit(X_arr[control_mask], Y_arr[control_mask])
        self.mu1_ = clone(self.outcome_estimator).fit(X_arr[treated_mask], Y_arr[treated_mask])

        mu0_hat = self.mu0_.predict(X_arr)
        mu1_hat = self.mu1_.predict(X_arr)

        # Step 2: Fit propensity model e(X)
        self.propensity_model_ = clone(self.propensity_estimator).fit(X_arr, T_arr)
        if hasattr(self.propensity_model_, "predict_proba"):
            e = self.propensity_model_.predict_proba(X_arr)[:, 1]
        else:
            e = self.propensity_model_.predict(X_arr)

        low, high = self.clip_range
        e = np.clip(e, low, high)

        # Step 3: Compute doubly robust scores (AIPW pseudo-outcomes)
        gamma = (
            (mu1_hat - mu0_hat)
            + (T_arr * (Y_arr - mu1_hat)) / e
            - ((1.0 - T_arr) * (Y_arr - mu0_hat)) / (1.0 - e)
        )
        self.gamma_scores_ = gamma

        # Step 4: Compute ATE and asymptotic standard error
        self.training_ate_ = float(np.mean(gamma))
        n = len(gamma)
        gamma_var = np.var(gamma, ddof=1) if n > 1 else 0.0
        self.ate_std_error_ = float(np.sqrt(gamma_var / n))

        # Step 5: Fit second-stage CATE regression model
        self.effect_model_ = clone(self.effect_estimator).fit(X_arr, gamma)

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = n
        self.training_treated_count_ = int(np.sum(treated_mask))
        self.training_control_count_ = int(np.sum(control_mask))

        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        cate = self.effect_model_.predict(X_arr)
        return np.asarray(cate, dtype=np.float64)

    def estimate_ate(self, X: Optional[Any] = None) -> float:
        self._check_is_fitted()
        if X is None:
            assert self.training_ate_ is not None
            return self.training_ate_
        X_arr, _, _ = self._validate_inputs(X)
        return float(np.mean(self.estimate_ite(X_arr)))

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = np.asarray(self.mu0_.predict(X_arr), dtype=np.float64)
        ite = self.estimate_ite(X_arr)
        y1_pred = y0_pred + ite
        return y0_pred, y1_pred

    def ate_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute asymptotic normal confidence interval for the ATE.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (e.g. 0.05 for 95% confidence).

        Returns
        -------
        lower, upper : tuple of float
        """
        self._check_is_fitted()
        assert self.training_ate_ is not None and self.ate_std_error_ is not None
        from scipy import stats
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        margin = z * self.ate_std_error_
        return (self.training_ate_ - margin, self.training_ate_ + margin)
