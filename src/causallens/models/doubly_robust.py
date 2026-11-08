"""Doubly Robust Estimator / Augmented Inverse Probability Weighting (AIPW).

Implements Double/Debiased Machine Learning (DML) cross-fitting following
Chernozhukov et al. (2018) to avoid in-sample nuisance overfitting bias.

Methodological Separation:
    - Population ATE: Cross-fitted AIPW influence function average. Enjoys double
      robustness and root-n asymptotic normality.
    - Individual CATE: Estimated via second-stage regression of AIPW pseudo-outcomes
      Gamma_i on covariates X_i. Point-wise double robustness does NOT automatically
      transfer to CATE predictions without additional non-parametric rate conditions.

References:
    - Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994).
      Estimation of regression coefficients when some regressors are not always observed.
      Journal of the American Statistical Association, 89(427), 846-866.
    - Chernozhukov, V., et al. (2018).
      Double/debiased machine learning for treatment and structural parameters.
      The Econometrics Journal, 21(1), C1-C68.
    - Kennedy, E. H. (2022). Towards optimal doubly robust estimation of heterogeneous
      causal effects. Electronic Journal of Statistics, 16(2), 5808-5849.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold

from causallens.models.base import BaseCausalEstimator


class DoublyRobustEstimator(BaseCausalEstimator):
    """Doubly Robust / Augmented Inverse Probability Weighting (AIPW) Estimator.

    Combines outcome regression models mu0(X), mu1(X) with a propensity score model e(X).
    To ensure root-n consistency and prevent in-sample overfitting bias from flexible
    nuisance learners, K-fold cross-fitting (Chernozhukov et al., 2018) is used:
        Gamma_i = (mu1(Xi) - mu0(Xi))
                + Ti * (Yi - mu1(Xi)) / e(Xi)
                - (1 - Ti) * (Yi - mu0(Xi)) / (1 - e(Xi))
    where mu and e for unit i are evaluated out-of-fold.

    Statistical Guarantees & Boundaries
    -----------------------------------
    1. Population ATE: Root-n consistency and asymptotic normality require strict positivity
       (bounded away from 0 and 1), regularity conditions, and that the product of nuisance
       estimation error rates satisfies ||mu_hat - mu|| * ||e_hat - e|| = o_P(n^{-1/2})
       (e.g., both converge at rate o_P(n^{-1/4})), or that one model class is correctly specified.
       Finite-sample clipping to `clip_range` and residual misspecification introduce bias.
    2. Cross-Fitting: K-fold cross-fitting eliminates first-order bias from in-sample overfitting
       of nuisance parameters. When sample size or class count is smaller than `cv`, the estimator
       falls back to in-sample nuisance fitting with appropriate disclosure.
    3. Individual CATE: Fitted via second-stage pseudo-outcome regression. Double robustness
       does NOT automatically guarantee point-wise unbiasedness for CATE without additional
       non-parametric rate conditions on the second-stage effect learner.
    """

    def __init__(
        self,
        outcome_estimator: Optional[Any] = None,
        propensity_estimator: Optional[Any] = None,
        effect_estimator: Optional[Any] = None,
        clip_range: Tuple[float, float] = (0.01, 0.99),
        cv: int = 5,
        random_state: int = 42,
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
        self.cv = cv
        self.random_state = random_state

        self.mu0_: Any = None
        self.mu1_: Any = None
        self.propensity_model_: Any = None
        self.effect_model_: Any = None
        self.gamma_scores_: Optional[np.ndarray] = None
        self.ate_std_error_: Optional[float] = None
        self.is_cross_fitted_: bool = False

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. Assumes all common causes of T and Y are observed in X.",
            "positivity": f"0 < e(X) < 1 for all X. Overlap required in both arms; clipped to {self.clip_range}.",
            "sutva": "No interference between units; single treatment version.",
            "double_robustness": (
                "Rate-double-robustness: Consistent if either outcome surface models OR propensity model is correctly specified, "
                "or if both nuisance error rates satisfy ||mu_hat - mu|| * ||e_hat - e|| = o_P(n^{-1/2}) with bounded overlap."
            ),
            "population_ate_guarantee": (
                "Root-n asymptotic normality requires bounded positivity, regular nuisance rates, and lack of severe propensity clipping."
            ),
            "cate_regression_boundary": (
                "Second-stage CATE regression estimates E[Gamma|X=x]. Point-wise consistency depends on the effect estimator model class and rate conditions."
            ),
            "cross_fitting": (
                f"{'K-fold cross-fitting (K=' + str(self.cv) + ') reduces first-order nuisance overfitting bias.' if self.is_cross_fitted_ else 'In-sample fitting fallback (cross-fitting disabled or sample size too small).'}"
            ),
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "DoublyRobustEstimator":
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        n = len(X_arr)
        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        n_treated = int(np.sum(treated_mask))
        n_control = int(np.sum(control_mask))

        # Check if stratified K-fold cross-fitting is feasible
        effective_cv = self.cv
        min_class_count = min(n_treated, n_control)

        if effective_cv > 1 and min_class_count >= effective_cv:
            # -------------------------------------------------------------
            # Cross-fitting: Nuisance models evaluate strictly out-of-fold
            # -------------------------------------------------------------
            mu0_oof = np.zeros(n, dtype=np.float64)
            mu1_oof = np.zeros(n, dtype=np.float64)
            e_oof = np.zeros(n, dtype=np.float64)

            skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in skf.split(X_arr, T_arr):
                X_tr, T_tr, Y_tr = X_arr[train_idx], T_arr[train_idx], Y_arr[train_idx]
                X_val = X_arr[val_idx]

                tr_ctrl = T_tr == 0.0
                tr_treat = T_tr == 1.0

                fold_mu0 = clone(self.outcome_estimator).fit(X_tr[tr_ctrl], Y_tr[tr_ctrl])
                fold_mu1 = clone(self.outcome_estimator).fit(X_tr[tr_treat], Y_tr[tr_treat])
                fold_e = clone(self.propensity_estimator).fit(X_tr, T_tr)

                mu0_oof[val_idx] = fold_mu0.predict(X_val)
                mu1_oof[val_idx] = fold_mu1.predict(X_val)

                if hasattr(fold_e, "predict_proba"):
                    e_oof[val_idx] = fold_e.predict_proba(X_val)[:, 1]
                else:
                    e_oof[val_idx] = fold_e.predict(X_val)

            low, high = self.clip_range
            e_oof = np.clip(e_oof, low, high)

            gamma = (
                (mu1_oof - mu0_oof)
                + (T_arr * (Y_arr - mu1_oof)) / e_oof
                - ((1.0 - T_arr) * (Y_arr - mu0_oof)) / (1.0 - e_oof)
            )
            self.is_cross_fitted_ = True
        else:
            # Fallback: In-sample fitting when sample size is too small for K folds
            self.mu0_ = clone(self.outcome_estimator).fit(X_arr[control_mask], Y_arr[control_mask])
            self.mu1_ = clone(self.outcome_estimator).fit(X_arr[treated_mask], Y_arr[treated_mask])
            self.propensity_model_ = clone(self.propensity_estimator).fit(X_arr, T_arr)

            mu0_hat = self.mu0_.predict(X_arr)
            mu1_hat = self.mu1_.predict(X_arr)

            if hasattr(self.propensity_model_, "predict_proba"):
                e_hat = self.propensity_model_.predict_proba(X_arr)[:, 1]
            else:
                e_hat = self.propensity_model_.predict(X_arr)

            low, high = self.clip_range
            e_hat = np.clip(e_hat, low, high)

            gamma = (
                (mu1_hat - mu0_hat)
                + (T_arr * (Y_arr - mu1_hat)) / e_hat
                - ((1.0 - T_arr) * (Y_arr - mu0_hat)) / (1.0 - e_hat)
            )
            self.is_cross_fitted_ = False

        self.gamma_scores_ = gamma

        # Fit full-sample nuisance models for out-of-sample prediction
        self.mu0_ = clone(self.outcome_estimator).fit(X_arr[control_mask], Y_arr[control_mask])
        self.mu1_ = clone(self.outcome_estimator).fit(X_arr[treated_mask], Y_arr[treated_mask])
        self.propensity_model_ = clone(self.propensity_estimator).fit(X_arr, T_arr)

        # 1. Population ATE and asymptotic influence-function standard error
        self.training_ate_ = float(np.mean(gamma))
        gamma_var = np.var(gamma, ddof=1) if n > 1 else 0.0
        self.ate_std_error_ = float(np.sqrt(gamma_var / n))

        # 2. CATE: Fit second-stage CATE regression model on cross-fitted pseudo-outcomes
        self.effect_model_ = clone(self.effect_estimator).fit(X_arr, gamma)

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = n
        self.training_treated_count_ = n_treated
        self.training_control_count_ = n_control

        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        """Estimate Individual Treatment Effect (CATE) using the second-stage pseudo-outcome regression."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        cate = self.effect_model_.predict(X_arr)
        return np.asarray(cate, dtype=np.float64)

    def estimate_ate(self, X: Optional[Any] = None) -> float:
        """Estimate population Average Treatment Effect (ATE)."""
        self._check_is_fitted()
        if X is None:
            assert self.training_ate_ is not None
            return self.training_ate_
        X_arr, _, _ = self._validate_inputs(X)
        return float(np.mean(self.estimate_ite(X_arr)))

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Predict counterfactual potential outcomes (Y(0), Y(1))."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = np.asarray(self.mu0_.predict(X_arr), dtype=np.float64)
        ite = self.estimate_ite(X_arr)
        y1_pred = y0_pred + ite
        return y0_pred, y1_pred

    def ate_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute asymptotic normal confidence interval strictly for the population ATE.

        Warning:
            This interval quantifies uncertainty in the population Average Treatment Effect (ATE).
            It does NOT apply as prediction intervals or point-wise uncertainty for individual
            treatment effect predictions (CATE/ITE).
        """
        self._check_is_fitted()
        assert self.training_ate_ is not None and self.ate_std_error_ is not None
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        margin = z * self.ate_std_error_
        return (float(self.training_ate_ - margin), float(self.training_ate_ + margin))

    def summary(self) -> Dict[str, Any]:
        base_s = super().summary()
        base_s.update({
            "is_cross_fitted": self.is_cross_fitted_,
            "cv_folds": self.cv if self.is_cross_fitted_ else 1,
            "ate_standard_error": self.ate_std_error_,
            "ate_95_ci": self.ate_confidence_interval(alpha=0.05),
            "guarantee_notes": "Double robustness applies to population ATE; CATE is a second-stage regression approximation.",
        })
        return base_s
