"""Causal Meta-Learners: S-Learner, T-Learner, and X-Learner.

References:
    - Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
      Metalearners for estimating heterogeneous treatment effects using machine learning.
      Proceedings of the National Academy of Sciences, 116(10), 4156-4165.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV

from causallens.models.base import BaseCausalEstimator


class SLearner(BaseCausalEstimator):
    """S-Learner (Single Model Meta-Learner).

    Estimates the response surface using a single supervised model:
        mu(x, t) = E[Y | X=x, T=t]
    Individual treatment effect is estimated as:
        tau_hat(x) = mu(x, 1) - mu(x, 0)

    Caveat: Regularization or feature selection may penalize or overlook
    the treatment indicator T when X has many dimensions, shrinking tau towards 0.
    """

    def __init__(self, base_estimator: Optional[Any] = None) -> None:
        super().__init__()
        self.base_estimator = base_estimator if base_estimator is not None else RidgeCV()
        self.model_: Any = None

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. All confounders must be observed in X.",
            "positivity": "0 < P(T=1 | X) < 1 for all X.",
            "sutva": "No interference between units; single treatment version.",
            "model_specification": "Assumes base regression model adequately captures the joint surface E[Y|X, T].",
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "SLearner":
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        # Augment feature matrix with treatment indicator
        XT = np.column_stack([X_arr, T_arr])

        self.model_ = clone(self.base_estimator)
        self.model_.fit(XT, Y_arr)

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = len(X_arr)
        self.training_treated_count_ = int(np.sum(T_arr == 1.0))
        self.training_control_count_ = int(np.sum(T_arr == 0.0))

        # Compute training ATE
        self.training_ate_ = float(np.mean(self.estimate_ite(X_arr)))
        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        n = X_arr.shape[0]

        X_t0 = np.column_stack([X_arr, np.zeros(n, dtype=np.float64)])
        X_t1 = np.column_stack([X_arr, np.ones(n, dtype=np.float64)])

        y0_pred = self.model_.predict(X_t0)
        y1_pred = self.model_.predict(X_t1)
        return np.asarray(y1_pred - y0_pred, dtype=np.float64)

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        n = X_arr.shape[0]

        X_t0 = np.column_stack([X_arr, np.zeros(n, dtype=np.float64)])
        X_t1 = np.column_stack([X_arr, np.ones(n, dtype=np.float64)])

        y0_pred = np.asarray(self.model_.predict(X_t0), dtype=np.float64)
        y1_pred = np.asarray(self.model_.predict(X_t1), dtype=np.float64)
        return y0_pred, y1_pred


class TLearner(BaseCausalEstimator):
    """T-Learner (Two-Model Meta-Learner).

    Fits two separate supervised models for control and treatment groups:
        mu0(x) = E[Y | X=x, T=0]
        mu1(x) = E[Y | X=x, T=1]
    Individual treatment effect is estimated as:
        tau_hat(x) = mu1(x) - mu0(x)

    Prevents treatment effect from being regularized away, but does not share
    data across treatment arms when learning covariate representations.
    """

    def __init__(
        self,
        base_estimator_control: Optional[Any] = None,
        base_estimator_treated: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.base_estimator_control = (
            base_estimator_control if base_estimator_control is not None else RidgeCV()
        )
        self.base_estimator_treated = (
            base_estimator_treated if base_estimator_treated is not None else RidgeCV()
        )
        self.model_control_: Any = None
        self.model_treated_: Any = None

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. All confounders must be observed in X.",
            "positivity": "0 < P(T=1 | X) < 1 for all X.",
            "sutva": "No interference between units; single treatment version.",
            "model_specification": "Assumes both outcome surface models mu0(X) and mu1(X) are well-specified.",
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "TLearner":
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        self.model_control_ = clone(self.base_estimator_control)
        self.model_treated_ = clone(self.base_estimator_treated)

        self.model_control_.fit(X_arr[control_mask], Y_arr[control_mask])
        self.model_treated_.fit(X_arr[treated_mask], Y_arr[treated_mask])

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = len(X_arr)
        self.training_treated_count_ = int(np.sum(treated_mask))
        self.training_control_count_ = int(np.sum(control_mask))

        self.training_ate_ = float(np.mean(self.estimate_ite(X_arr)))
        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = self.model_control_.predict(X_arr)
        y1_pred = self.model_treated_.predict(X_arr)
        return np.asarray(y1_pred - y0_pred, dtype=np.float64)

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = np.asarray(self.model_control_.predict(X_arr), dtype=np.float64)
        y1_pred = np.asarray(self.model_treated_.predict(X_arr), dtype=np.float64)
        return y0_pred, y1_pred


class XLearner(BaseCausalEstimator):
    """X-Learner (Crossover Imputed Effect Meta-Learner).

    Designed by Künzel et al. (2019) specifically for unbalanced treatment arms.
    Algorithm:
    1. Estimate response functions mu0(x) and mu1(x) like a T-Learner.
    2. Impute counterfactual effects:
       - D1 = Y1 - mu0(X1) on treated units
       - D0 = mu1(X0) - Y0 on control units
    3. Fit two effect estimators:
       - tau1(x) predicting D1 from X1
       - tau0(x) predicting D0 from X0
    4. Fit propensity score estimator e(x) = P(T=1 | X=x).
    5. Combine estimates via propensity weighting:
       tau_hat(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x)
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        propensity_estimator: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.base_estimator = base_estimator if base_estimator is not None else RidgeCV()
        self.propensity_estimator = (
            propensity_estimator
            if propensity_estimator is not None
            else LogisticRegression(max_iter=1000)
        )
        self.mu0_: Any = None
        self.mu1_: Any = None
        self.tau0_: Any = None
        self.tau1_: Any = None
        self.propensity_model_: Any = None

    @property
    def assumptions(self) -> Dict[str, str]:
        return {
            "unconfoundedness": "(Y(0), Y(1)) _|_ T | X. All confounders must be observed in X.",
            "positivity": "0 < P(T=1 | X) < 1 for all X. Propensity weights require support in both arms.",
            "sutva": "No interference between units; single treatment version.",
            "crossover_validity": (
                "Imputed counterfactual outcome errors are mean-zero conditionally on X."
            ),
        }

    def fit(self, X: Any, T: Any, Y: Any) -> "XLearner":
        X_arr, T_arr, Y_arr = self._validate_inputs(X, T, Y)
        assert T_arr is not None and Y_arr is not None

        treated_mask = T_arr == 1.0
        control_mask = T_arr == 0.0

        X_treat, Y_treat = X_arr[treated_mask], Y_arr[treated_mask]
        X_ctrl, Y_ctrl = X_arr[control_mask], Y_arr[control_mask]

        # Stage 1: Fit outcome response models
        self.mu0_ = clone(self.base_estimator).fit(X_ctrl, Y_ctrl)
        self.mu1_ = clone(self.base_estimator).fit(X_treat, Y_treat)

        # Stage 2: Compute imputed counterfactual treatment effects
        d1 = Y_treat - self.mu0_.predict(X_treat)
        d0 = self.mu1_.predict(X_ctrl) - Y_ctrl

        # Stage 3: Fit second-stage effect models
        self.tau1_ = clone(self.base_estimator).fit(X_treat, d1)
        self.tau0_ = clone(self.base_estimator).fit(X_ctrl, d0)

        # Stage 4: Fit propensity score estimator
        self.propensity_model_ = clone(self.propensity_estimator).fit(X_arr, T_arr)

        self.fitted_ = True
        self.n_features_in_ = X_arr.shape[1]
        self.training_sample_size_ = len(X_arr)
        self.training_treated_count_ = int(np.sum(treated_mask))
        self.training_control_count_ = int(np.sum(control_mask))

        self.training_ate_ = float(np.mean(self.estimate_ite(X_arr)))
        return self

    def estimate_ite(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)

        # Predict propensity score e(x) = P(T=1 | X=x)
        if hasattr(self.propensity_model_, "predict_proba"):
            propensities = self.propensity_model_.predict_proba(X_arr)[:, 1]
        else:
            propensities = self.propensity_model_.predict(X_arr)
        propensities = np.clip(propensities, 0.001, 0.999)

        tau0_pred = self.tau0_.predict(X_arr)
        tau1_pred = self.tau1_.predict(X_arr)

        # Weighted combination: e(x)*tau0 + (1-e(x))*tau1
        ite = propensities * tau0_pred + (1.0 - propensities) * tau1_pred
        return np.asarray(ite, dtype=np.float64)

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        y0_pred = np.asarray(self.mu0_.predict(X_arr), dtype=np.float64)
        ite = self.estimate_ite(X_arr)
        y1_pred = y0_pred + ite
        return y0_pred, y1_pred
