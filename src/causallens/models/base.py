"""Base class and common contracts for causal treatment-effect estimators.

All estimators operate under the potential outcomes framework (Neyman-Rubin model)
and require explicit statements of causal identification assumptions:
1. Positivity (Overlap): 0 < P(T=1 | X=x) < 1 for all x in covariate support.
2. Unconfoundedness (Conditional Ignorability): (Y(0), Y(1)) _|_ T | X.
3. Stable Unit Treatment Value Assumption (SUTVA): No interference between units
   and no hidden variations of treatment.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator


class BaseCausalEstimator(BaseEstimator, ABC):
    """Abstract base class for all CausalLens treatment-effect estimators.

    Inherits from scikit-learn's BaseEstimator to provide native get_params/set_params
    and clean cloning capabilities.
    """

    def __init__(self) -> None:
        self.fitted_: bool = False
        self.n_features_in_: int = 0
        self.training_ate_: Optional[float] = None
        self.training_sample_size_: int = 0
        self.training_treated_count_: int = 0
        self.training_control_count_: int = 0

    @property
    @abstractmethod
    def assumptions(self) -> Dict[str, str]:
        """Formal identification assumptions under which estimates have causal validity."""
        pass

    def _validate_inputs(
        self,
        X: Any,
        T: Optional[Any] = None,
        Y: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Validate shapes, numeric dtypes, and finite values for inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        T : array-like of shape (n_samples,), optional
            Binary treatment indicator with values in {0, 1}.
        Y : array-like of shape (n_samples,), optional
            Continuous or binary observed outcome.

        Returns
        -------
        X_arr, T_arr, Y_arr : tuple of np.ndarray
        """
        # Validate X
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array of shape (n_samples, n_features), got shape {X_arr.shape}")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(f"X cannot be empty; got shape {X_arr.shape}")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains non-finite values (NaN or Inf)")

        # Enforce consistent feature count if fitted (F13)
        if T is None and self.fitted_:
            names = getattr(self, "feature_names_in_", None)
            if names is not None and hasattr(X, "columns") and tuple(X.columns) != names:
                raise ValueError("Prediction feature names or order differ from training")
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature count mismatch: estimator was fitted with {self.n_features_in_} features, "
                    f"but received input with {X_arr.shape[1]} features."
                )

        # Validate T if provided
        T_arr = None
        if T is not None:
            T_arr = np.asarray(T, dtype=np.float64)
            if T_arr.ndim != 1:
                if T_arr.ndim == 2 and T_arr.shape[1] == 1:
                    T_arr = T_arr.ravel()
                else:
                    raise ValueError(f"T must be 1D array of shape (n_samples,), got shape {T_arr.shape}")
            if T_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(f"Sample size mismatch: X has {X_arr.shape[0]} rows but T has {T_arr.shape[0]}")
            if not np.all(np.isfinite(T_arr)):
                raise ValueError("T contains non-finite values (NaN or Inf)")
            unique_t = np.unique(T_arr)
            if not np.all(np.isin(unique_t, [0.0, 1.0])):
                raise ValueError(f"T must only contain binary treatment values {{0, 1}}, got values: {unique_t}")
            if len(unique_t) < 2:
                raise ValueError(f"Both treatment and control units must be present; observed only T={unique_t}")

        # Validate Y if provided
        Y_arr = None
        if Y is not None:
            Y_arr = np.asarray(Y, dtype=np.float64)
            if Y_arr.ndim != 1:
                if Y_arr.ndim == 2 and Y_arr.shape[1] == 1:
                    Y_arr = Y_arr.ravel()
                else:
                    raise ValueError(f"Y must be 1D array of shape (n_samples,), got shape {Y_arr.shape}")
            if Y_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(f"Sample size mismatch: X has {X_arr.shape[0]} rows but Y has {Y_arr.shape[0]}")
            if not np.all(np.isfinite(Y_arr)):
                raise ValueError("Y contains non-finite values (NaN or Inf)")

        if T is not None and Y is not None:
            self.feature_names_in_ = tuple(X.columns) if hasattr(X, "columns") else None
        return X_arr, T_arr, Y_arr

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if estimator has not been fitted."""
        if not getattr(self, "fitted_", False):
            raise RuntimeError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' first.")

    @abstractmethod
    def fit(self, X: Any, T: Any, Y: Any) -> "BaseCausalEstimator":
        """Fit estimator to observational data (X, T, Y)."""
        pass

    @abstractmethod
    def estimate_ite(self, X: Any) -> np.ndarray:
        """Estimate Individual / Conditional Average Treatment Effect (CATE/ITE).

        tau(x) = E[Y(1) - Y(0) | X = x]
        """
        pass

    def estimate_ate(self, X: Optional[Any] = None) -> float:
        """Estimate Average Treatment Effect (ATE).

        ATE = E[Y(1) - Y(0)]

        If X is None, returns the estimated ATE over the training sample.
        If X is provided, returns the sample average of estimated ITEs over X.
        """
        self._check_is_fitted()
        if X is None:
            if self.training_ate_ is not None:
                return float(self.training_ate_)
            raise RuntimeError("Training ATE is not available. Please pass X to estimate_ate(X).")
        X_arr, _, _ = self._validate_inputs(X)
        ite = self.estimate_ite(X_arr)
        return float(np.mean(ite))

    def predict_potential_outcomes(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Predict counterfactual potential outcomes (Y(0), Y(1)) for given covariates."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X)
        ite = self.estimate_ite(X_arr)
        return np.zeros_like(ite), ite

    def summary(self) -> Dict[str, Any]:
        """Return structured summary metadata of the fitted model."""
        self._check_is_fitted()
        return {
            "estimator_name": self.__class__.__name__,
            "n_features_in": self.n_features_in_,
            "training_sample_size": self.training_sample_size_,
            "training_treated": self.training_treated_count_,
            "training_control": self.training_control_count_,
            "training_ate": self.training_ate_,
            "assumptions": self.assumptions,
        }
