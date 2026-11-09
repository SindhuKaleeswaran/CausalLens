"""Evaluation metrics for causal treatment-effect estimators.

Metrics include:
- PEHE (Precision in Estimation of Heterogeneous Effect): sqrt(mean((tau_hat - tau_true)^2))
- Absolute ATE error: |ate_hat - ate_true|
- Policy Value: expected outcome under treatment assignment rule pi(x) = I(tau_hat(x) > threshold)
- Policy Risk: V(pi*) - V(pi_hat), the regret relative to oracle assignment
"""
from typing import Any, Dict, Optional, Union
import numpy as np


def _aligned_vectors(*values):
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if any(a.ndim != 1 or not a.size for a in arrays):
        raise ValueError("Expected nonempty one-dimensional vectors")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Shape mismatch between vectors")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected finite vector values")
    return arrays


def pehe_score(
    tau_true: Union[np.ndarray, list],
    tau_pred: Union[np.ndarray, list],
) -> float:
    """Calculate the Precision in Estimation of Heterogeneous Effect (PEHE).

    sqrt(eps_PEHE) = sqrt( 1/n * sum_i (tau_pred_i - tau_true_i)^2 )

    Parameters
    ----------
    tau_true : array-like of shape (n_samples,)
        Ground truth treatment effect (e.g. mu1 - mu0).
    tau_pred : array-like of shape (n_samples,)
        Predicted individual treatment effect from causal model.

    Returns
    -------
    pehe : float
        Root mean squared error between predicted and true treatment effects.
    """
    t_true, t_pred = _aligned_vectors(tau_true, tau_pred)

    return float(np.sqrt(np.mean((t_pred - t_true) ** 2)))


def ate_error(
    ate_true: float,
    ate_pred: float,
) -> float:
    """Calculate the absolute error in Average Treatment Effect estimation.

    |ate_pred - ate_true|

    Parameters
    ----------
    ate_true : float
        Ground truth average treatment effect.
    ate_pred : float
        Estimated average treatment effect.

    Returns
    -------
    error : float
        Absolute error.
    """
    if not (np.isfinite(ate_true) and np.isfinite(ate_pred)):
        raise ValueError("ATE values must be finite")
    return float(np.abs(ate_pred - ate_true))


def policy_value(
    y0_true: Union[np.ndarray, list],
    y1_true: Union[np.ndarray, list],
    tau_pred: Union[np.ndarray, list],
    threshold: float = 0.0,
) -> float:
    """Calculate empirical policy value under the estimated assignment rule.

    Policy: pi(x) = 1 if tau_pred(x) > threshold else 0
    Value: V(pi) = 1/n * sum_i [ pi_i * Y1_i + (1 - pi_i) * Y0_i ]

    Parameters
    ----------
    y0_true : array-like of shape (n_samples,)
        True potential outcomes under control.
    y1_true : array-like of shape (n_samples,)
        True potential outcomes under treatment.
    tau_pred : array-like of shape (n_samples,)
        Predicted treatment effect used to decide treatment.
    threshold : float, default=0.0
        Treatment threshold (treat when expected benefit > threshold).

    Returns
    -------
    val : float
        Expected outcome under policy assignment.
    """
    y0, y1, pred = _aligned_vectors(y0_true, y1_true, tau_pred)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    pi = (pred > threshold).astype(float)
    outcomes = pi * y1 + (1.0 - pi) * y0
    return float(np.mean(outcomes))


def policy_risk(
    y0_true: Union[np.ndarray, list],
    y1_true: Union[np.ndarray, list],
    tau_pred: Union[np.ndarray, list],
    threshold: float = 0.0,
) -> float:
    """Calculate policy regret relative to the optimal oracle policy.

    Risk = V(pi*) - V(pi_hat)
    where pi* = I(Y1 > Y0). The threshold applies to the model rule.

    Parameters
    ----------
    y0_true : array-like of shape (n_samples,)
    y1_true : array-like of shape (n_samples,)
    tau_pred : array-like of shape (n_samples,)
    threshold : float, default=0.0

    Returns
    -------
    risk : float
        Regret (non-negative, 0 indicates optimal decisions).
    """
    y0 = np.asarray(y0_true, dtype=np.float64)
    y1 = np.asarray(y1_true, dtype=np.float64)
    tau_star = y1 - y0

    v_oracle = policy_value(y0, y1, tau_star)
    v_estimated = policy_value(y0, y1, tau_pred, threshold=threshold)
    return float(max(0.0, v_oracle - v_estimated))


def evaluate_estimator(
    estimator_name: str,
    tau_true: np.ndarray,
    tau_pred: np.ndarray,
    ate_true: float,
    ate_pred: float,
    y0_true: Optional[np.ndarray] = None,
    y1_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compile a full dictionary of benchmark performance metrics for an estimator."""
    pehe = pehe_score(tau_true, tau_pred)
    ate_err = ate_error(ate_true, ate_pred)
    res: Dict[str, Any] = {
        "estimator": estimator_name,
        "pehe": round(pehe, 4),
        "ate_error": round(ate_err, 4),
        "ate_pred": round(float(ate_pred), 4),
        "ate_true": round(float(ate_true), 4),
    }
    if y0_true is not None and y1_true is not None:
        res["policy_value"] = round(policy_value(y0_true, y1_true, tau_pred), 4)
        res["policy_risk"] = round(policy_risk(y0_true, y1_true, tau_pred), 4)
    return res
