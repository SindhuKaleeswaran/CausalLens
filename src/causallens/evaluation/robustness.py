"""Causal refutation and robustness testing suite.

Refutation tests evaluate whether causal conclusions withstand stress testing:
1. Placebo Treatment Refutation: Shuffles treatment assignments. A valid causal model
   should find an estimated effect close to zero.
2. Random Confounder Refutation: Injects an unobserved synthetic random variable.
   A robust estimator should maintain stable treatment effect estimates.
3. Nonparametric Bootstrap Refutation: Resamples data to produce empirical confidence
   intervals and assess estimator stability.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import clone

from causallens.models.base import BaseCausalEstimator


def refute_placebo_treatment(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_simulations: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Test if replacing treatment with an independent random assignment collapses the effect.

    Parameters
    ----------
    model : BaseCausalEstimator
        Fitted or unfitted causal estimator.
    X, T, Y : np.ndarray
        Covariates, treatment, and outcome arrays.
    n_simulations : int, default=10
        Number of placebo permutations to evaluate.
    random_state : int, default=42

    Returns
    -------
    result : dict
        Contains original ATE, mean placebo ATE, standard deviation, and pass/fail verdict.
    """
    rng = np.random.RandomState(random_state)
    original_model = clone(model) if hasattr(model, "get_params") else model.__class__()
    original_model.fit(X, T, Y)
    original_ate = float(original_model.estimate_ate(X))

    placebo_ates: List[float] = []
    n = len(T)

    for _ in range(n_simulations):
        t_perm = rng.permutation(T)
        # Ensure both arms remain present
        if len(np.unique(t_perm)) < 2:
            continue
        placebo_model = model.__class__()
        placebo_model.fit(X, t_perm, Y)
        placebo_ates.append(float(placebo_model.estimate_ate(X)))

    placebo_arr = np.array(placebo_ates)
    mean_placebo = float(np.mean(placebo_arr))
    std_placebo = float(np.std(placebo_arr))

    # Two-sided empirical p-value: proportion of placebos exceeding original effect
    p_value = float(np.mean(np.abs(placebo_arr) >= np.abs(original_ate)))

    # Pass criterion: placebo effect should collapse substantially (relative to original effect)
    passed = bool(abs(mean_placebo) < max(0.5, 0.35 * abs(original_ate)))

    return {
        "test_name": "placebo_treatment_refutation",
        "original_ate": round(original_ate, 4),
        "placebo_ate_mean": round(mean_placebo, 4),
        "placebo_ate_std": round(std_placebo, 4),
        "empirical_p_value": round(p_value, 4),
        "n_simulations": len(placebo_ates),
        "passed": passed,
        "interpretation": (
            "PASSED: Shuffling treatment assignment collapses estimated effect toward zero, "
            "confirming that the model detects a treatment-dependent signal."
            if passed
            else "FAILED: Estimator produces non-trivial effect even when treatment is purely random noise."
        ),
    }


def refute_random_confounder(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_simulations: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Test sensitivity by injecting an independent random noise variable into X.

    Parameters
    ----------
    model : BaseCausalEstimator
    X, T, Y : np.ndarray
    n_simulations : int, default=10
    random_state : int, default=42

    Returns
    -------
    result : dict
    """
    rng = np.random.RandomState(random_state)
    original_model = model.__class__()
    original_model.fit(X, T, Y)
    original_ate = float(original_model.estimate_ate(X))

    perturbed_ates: List[float] = []
    n = len(X)

    for _ in range(n_simulations):
        # Add standard Gaussian random column
        z = rng.normal(size=(n, 1))
        X_perturbed = np.hstack([X, z])
        perturbed_model = model.__class__()
        perturbed_model.fit(X_perturbed, T, Y)
        perturbed_ates.append(float(perturbed_model.estimate_ate(X_perturbed)))

    perturbed_arr = np.array(perturbed_ates)
    mean_perturbed = float(np.mean(perturbed_arr))
    std_perturbed = float(np.std(perturbed_arr))

    # Relative percentage shift from original ATE
    rel_shift = (
        float(abs(mean_perturbed - original_ate) / max(0.1, abs(original_ate))) * 100.0
    )
    passed = bool(rel_shift <= 20.0)

    return {
        "test_name": "random_confounder_refutation",
        "original_ate": round(original_ate, 4),
        "perturbed_ate_mean": round(mean_perturbed, 4),
        "perturbed_ate_std": round(std_perturbed, 4),
        "percent_shift": round(rel_shift, 2),
        "passed": passed,
        "interpretation": (
            f"PASSED: Adding an independent random covariate shifted ATE by only {rel_shift:.1f}%, "
            "demonstrating estimator stability."
            if passed
            else f"FAILED: Estimator was unstable; adding noise shifted ATE by {rel_shift:.1f}%."
        ),
    }


def bootstrap_ate_confidence_interval(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bootstraps: int = 30,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Compute empirical bootstrap confidence interval for the ATE.

    Parameters
    ----------
    model : BaseCausalEstimator
    X, T, Y : np.ndarray
    n_bootstraps : int, default=30
    alpha : float, default=0.05
    random_state : int, default=42

    Returns
    -------
    lower_ci, upper_ci, boot_ates : tuple
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    boot_ates: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both arms present in resampled data
        if len(np.unique(T[idx])) < 2:
            continue
        b_model = model.__class__()
        try:
            b_model.fit(X[idx], T[idx], Y[idx])
            boot_ates.append(float(b_model.estimate_ate(X[idx])))
        except Exception:
            continue

    if len(boot_ates) == 0:
        raise RuntimeError("Bootstrap failed to converge on any resample")

    ates_arr = np.array(boot_ates)
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    lower = float(np.percentile(ates_arr, low_pct))
    upper = float(np.percentile(ates_arr, high_pct))
    return lower, upper, ates_arr
