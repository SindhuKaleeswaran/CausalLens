"""Causal refutation and robustness testing suite.

Refutation and sensitivity tests evaluate whether causal conclusions withstand stress testing:
1. Placebo Treatment Stress Test: Evaluates empirical distribution across repeated treatment
   permutations using finite-sample Monte Carlo p-values (Davison & Hinkley 1997; Phipson & Smyth 2010).
2. Irrelevant Covariate Sensitivity: Injects independent Gaussian noise features into X.
   A well-regularized estimator should maintain stable treatment effect estimates without overfitting noise.
3. Subsample Stability: Evaluates treatment effect stability across random sub-cohorts.
4. Estimator Disagreement: Quantifies the discrepancy across different causal modeling paradigms.
5. Formal Omitted Variable Sensitivity: Implements the exact Cinelli & Hazlett (2020 JRSS-B)
   Robustness Value (RV) framework on a partially linear regression specification.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.base import clone

from causallens.models.base import BaseCausalEstimator


def refute_placebo_treatment(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_simulations: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Exploratory placebo stress test evaluating effect collapse under treatment permutation.

    Methodology: Permute T repeatedly without replacement, refit the model on (X, T_perm, Y),
    and analyze the resulting empirical distribution of placebo ATE estimates.
    The empirical p-value uses the exact finite-sample Monte Carlo formula
    (Davison & Hinkley 1997; Phipson & Smyth 2010):
        p_MC = (1 + sum(abs(placebo_ate) >= abs(original_ate))) / (1 + B)
    which strictly avoids reporting an impossible p = 0 under finite permutations.

    Important Note on Calibrated Testing:
    Unconditional permutation breaks P(T|X) and disrupts observational confounding.
    This diagnostic serves as an exploratory stress test evaluating whether the model
    detects treatment-dependent variation, rather than a formally calibrated hypothesis test.
    """
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    rng = np.random.RandomState(random_state)
    original_model = clone(model)
    original_model.fit(X, T, Y)
    original_ate = float(original_model.estimate_ate(X))

    placebo_ates: List[float] = []

    for _ in range(n_simulations):
        t_perm = rng.permutation(T)
        if len(np.unique(t_perm)) < 2:
            continue
        placebo_model = clone(model)
        placebo_model.fit(X, t_perm, Y)
        placebo_ates.append(float(placebo_model.estimate_ate(X)))

    b_actual = len(placebo_ates)
    if b_actual == 0:
        raise RuntimeError("All permutation trials resulted in degenerate treatment assignments")

    placebo_arr = np.array(placebo_ates)
    mean_placebo = float(np.mean(placebo_arr))
    std_placebo = float(np.std(placebo_arr))
    median_placebo = float(np.median(placebo_arr))
    p05 = float(np.percentile(placebo_arr, 5))
    p95 = float(np.percentile(placebo_arr, 95))

    # Exact finite-sample Monte Carlo p-value with Davison-Hinkley continuity correction
    n_extreme = int(np.sum(np.abs(placebo_arr) >= np.abs(original_ate)))
    mc_p_value = float((1.0 + n_extreme) / (1.0 + b_actual))

    # Pass criterion: empirical distribution is centered near zero relative to original effect
    # and either no permutation exceeded original effect (finite B) or Monte Carlo p-value <= 0.15
    passed = bool(
        abs(mean_placebo) < max(0.5, 0.35 * abs(original_ate))
        and (n_extreme == 0 or mc_p_value <= 0.15)
    )

    return {
        "test_name": "placebo_treatment_stress_test",
        "methodology": "Random permutation of treatment indicator vector across repeated Monte Carlo trials.",
        "assumptions": "Under true treatment dependence, shuffling treatment assignments breaks the relationship with outcomes.",
        "original_ate": round(original_ate, 4),
        "placebo_ate_mean": round(mean_placebo, 4),
        "placebo_ate_median": round(median_placebo, 4),
        "placebo_ate_std": round(std_placebo, 4),
        "placebo_ci_90": (round(p05, 4), round(p95, 4)),
        "n_permutations": b_actual,
        "n_extreme_permutations": n_extreme,
        "monte_carlo_p_value": round(mc_p_value, 4),
        "minimum_possible_p_value": round(1.0 / (1.0 + b_actual), 4),
        "passed": passed,
        "interpretation": (
            f"PASSED: Permuting treatment drops estimated ATE from {original_ate:.2f} to {mean_placebo:.2f} "
            f"(Monte Carlo p-value = {mc_p_value:.4f} across {b_actual} trials). "
            "Confirms that the model captures a treatment-dependent signal."
            if passed
            else f"WARNING: Estimator produces non-trivial effect ({mean_placebo:.2f}) even under randomized treatment."
        ),
        "limitations": (
            "Exploratory stress test only. Unconditional treatment permutation breaks the observational assignment mechanism P(T|X) "
            "and does not constitute formal proof of unconfoundedness."
        ),
    }


def refute_irrelevant_covariate(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_simulations: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Test estimator stability when independent noise covariates are added to X.

    Methodology: Injects an independent standard Gaussian random feature into X and measures
    the shift in estimated ATE. This tests whether the estimator overfits irrelevant noise.
    """
    rng = np.random.RandomState(random_state)
    original_model = clone(model)
    original_model.fit(X, T, Y)
    original_ate = float(original_model.estimate_ate(X))

    perturbed_ates: List[float] = []
    n = len(X)

    for _ in range(n_simulations):
        z = rng.normal(size=(n, 1))
        X_perturbed = np.hstack([X, z])
        perturbed_model = clone(model)
        perturbed_model.fit(X_perturbed, T, Y)
        perturbed_ates.append(float(perturbed_model.estimate_ate(X_perturbed)))

    perturbed_arr = np.array(perturbed_ates)
    mean_perturbed = float(np.mean(perturbed_arr))
    std_perturbed = float(np.std(perturbed_arr))

    rel_shift = (
        float(abs(mean_perturbed - original_ate) / max(0.1, abs(original_ate))) * 100.0
    )
    passed = bool(rel_shift <= 15.0)

    return {
        "test_name": "irrelevant_covariate_sensitivity",
        "methodology": "Addition of an independent random Gaussian noise feature to X.",
        "assumptions": "A well-regularized causal estimator should not significantly alter its effect estimate when uninformative noise is introduced.",
        "original_ate": round(original_ate, 4),
        "perturbed_ate_mean": round(mean_perturbed, 4),
        "perturbed_ate_std": round(std_perturbed, 4),
        "percent_shift": round(rel_shift, 2),
        "passed": passed,
        "interpretation": (
            f"PASSED: Adding an irrelevant noise covariate shifted the estimated ATE by only {rel_shift:.1f}%, "
            "demonstrating stability against uninformative features."
            if passed
            else f"WARNING: Estimator showed sensitivity to noise, shifting ATE by {rel_shift:.1f}%."
        ),
        "limitations": "This tests sensitivity to irrelevant noise features, NOT unobserved confounding.",
    }


def refute_subsample_stability(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    subsample_fraction: float = 0.8,
    n_simulations: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Test stability of the estimator across random sub-cohorts (subsampling without replacement)."""
    rng = np.random.RandomState(random_state)
    original_model = clone(model)
    original_model.fit(X, T, Y)
    original_ate = float(original_model.estimate_ate(X))

    n = len(X)
    sub_size = int(n * subsample_fraction)
    sub_ates: List[float] = []

    for _ in range(n_simulations):
        idx = rng.choice(n, size=sub_size, replace=False)
        if len(np.unique(T[idx])) < 2:
            continue
        sub_model = clone(model)
        try:
            sub_model.fit(X[idx], T[idx], Y[idx])
            sub_ates.append(float(sub_model.estimate_ate(X[idx])))
        except Exception:
            continue

    sub_arr = np.array(sub_ates)
    mean_sub = float(np.mean(sub_arr))
    std_sub = float(np.std(sub_arr))
    rel_shift = float(abs(mean_sub - original_ate) / max(0.1, abs(original_ate))) * 100.0
    passed = bool(rel_shift <= 20.0 and std_sub < 1.0)

    return {
        "test_name": "subsample_stability",
        "methodology": f"Random subsampling at {int(subsample_fraction*100)}% sample size across {n_simulations} iterations.",
        "assumptions": "Consistent causal estimators should yield stable estimates across representative subsamples.",
        "original_ate": round(original_ate, 4),
        "subsample_ate_mean": round(mean_sub, 4),
        "subsample_ate_std": round(std_sub, 4),
        "percent_shift": round(rel_shift, 2),
        "passed": passed,
        "interpretation": (
            f"PASSED: Subsample mean ATE ({mean_sub:.2f}) closely aligns with full sample ({original_ate:.2f}), "
            f"with low standard deviation ({std_sub:.3f})."
            if passed
            else f"WARNING: Estimator fluctuates substantially across subsamples (std = {std_sub:.3f})."
        ),
        "limitations": "Evaluates sample variance; cannot detect systematic biases present in the entire cohort.",
    }


def compute_estimator_disagreement(
    models: Dict[str, BaseCausalEstimator],
    X: np.ndarray,
) -> Dict[str, Any]:
    """Measure disagreement across different causal estimators on the same evaluation cohort."""
    predictions: Dict[str, np.ndarray] = {}
    ates: Dict[str, float] = {}

    for name, m in models.items():
        if getattr(m, "fitted_", False):
            ite = m.estimate_ite(X)
            predictions[name] = ite
            ates[name] = float(np.mean(ite))

    if len(predictions) < 2:
        return {"error": "At least two fitted estimators required to compute disagreement"}

    ate_values = list(ates.values())
    ate_spread = float(np.max(ate_values) - np.min(ate_values))
    ate_std = float(np.std(ate_values))

    # Mean pairwise difference in individual treatment effect estimates
    names = list(predictions.keys())
    pairwise_diffs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = float(np.mean(np.abs(predictions[names[i]] - predictions[names[j]])))
            pairwise_diffs.append(diff)

    mean_ite_disagreement = float(np.mean(pairwise_diffs))
    passed = bool(ate_spread < 1.5)

    return {
        "test_name": "estimator_disagreement",
        "methodology": "Pairwise comparison of ATE and CATE estimates across distinct causal modeling architectures.",
        "assumptions": "If identification assumptions hold and models are well-specified, effect estimates should roughly agree.",
        "ate_by_model": {k: round(v, 4) for k, v in ates.items()},
        "ate_spread": round(ate_spread, 4),
        "ate_std": round(ate_std, 4),
        "mean_ite_disagreement": round(mean_ite_disagreement, 4),
        "passed": passed,
        "interpretation": (
            f"Estimator agreement is high (ATE spread: {ate_spread:.2f}, ITE mean absolute diff: {mean_ite_disagreement:.2f})."
            if passed
            else f"High estimator disagreement (ATE spread: {ate_spread:.2f}). Model choice materially influences conclusions."
        ),
        "limitations": "Model consensus is not proof of causal truth; shared violations of unconfoundedness bias all models similarly.",
    }


def sensitivity_unobserved_confounding(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Formal Cinelli & Hazlett (2020) Omitted Variable Sensitivity Analysis on Linear Specification.

    Computes the exact Robustness Value (RV) for the treatment effect in the linear model:
        Y = tau * T + X * beta + epsilon

    References:
        Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted
        variable bias to more than one confounder. Journal of the Royal Statistical
        Society Series B: Statistical Methodology, 82(1), 39-67.

    Formulas:
        t = tau_hat_ols / SE(tau_hat_ols)
        df = n - k - 2 (where k is number of covariates in X)
        f_q = |t| / sqrt(df)
        RV_{q=1} = 0.5 * ( sqrt(f_q^4 + 4*f_q^2) - f_q^2 )
        f_alpha = max(0, (|t| - t_crit) / sqrt(df))
        RV_{alpha} = 0.5 * ( sqrt(f_alpha^4 + 4*f_alpha^2) - f_alpha^2 )

    Interpretation:
        RV_{q=1} is the minimum partial R^2 that an unobserved confounder U must have with
        BOTH treatment T (conditional on X) and outcome Y (conditional on X, T) to attenuate
        the point estimate tau_hat to zero.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64).ravel()
    Y_arr = np.asarray(Y, dtype=np.float64).ravel()

    n = len(Y_arr)
    k = X_arr.shape[1]
    df = n - k - 2

    if df <= 1:
        raise ValueError(f"Insufficient degrees of freedom (df = {df}) for sensitivity analysis")

    # Fit OLS of Y on [1, T, X]
    M = np.column_stack([np.ones(n, dtype=np.float64), T_arr, X_arr])
    # Solve normal equations via QR / least squares for numerical stability
    beta_ols, residuals, rank, s = np.linalg.lstsq(M, Y_arr, rcond=None)

    y_pred = M @ beta_ols
    res = Y_arr - y_pred
    sigma2 = np.sum(res ** 2) / df

    # Invert (M^T M) with rank safety (F12)
    if rank < M.shape[1]:
        inv_mtm = np.linalg.pinv(M.T @ M)
    else:
        try:
            inv_mtm = np.linalg.inv(M.T @ M)
        except np.linalg.LinAlgError:
            inv_mtm = np.linalg.pinv(M.T @ M)

    se_ols = float(np.sqrt(sigma2 * max(0.0, inv_mtm[1, 1])))
    tau_ols = float(beta_ols[1])

    t_stat = tau_ols / se_ols if se_ols > 1e-12 else 0.0
    abs_t = abs(t_stat)

    # Partial R^2 of treatment with outcome given X
    partial_r2 = float((t_stat ** 2) / (t_stat ** 2 + df))

    # 1. Robustness Value to bring point estimate to zero (q = 1)
    f_q = abs_t / np.sqrt(df)
    rv_q1 = float(0.5 * (np.sqrt(f_q ** 4 + 4.0 * f_q ** 2) - f_q ** 2))

    # 2. Robustness Value to bring 95% confidence interval to include zero
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    if abs_t > t_crit:
        f_alpha = (abs_t - t_crit) / np.sqrt(df)
        rv_alpha = float(0.5 * (np.sqrt(f_alpha ** 4 + 4.0 * f_alpha ** 2) - f_alpha ** 2))
    else:
        rv_alpha = 0.0

    rv_q1_pct = round(rv_q1 * 100.0, 2)
    rv_alpha_pct = round(rv_alpha * 100.0, 2)

    return {
        "test_name": "cinelli_hazlett_linear_sensitivity",
        "methodology": "Exact Cinelli & Hazlett (2020 JRSS-B) Robustness Value on OLS specification Y ~ T + X.",
        "assumptions": "Assumes a parametric linear outcome equation for benchmark sensitivity benchmarking.",
        "ols_treatment_effect": round(tau_ols, 4),
        "ols_standard_error": round(se_ols, 4),
        "t_statistic": round(t_stat, 4),
        "degrees_of_freedom": int(df),
        "partial_r2_treatment": round(partial_r2, 4),
        "robustness_value_rv": round(rv_q1, 4),
        "rv_alpha": round(rv_alpha, 4),
        "robustness_value_point_estimate_rv_q1": round(rv_q1, 4),
        "robustness_value_point_estimate_percent": rv_q1_pct,
        "robustness_value_significance_rv_alpha": round(rv_alpha, 4),
        "robustness_value_significance_percent": rv_alpha_pct,
        "significance_level": alpha,
        "passed": bool(rv_q1 > 0.0),
        "interpretation": (
            f"Cinelli-Hazlett Robustness Value RV(q=1) = {rv_q1_pct}%: An unobserved confounder would need to explain "
            f"at least {rv_q1_pct}% of the residual variation in treatment assignment AND {rv_q1_pct}% of the residual "
            f"variation in the outcome to attenuate the OLS treatment effect to zero. "
            f"To render the effect statistically insignificant at alpha={alpha}, RV(alpha) = {rv_alpha_pct}%."
        ),
        "limitations": (
            "This formal sensitivity analysis is derived strictly for the linear regression specification Y ~ T + X. "
            "While it provides a rigorous quantitative benchmark, it does not directly apply point-wise to nonlinear or "
            "doubly robust estimators (AIPW), which require distinct semiparametric sensitivity bounds."
        ),
    }


def bootstrap_ate_confidence_interval(
    model: BaseCausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bootstraps: int = 25,
    alpha: float = 0.05,
    random_state: int = 42,
    statistic: str = "sample_mean_cate",
) -> Tuple[float, float, np.ndarray]:
    """Compute empirical bootstrap confidence interval for ATE by refitting across resamples.

    Parameters
    ----------
    model : BaseCausalEstimator
        Estimator instance to clone and fit across bootstrap resamples.
    X : np.ndarray
        Covariates.
    T : np.ndarray
        Treatment assignment indicator.
    Y : np.ndarray
        Observed outcomes.
    n_bootstraps : int, default=25
        Number of bootstrap replications. For rigorous tail percentiles (e.g. 95% CI),
        >= 100 resamples are recommended in final reporting.
    alpha : float, default=0.05
        Significance level (e.g. 0.05 for 95% interval).
    random_state : int, default=42
        Seed for reproducibility.
    statistic : {"sample_mean_cate", "training_ate"}, default="sample_mean_cate"
        Target statistic to evaluate on each resample (F9):
        - "sample_mean_cate": Evaluates estimate_ate(X[idx]) (sample average of predicted CATEs).
        - "training_ate": Evaluates estimate_ate() (training-sample Hájek / AIPW parameter).

    Returns
    -------
    lower, upper, ates_arr : tuple
        Lower percentile, upper percentile, and array of successful bootstrap estimates.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    boot_ates: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(T[idx])) < 2:
            continue
        b_model = clone(model)
        try:
            b_model.fit(X[idx], T[idx], Y[idx])
            if statistic == "training_ate":
                val = b_model.estimate_ate()
            else:
                val = b_model.estimate_ate(X[idx])
            boot_ates.append(float(val))
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
