"""Multi-replication causal benchmark runner for CFRNet IHDP-100.

Evaluates causal estimators against ground-truth potential outcomes
across benchmark replications without leaking oracle data into training.
Preserves model hyperparameters via sklearn.base.clone, tracks runtime,
and exports reproducible structured results.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone

from causallens.data.loader import load_ihdp, load_ihdp_truth
from causallens.evaluation.metrics import ate_error, pehe_score, policy_risk, policy_value
from causallens.models import (
    BaseCausalEstimator,
    DoublyRobustEstimator,
    IPWEstimator,
    NaiveEstimator,
    SLearner,
    TLearner,
    XLearner,
)


def get_default_models() -> Dict[str, BaseCausalEstimator]:
    """Return dictionary of the six standard CausalLens estimators."""
    return {
        "Naive (Unadjusted)": NaiveEstimator(),
        "S-Learner": SLearner(),
        "T-Learner": TLearner(),
        "X-Learner": XLearner(),
        "IPW + Weighted Reg": IPWEstimator(),
        "Doubly Robust (AIPW)": DoublyRobustEstimator(),
    }


def evaluate_replication(
    replication: int = 0,
    models: Optional[Dict[str, BaseCausalEstimator]] = None,
    data_dir: Optional[Any] = None,
) -> pd.DataFrame:
    """Train and evaluate models on a single IHDP replication.

    Parameters
    ----------
    replication : int, default=0
        0-based replication index (0 to 99).
    models : dict of BaseCausalEstimator, optional
        Estimators to evaluate.
    data_dir : path-like, optional
        Path to directory containing raw IHDP NPZ archives.

    Returns
    -------
    df_results : pd.DataFrame
        Table with metrics and runtime for each model on the test split.
    """
    if models is None:
        models = get_default_models()

    # Load observational data (Strict isolation: train receives only X, T, Y)
    train = load_ihdp(replication, split="train", data_dir=data_dir)
    test = load_ihdp(replication, split="test", data_dir=data_dir)

    # Load oracle ground-truth strictly for evaluation
    truth = load_ihdp_truth(replication, split="test", data_dir=data_dir)
    tau_true = truth["tau"]
    ate_true = float(np.mean(tau_true))

    records: List[Dict[str, Any]] = []

    for name, model in models.items():
        t0 = time.perf_counter()
        # Clone ensures hyperparameters and state are cleanly preserved
        try:
            fitted_model = clone(model)
            fitted_model.fit(train["X"], train["T"], train["Y"])
            fit_time = time.perf_counter() - t0

            # Predict out-of-sample on test split
            t1 = time.perf_counter()
            ite_pred = fitted_model.estimate_ite(test["X"])
            ate_pred = fitted_model.estimate_ate(test["X"])
            predict_time = time.perf_counter() - t1

            # Compute ground-truth metrics
            root_pehe = pehe_score(tau_true, ite_pred)
            pehe_sq = float(np.mean((ite_pred - tau_true) ** 2))
            ate_err = ate_error(ate_true, ate_pred)
            signed_ate_err = float(ate_pred - ate_true)
            p_val = policy_value(truth["mu0"], truth["mu1"], ite_pred)
            p_risk = policy_risk(truth["mu0"], truth["mu1"], ite_pred)

            # In-sample ATE parameter from training fit if available
            try:
                in_sample_ate = float(fitted_model.estimate_ate(None))
            except Exception:
                in_sample_ate = np.nan

            # CATE effect correlation metrics
            if np.std(ite_pred) > 1e-12 and np.std(tau_true) > 1e-12:
                cate_r = float(stats.pearsonr(tau_true, ite_pred)[0])
                cate_rho = float(stats.spearmanr(tau_true, ite_pred)[0])
            else:
                cate_r = np.nan
                cate_rho = np.nan

            records.append({
                "replication": replication,
                "model": name,
                "status": "SUCCESS",
                "seed": getattr(model, "random_state", None),
                "pehe": round(root_pehe, 4),  # backwards compatibility root-PEHE
                "root_pehe": round(root_pehe, 4),
                "pehe_sq": round(pehe_sq, 4),
                "ate_error": round(ate_err, 4),
                "signed_ate_error": round(signed_ate_err, 4),
                "ate_pred": round(float(ate_pred), 4),
                "ate_true": round(ate_true, 4),
                "in_sample_ate": round(in_sample_ate, 4) if np.isfinite(in_sample_ate) else np.nan,
                "cate_pearson_r": round(cate_r, 4),
                "cate_spearman_r": round(cate_rho, 4),
                "policy_value": round(p_val, 4),
                "policy_risk": round(p_risk, 4),
                "fit_time_sec": round(fit_time, 4),
                "predict_time_sec": round(predict_time, 4),
                "total_time_sec": round(fit_time + predict_time, 4),
                "error_message": "",
            })
        except Exception as ex:
            records.append({
                "replication": replication,
                "model": name,
                "status": "FAILED",
                "seed": getattr(model, "random_state", None),
                "pehe": np.nan,
                "root_pehe": np.nan,
                "pehe_sq": np.nan,
                "ate_error": np.nan,
                "signed_ate_error": np.nan,
                "ate_pred": np.nan,
                "ate_true": round(ate_true, 4),
                "in_sample_ate": np.nan,
                "cate_pearson_r": np.nan,
                "cate_spearman_r": np.nan,
                "policy_value": np.nan,
                "policy_risk": np.nan,
                "fit_time_sec": np.nan,
                "predict_time_sec": np.nan,
                "total_time_sec": round(time.perf_counter() - t0, 4),
                "error_message": str(ex),
            })

    return pd.DataFrame(records)


def run_ihdp_benchmark(
    replications: Sequence[int] = (0, 1, 2),
    models: Optional[Dict[str, BaseCausalEstimator]] = None,
    data_dir: Optional[Any] = None,
    verbose: bool = True,
    export_dir: Optional[Any] = None,
    results_dir: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run benchmark experiments across multiple IHDP replications.

    Parameters
    ----------
    replications : sequence of int
        Replication indices to evaluate (0 to 99).
    models : dict, optional
    data_dir : path-like, optional
    verbose : bool, default=True
    export_dir : path-like, optional
        Directory where reports/summary CSV and JSON results will be saved.
    results_dir : path-like, optional
        Directory where results/ihdp100_unified_results.csv and summary will be saved.

    Returns
    -------
    summary_df, detailed_df : tuple of pd.DataFrame
    """
    all_runs: List[pd.DataFrame] = []
    n_reps = len(replications)

    if not n_reps or len(set(replications)) != n_reps:
        raise ValueError("Expected nonempty unique replication indices")
    for i, rep in enumerate(replications):
        if verbose:
            print(f"[{i+1}/{n_reps}] Evaluating IHDP replication {rep}...")
        df_rep = evaluate_replication(rep, models=models, data_dir=data_dir)
        all_runs.append(df_rep)

    detailed_df = pd.concat(all_runs, ignore_index=True)

    # Compute aggregate summary across runs using explicit mean and median bootstrap CIs
    summary_df = aggregate_benchmark_results(
        detailed_df, export_dir=export_dir, results_dir=results_dir
    )
    return summary_df, detailed_df


def compute_bootstrap_ci(
    values: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute nonparametric percentile bootstrap confidence interval for a target statistic.

    Parameters
    ----------
    values : np.ndarray
        Array of replication metrics.
    statistic_fn : callable
        Function computing summary statistic (e.g., np.mean, np.median).
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    alpha : float, default=0.05
        Significance level (0.05 yields 95% confidence interval).
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    tuple of (low, high) confidence bounds.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        val = float(statistic_fn(arr)) if n == 1 else np.nan
        return (val, val)

    rng = np.random.RandomState(random_state)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[b] = float(statistic_fn(sample))

    low = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return (round(low, 6), round(high, 6))


def aggregate_benchmark_results(
    detailed_df: pd.DataFrame,
    export_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Aggregate replication-level metrics across IHDP benchmark fits.

    Computes Mean, SD, 95% Bootstrap CI for the Mean, Median, and 95% Bootstrap CI
    for the Median for all primary evaluation dimensions. Explicitly separates
    and identifies mean-target vs median-target intervals to prevent labeling ambiguity.
    """
    if detailed_df.empty or detailed_df[["model", "replication", "status"]].isna().any().any():
        raise ValueError("Expected populated replication records")
    if detailed_df.duplicated(["replication", "model"]).any():
        raise ValueError("Duplicate replication and model")
    if not detailed_df.status.isin(["SUCCESS", "FAILED"]).all():
        raise ValueError("Unknown fit status")
    required = ["root_pehe", "pehe_sq", "ate_error", "policy_risk", "policy_value", "total_time_sec"]
    if not np.isfinite(detailed_df.loc[detailed_df.status == "SUCCESS", required]).all().all():
        raise ValueError("Successful primary metrics must be finite")
    expected = detailed_df.replication.nunique()
    if not detailed_df.groupby("model").size().eq(expected).all():
        raise ValueError("Missing model and replication combinations")
    summary_rows = []
    for model_name, grp in detailed_df.groupby("model"):
        n_total = len(grp)
        succ = grp[grp["status"] == "SUCCESS"]
        n_succ = len(succ)
        n_fail = n_total - n_succ

        def stats_for(col):
            if n_succ == 0 or col not in succ:
                return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            vals = succ[col].dropna().values
            if not len(vals):
                return (np.nan,) * 7
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            med = float(np.median(vals))
            mean_lo, mean_hi = compute_bootstrap_ci(vals, np.mean, n_bootstrap=1000, alpha=0.05, random_state=42)
            med_lo, med_hi = compute_bootstrap_ci(vals, np.median, n_bootstrap=1000, alpha=0.05, random_state=42)
            return (m, s, med, mean_lo, mean_hi, med_lo, med_hi)

        pehe_m, pehe_s, pehe_med, pehe_mean_lo, pehe_mean_hi, pehe_med_lo, pehe_med_hi = stats_for("root_pehe")
        pehe_sq_m, pehe_sq_s, pehe_sq_med, _, _, _, _ = stats_for("pehe_sq")
        ate_m, ate_s, ate_med, ate_mean_lo, ate_mean_hi, ate_med_lo, ate_med_hi = stats_for("ate_error")
        prisk_m, prisk_s, prisk_med, prisk_mean_lo, prisk_mean_hi, prisk_med_lo, prisk_med_hi = stats_for("policy_risk")
        pval_m, pval_s, pval_med, _, _, _, _ = stats_for("policy_value")
        r_m, r_s, r_med, _, _, _, _ = stats_for("cate_pearson_r")
        time_m, time_s, time_med, _, _, _, _ = stats_for("total_time_sec")
        time_total = float(succ["total_time_sec"].sum()) if n_succ > 0 else 0.0

        summary_rows.append({
            "model": model_name,
            "n_successful": n_succ,
            "n_failed": n_fail,
            "root_pehe_mean": pehe_m,
            "root_pehe_std": pehe_s,
            "root_pehe_mean_ci95_low": pehe_mean_lo,
            "root_pehe_mean_ci95_high": pehe_mean_hi,
            "root_pehe_median": pehe_med,
            "root_pehe_median_ci95_low": pehe_med_lo,
            "root_pehe_median_ci95_high": pehe_med_hi,
            # Explicit aliases preserving backwards-compatibility
            "root_pehe_ci95_low": pehe_mean_lo,
            "root_pehe_ci95_high": pehe_mean_hi,
            "pehe_mean": pehe_m,
            "pehe_std": pehe_s,
            "pehe_sq_mean": pehe_sq_m,
            "pehe_sq_std": pehe_sq_s,
            "pehe_sq_median": pehe_sq_med,
            "ate_error_mean": ate_m,
            "ate_error_std": ate_s,
            "ate_error_mean_ci95_low": ate_mean_lo,
            "ate_error_mean_ci95_high": ate_mean_hi,
            "ate_error_median": ate_med,
            "ate_error_median_ci95_low": ate_med_lo,
            "ate_error_median_ci95_high": ate_med_hi,
            # Explicit aliases preserving backwards-compatibility
            "ate_error_ci95_low": ate_mean_lo,
            "ate_error_ci95_high": ate_mean_hi,
            "policy_risk_mean": prisk_m,
            "policy_risk_std": prisk_s,
            "policy_risk_mean_ci95_low": prisk_mean_lo,
            "policy_risk_mean_ci95_high": prisk_mean_hi,
            "policy_risk_median": prisk_med,
            "policy_risk_median_ci95_low": prisk_med_lo,
            "policy_risk_median_ci95_high": prisk_med_hi,
            # Explicit aliases preserving backwards-compatibility
            "policy_risk_ci95_low": prisk_mean_lo,
            "policy_risk_ci95_high": prisk_mean_hi,
            "policy_value_mean": pval_m,
            "policy_value_std": pval_s,
            "cate_pearson_r_n_defined": int(succ["cate_pearson_r"].count()),
            "cate_pearson_r_mean": r_m,
            "cate_pearson_r_median": r_med,
            "avg_runtime_sec": time_m,
            "total_runtime_sec": round(time_total, 4),
            "pehe_summary": f"{pehe_m:.3f} +/- {pehe_s:.3f}" if pd.notnull(pehe_s) else f"{pehe_m:.3f}",
            "pehe_mean_ci_summary": f"[{pehe_mean_lo:.3f}, {pehe_mean_hi:.3f}]",
            "pehe_median_ci_summary": f"{pehe_med:.3f} [{pehe_med_lo:.3f}, {pehe_med_hi:.3f}]",
            "ate_error_summary": f"{ate_m:.3f} +/- {ate_s:.3f}" if pd.notnull(ate_s) else f"{ate_m:.3f}",
            "ate_error_mean_ci_summary": f"[{ate_mean_lo:.3f}, {ate_mean_hi:.3f}]",
            "ate_error_median_ci_summary": f"{ate_med:.3f} [{ate_med_lo:.3f}, {ate_med_hi:.3f}]",
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="root_pehe_mean", ascending=True).reset_index(drop=True)

    # Export to legacy report dir if requested
    if export_dir is not None:
        p_dir = Path(export_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(p_dir / "benchmark_summary.csv", index=False)
        detailed_df.to_csv(p_dir / "benchmark_detailed.csv", index=False)
        with open(p_dir / "benchmark_summary.json", "w") as f:
            json.dump(json.loads(summary_df.to_json(orient="records", double_precision=15)), f, indent=2, allow_nan=False)

    # Export to results/ directory if requested
    if results_dir is not None:
        r_dir = Path(results_dir)
        r_dir.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(r_dir / "ihdp100_unified_results.csv", index=False)
        summary_df.to_csv(r_dir / "ihdp100_summary.csv", index=False)

    return summary_df


def load_benchmark_artifacts(results_dir: Path, names):
    """Read a full 100-replication release and reject partial or inconsistent files."""
    detail = pd.read_csv(results_dir / "ihdp100_unified_results.csv")
    summary = pd.read_csv(results_dir / "ihdp100_summary.csv")
    expected_keys = pd.MultiIndex.from_product([range(100), sorted(names)])
    keys = pd.MultiIndex.from_frame(detail[["replication", "model"]])
    if not keys.is_unique or not keys.sort_values().equals(expected_keys):
        raise ValueError("Expected one record per replication and estimator")
    if not detail.status.isin(["SUCCESS", "FAILED"]).all():
        raise ValueError("Unknown fit status")
    successful = detail[detail.status == "SUCCESS"]
    if not np.isfinite(successful[["root_pehe", "ate_error", "policy_risk"]]).all().all():
        raise ValueError("Invalid successful metrics")
    indexed = summary.set_index("model").reindex(sorted(names))
    counts = successful.groupby("model").size().reindex(sorted(names), fill_value=0)
    if len(summary) != len(names) or not indexed.n_successful.eq(counts).all() or not indexed.n_failed.eq(100-counts).all():
        raise ValueError("Summary fit counts differ from detailed records")
    for metric in ("root_pehe", "ate_error", "policy_risk"):
        means = successful.groupby("model")[metric].mean().reindex(sorted(names))
        if not np.allclose(indexed[metric + "_mean"], means, equal_nan=True):
            raise ValueError("Summary and replication metrics differ")
    for col in ("pehe_mean_ci_summary", "pehe_median_ci_summary", "ate_error_mean_ci_summary", "ate_error_median_ci_summary"):
        if col not in indexed:
            raise ValueError("Missing labeled uncertainty intervals")
    return detail, summary
