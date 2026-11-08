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
        fitted_model = clone(model)
        try:
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
                cate_r = 0.0
                cate_rho = 0.0

            records.append({
                "replication": replication,
                "model": name,
                "status": "SUCCESS",
                "seed": 42 + replication,
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
                "seed": 42 + replication,
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

    for i, rep in enumerate(replications):
        if verbose:
            print(f"[{i+1}/{n_reps}] Evaluating IHDP replication {rep}...")
        df_rep = evaluate_replication(rep, models=models, data_dir=data_dir)
        all_runs.append(df_rep)

    detailed_df = pd.concat(all_runs, ignore_index=True)

    # Compute aggregate summary across runs
    summary_rows = []
    for model_name, grp in detailed_df.groupby("model"):
        n_total = len(grp)
        succ = grp[grp["status"] == "SUCCESS"]
        n_succ = len(succ)
        n_fail = n_total - n_succ

        def stats_for(col):
            if n_succ == 0 or col not in succ:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            vals = succ[col].dropna()
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            med = float(vals.median())
            ci_hw = 1.96 * s / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            return (m, s, med, m - ci_hw, m + ci_hw)

        pehe_m, pehe_s, pehe_med, pehe_lo, pehe_hi = stats_for("root_pehe")
        pehe_sq_m, pehe_sq_s, pehe_sq_med, _, _ = stats_for("pehe_sq")
        ate_m, ate_s, ate_med, ate_lo, ate_hi = stats_for("ate_error")
        prisk_m, prisk_s, prisk_med, prisk_lo, prisk_hi = stats_for("policy_risk")
        pval_m, pval_s, pval_med, _, _ = stats_for("policy_value")
        r_m, r_s, r_med, _, _ = stats_for("cate_pearson_r")
        time_m, time_s, time_med, _, _ = stats_for("total_time_sec")
        time_total = float(succ["total_time_sec"].sum()) if n_succ > 0 else 0.0

        summary_rows.append({
            "model": model_name,
            "n_successful": n_succ,
            "n_failed": n_fail,
            "root_pehe_mean": pehe_m,
            "root_pehe_std": pehe_s,
            "root_pehe_median": pehe_med,
            "root_pehe_ci95_low": pehe_lo,
            "root_pehe_ci95_high": pehe_hi,
            "pehe_mean": pehe_m,
            "pehe_std": pehe_s,
            "pehe_sq_mean": pehe_sq_m,
            "pehe_sq_std": pehe_sq_s,
            "pehe_sq_median": pehe_sq_med,
            "ate_error_mean": ate_m,
            "ate_error_std": ate_s,
            "ate_error_median": ate_med,
            "ate_error_ci95_low": ate_lo,
            "ate_error_ci95_high": ate_hi,
            "policy_risk_mean": prisk_m,
            "policy_risk_std": prisk_s,
            "policy_risk_median": prisk_med,
            "policy_risk_ci95_low": prisk_lo,
            "policy_risk_ci95_high": prisk_hi,
            "policy_value_mean": pval_m,
            "policy_value_std": pval_s,
            "cate_pearson_r_mean": r_m,
            "cate_pearson_r_median": r_med,
            "avg_runtime_sec": time_m,
            "total_runtime_sec": round(time_total, 4),
            "pehe_summary": f"{pehe_m:.3f} +/- {pehe_s:.3f}" if pd.notnull(pehe_s) else f"{pehe_m:.3f}",
            "ate_error_summary": f"{ate_m:.3f} +/- {ate_s:.3f}" if pd.notnull(ate_s) else f"{ate_m:.3f}",
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="root_pehe_mean", ascending=True).reset_index(drop=True)

    # Export to legacy report dir if requested
    if export_dir is not None:
        p_dir = Path(export_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(p_dir / "benchmark_summary.csv", index=False)
        detailed_df.to_csv(p_dir / "benchmark_detailed.csv", index=False)
        with open(p_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    # Export to results/ directory if requested
    if results_dir is not None:
        r_dir = Path(results_dir)
        r_dir.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(r_dir / "ihdp100_unified_results.csv", index=False)
        summary_df.to_csv(r_dir / "ihdp100_summary.csv", index=False)

    return summary_df, detailed_df
