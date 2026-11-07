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
            pehe = pehe_score(tau_true, ite_pred)
            ate_err = ate_error(ate_true, ate_pred)
            p_val = policy_value(truth["mu0"], truth["mu1"], ite_pred)
            p_risk = policy_risk(truth["mu0"], truth["mu1"], ite_pred)

            records.append({
                "replication": replication,
                "model": name,
                "status": "SUCCESS",
                "pehe": round(pehe, 4),
                "ate_error": round(ate_err, 4),
                "ate_pred": round(float(ate_pred), 4),
                "ate_true": round(ate_true, 4),
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
                "pehe": np.nan,
                "ate_error": np.nan,
                "ate_pred": np.nan,
                "ate_true": round(ate_true, 4),
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
        Directory where CSV and JSON results will be saved.

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

    # Compute aggregate summary across successful runs
    success_df = detailed_df[detailed_df["status"] == "SUCCESS"]

    grouped = success_df.groupby("model").agg(
        n_successful=("replication", "count"),
        pehe_mean=("pehe", "mean"),
        pehe_std=("pehe", "std"),
        ate_error_mean=("ate_error", "mean"),
        ate_error_std=("ate_error", "std"),
        policy_risk_mean=("policy_risk", "mean"),
        avg_runtime_sec=("total_time_sec", "mean"),
    ).reset_index()

    grouped["pehe_summary"] = grouped.apply(
        lambda r: f"{r['pehe_mean']:.3f} +/- {r['pehe_std']:.3f}" if pd.notnull(r["pehe_std"]) else f"{r['pehe_mean']:.3f}",
        axis=1,
    )
    grouped["ate_error_summary"] = grouped.apply(
        lambda r: f"{r['ate_error_mean']:.3f} +/- {r['ate_error_std']:.3f}" if pd.notnull(r["ate_error_std"]) else f"{r['ate_error_mean']:.3f}",
        axis=1,
    )

    # Sort so best models (lowest PEHE) appear first
    summary_df = grouped.sort_values(by="pehe_mean", ascending=True).reset_index(drop=True)

    # Optional export
    if export_dir is not None:
        p_dir = Path(export_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(p_dir / "benchmark_summary.csv", index=False)
        detailed_df.to_csv(p_dir / "benchmark_detailed.csv", index=False)
        with open(p_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    return summary_df, detailed_df
