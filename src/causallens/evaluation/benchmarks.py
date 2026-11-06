"""Multi-replication causal benchmark runner for CFRNet IHDP-100.

Evaluates causal estimators against ground-truth potential outcomes
across multiple benchmark replications without leaking oracle data into training.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

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
        "IPW": IPWEstimator(),
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
        Estimators to evaluate. Defaults to all six CausalLens models.
    data_dir : path-like, optional
        Path to directory containing raw IHDP NPZ archives.

    Returns
    -------
    df_results : pd.DataFrame
        Table with metrics for each model on the test split of this replication.
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
        # Fit model on training observational split
        fitted_model = model.__class__()
        fitted_model.fit(train["X"], train["T"], train["Y"])

        # Predict out-of-sample on test split
        ite_pred = fitted_model.estimate_ite(test["X"])
        ate_pred = fitted_model.estimate_ate(test["X"])

        # Compute ground-truth metrics
        pehe = pehe_score(tau_true, ite_pred)
        ate_err = ate_error(ate_true, ate_pred)
        p_val = policy_value(truth["mu0"], truth["mu1"], ite_pred)
        p_risk = policy_risk(truth["mu0"], truth["mu1"], ite_pred)

        records.append({
            "replication": replication,
            "model": name,
            "pehe": round(pehe, 4),
            "ate_error": round(ate_err, 4),
            "ate_pred": round(float(ate_pred), 4),
            "ate_true": round(ate_true, 4),
            "policy_value": round(p_val, 4),
            "policy_risk": round(p_risk, 4),
        })

    return pd.DataFrame(records)


def run_ihdp_benchmark(
    replications: Sequence[int] = (0, 1, 2),
    models: Optional[Dict[str, BaseCausalEstimator]] = None,
    data_dir: Optional[Any] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark experiments across multiple IHDP replications.

    Parameters
    ----------
    replications : sequence of int
        List of replication indices to evaluate (e.g. range(10) or [0]).
    models : dict, optional
    data_dir : path-like, optional
    verbose : bool, default=True

    Returns
    -------
    summary_df : pd.DataFrame
        Aggregated results showing mean and standard error for PEHE and ATE error.
    """
    all_runs: List[pd.DataFrame] = []

    for rep in replications:
        if verbose:
            print(f"Evaluating IHDP replication {rep}...")
        df_rep = evaluate_replication(rep, models=models, data_dir=data_dir)
        all_runs.append(df_rep)

    combined = pd.concat(all_runs, ignore_index=True)

    # Compute aggregate summary across replications
    grouped = combined.groupby("model").agg(
        pehe_mean=("pehe", "mean"),
        pehe_std=("pehe", "std"),
        ate_error_mean=("ate_error", "mean"),
        ate_error_std=("ate_error", "std"),
        policy_risk_mean=("policy_risk", "mean"),
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
    grouped = grouped.sort_values(by="pehe_mean", ascending=True).reset_index(drop=True)
    return grouped
