"""Policy decision simulation, capacity-constrained allocation, and paired benchmark comparisons.

Provides rigorous policy evaluation under the potential outcomes framework:
- Policy Value: V(pi) = 1/n * sum_i [ pi_i * Y1_i + (1 - pi_i) * Y0_i ]
- Policy Regret: V(pi*) - V(pi) >= 0 (excess loss relative to oracle allocation)
- Policy Gain: V(pi) - V(treat_none)
- Capacity-constrained budget simulations
- Paired statistical significance testing across benchmark replications
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def evaluate_policy_rule(
    y0_true: np.ndarray,
    y1_true: np.ndarray,
    decisions: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate empirical outcomes under a deterministic treatment assignment rule.

    Parameters
    ----------
    y0_true : np.ndarray of shape (n,)
        Potential outcomes under control.
    y1_true : np.ndarray of shape (n,)
        Potential outcomes under treatment.
    decisions : np.ndarray of shape (n,)
        Binary treatment assignment indicators in {0, 1}.

    Returns
    -------
    dict containing policy_value, treated_count, treated_fraction, mean_tau_treated, mean_tau_untreated.
    """
    y0 = np.asarray(y0_true, dtype=np.float64)
    y1 = np.asarray(y1_true, dtype=np.float64)
    pi = np.asarray(decisions, dtype=np.float64)

    if not (y0.shape == y1.shape == pi.shape):
        raise ValueError("y0_true, y1_true, and decisions must have matching shapes")
    n = len(y0)
    if n == 0:
        raise ValueError("Arrays cannot be empty")

    realized = pi * y1 + (1.0 - pi) * y0
    v = float(np.mean(realized))
    n_treated = int(np.sum(pi == 1.0))
    n_untreated = n - n_treated
    tau_true = y1 - y0

    mean_tau_tr = float(np.mean(tau_true[pi == 1.0])) if n_treated > 0 else 0.0
    mean_tau_un = float(np.mean(tau_true[pi == 0.0])) if n_untreated > 0 else 0.0

    return {
        "policy_value": round(v, 4),
        "treated_count": n_treated,
        "treated_fraction": round(n_treated / n, 4),
        "mean_tau_treated": round(mean_tau_tr, 4),
        "mean_tau_untreated": round(mean_tau_un, 4),
    }


def simulate_capacity_constrained_policy(
    y0_true: np.ndarray,
    y1_true: np.ndarray,
    tau_pred: np.ndarray,
    capacity_fraction: float = 0.4,
    min_benefit_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Simulate budget-constrained treatment assignment comparing model-guided vs oracle vs baselines.

    Under capacity constraint K = int(round(capacity_fraction * n)):
    1. Model Policy: Treats the top K units with highest predicted tau_hat > threshold.
    2. Oracle Policy: Treats the top K units with highest true tau > threshold.
    3. Random Policy: Treats a random subset of K units.
    4. Treat All: Assigns treatment to all N units.
    5. Treat None: Withholds treatment from all N units.

    Parameters
    ----------
    y0_true : np.ndarray
    y1_true : np.ndarray
    tau_pred : np.ndarray
    capacity_fraction : float, default=0.4
        Maximum proportion of units that can receive treatment (0.0 to 1.0).
    min_benefit_threshold : float, default=0.0
        Minimum expected treatment benefit required to consider treatment.

    Returns
    -------
    dict with summary metrics for each strategy and index allocations.
    """
    y0 = np.asarray(y0_true, dtype=np.float64)
    y1 = np.asarray(y1_true, dtype=np.float64)
    pred = np.asarray(tau_pred, dtype=np.float64)
    tau_star = y1 - y0
    n = len(y0)

    if not 0.0 <= capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be in [0.0, 1.0]")

    k_max = int(np.round(capacity_fraction * n))

    # Baseline 1: Treat None (pi = 0)
    pi_none = np.zeros(n, dtype=np.float64)
    res_none = evaluate_policy_rule(y0, y1, pi_none)
    v_none = res_none["policy_value"]

    # Baseline 2: Treat All (pi = 1)
    pi_all = np.ones(n, dtype=np.float64)
    res_all = evaluate_policy_rule(y0, y1, pi_all)

    # Strategy 3: Oracle Constrained Allocation
    oracle_order = np.argsort(-tau_star)  # descending order of true effect
    pi_oracle = np.zeros(n, dtype=np.float64)
    if k_max > 0:
        top_k_oracle = oracle_order[:k_max]
        # Only treat if benefit > threshold
        eligible_oracle = [idx for idx in top_k_oracle if tau_star[idx] > min_benefit_threshold]
        pi_oracle[eligible_oracle] = 1.0
    res_oracle = evaluate_policy_rule(y0, y1, pi_oracle)
    v_oracle = res_oracle["policy_value"]

    # Strategy 4: Model-Guided Allocation
    model_order = np.argsort(-pred)  # descending order of predicted effect
    pi_model = np.zeros(n, dtype=np.float64)
    selected_indices: List[int] = []
    if k_max > 0:
        top_k_model = model_order[:k_max]
        eligible_model = [idx for idx in top_k_model if pred[idx] > min_benefit_threshold]
        pi_model[eligible_model] = 1.0
        selected_indices = list(eligible_model)
    res_model = evaluate_policy_rule(y0, y1, pi_model)
    v_model = res_model["policy_value"]

    # Strategy 5: Random Allocation (Analytical Expected Value)
    p_rand = k_max / n if n > 0 else 0.0
    v_random = float(p_rand * np.mean(y1) + (1.0 - p_rand) * np.mean(y0))

    # Regrets: Excess loss vs constrained oracle
    regret_model = max(0.0, v_oracle - v_model)
    regret_random = max(0.0, v_oracle - v_random)
    regret_all = max(0.0, v_oracle - res_all["policy_value"])
    regret_none = max(0.0, v_oracle - v_none)

    # Gains: Improvement over Treat None status quo
    gain_model = v_model - v_none
    gain_oracle = v_oracle - v_none
    gain_random = v_random - v_none
    gain_all = res_all["policy_value"] - v_none

    return {
        "capacity_fraction": capacity_fraction,
        "max_capacity_units": k_max,
        "total_units": n,
        "strategies": {
            "Model-Guided": {
                "policy_value": round(v_model, 4),
                "gain_vs_treat_none": round(gain_model, 4),
                "policy_regret": round(regret_model, 4),
                "treated_count": res_model["treated_count"],
                "treated_fraction": res_model["treated_fraction"],
                "mean_tau_treated": res_model["mean_tau_treated"],
                "mean_tau_untreated": res_model["mean_tau_untreated"],
            },
            "Benchmark Oracle (Optimal)": {
                "policy_value": round(v_oracle, 4),
                "gain_vs_treat_none": round(gain_oracle, 4),
                "policy_regret": 0.0,
                "treated_count": res_oracle["treated_count"],
                "treated_fraction": res_oracle["treated_fraction"],
                "mean_tau_treated": res_oracle["mean_tau_treated"],
                "mean_tau_untreated": res_oracle["mean_tau_untreated"],
            },
            "Random Assignment": {
                "policy_value": round(v_random, 4),
                "gain_vs_treat_none": round(gain_random, 4),
                "policy_regret": round(regret_random, 4),
                "treated_count": k_max,
                "treated_fraction": round(p_rand, 4),
                "mean_tau_treated": round(float(np.mean(tau_star)), 4),
                "mean_tau_untreated": round(float(np.mean(tau_star)), 4),
            },
            "Treat All": {
                "policy_value": round(res_all["policy_value"], 4),
                "gain_vs_treat_none": round(gain_all, 4),
                "policy_regret": round(regret_all, 4),
                "treated_count": n,
                "treated_fraction": 1.0,
                "mean_tau_treated": res_all["mean_tau_treated"],
                "mean_tau_untreated": 0.0,
            },
            "Treat None": {
                "policy_value": round(v_none, 4),
                "gain_vs_treat_none": 0.0,
                "policy_regret": round(regret_none, 4),
                "treated_count": 0,
                "treated_fraction": 0.0,
                "mean_tau_treated": 0.0,
                "mean_tau_untreated": res_none["mean_tau_untreated"],
            },
        },
        "selected_indices": selected_indices,
    }


def compute_paired_benchmark_differences(
    detailed_df: pd.DataFrame,
    baseline_model: str = "T-Learner",
    metric_col: str = "root_pehe",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute rigorous paired statistical differences against a reference model across replications.

    Calculates:
    - Delta = Metric(Competitor) - Metric(Baseline)
    - Paired t-test p-value
    - Wilcoxon signed-rank test p-value
    - 95% paired bootstrap confidence interval for Delta
    """
    models = sorted(detailed_df["model"].unique())
    if baseline_model not in models:
        raise ValueError(f"Baseline model '{baseline_model}' not found in benchmark records")

    base_records = (
        detailed_df[detailed_df["model"] == baseline_model]
        .sort_values("replication")
        .set_index("replication")[metric_col]
    )

    records: List[Dict[str, Any]] = []
    rng = np.random.RandomState(random_state)

    for m in models:
        if m == baseline_model:
            continue
        m_records = (
            detailed_df[detailed_df["model"] == m]
            .sort_values("replication")
            .set_index("replication")[metric_col]
        )
        common_reps = sorted(list(set(base_records.index) & set(m_records.index)))
        if len(common_reps) < 5:
            continue

        base_vals = base_records.loc[common_reps].values
        comp_vals = m_records.loc[common_reps].values
        deltas = comp_vals - base_vals  # Positive means competitor has higher error (worse)

        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        std_delta = float(np.std(deltas, ddof=1))

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(comp_vals, base_vals)

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(comp_vals, base_vals)
        except Exception:
            w_pval = np.nan

        # Paired bootstrap 95% CI for mean delta
        n_boot = 500
        boot_means = [np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(n_boot)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        # Win rate (% of replications where baseline beat competitor)
        baseline_win_rate = float(np.mean(base_vals < comp_vals) * 100.0)

        records.append({
            "comparison": f"{m} vs {baseline_model}",
            "competitor": m,
            "baseline": baseline_model,
            "metric": metric_col,
            "n_replications": len(common_reps),
            "mean_difference": round(mean_delta, 4),
            "median_difference": round(median_delta, 4),
            "std_difference": round(std_delta, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "baseline_win_rate_pct": round(baseline_win_rate, 1),
            "paired_t_pval": float(t_pval),
            "wilcoxon_pval": float(w_pval) if np.isfinite(w_pval) else np.nan,
            "statistically_significant_05": bool(float(t_pval) < 0.05),
        })

    return pd.DataFrame(records)
