"""Unit tests for policy decision simulation and paired benchmark differences."""
import numpy as np
import pandas as pd
import pytest

from causallens.evaluation.policy import (
    compute_paired_benchmark_differences,
    evaluate_policy_rule,
    simulate_capacity_constrained_policy,
)


@pytest.fixture
def synthetic_policy_data():
    rng = np.random.RandomState(42)
    n = 50
    y0 = rng.normal(loc=2.0, scale=0.5, size=n)
    tau_true = 1.0 + rng.normal(loc=2.0, scale=1.0, size=n)
    y1 = y0 + tau_true
    tau_pred = tau_true + rng.normal(scale=0.3, size=n)
    return y0, y1, tau_true, tau_pred


def test_evaluate_policy_rule_known_values():
    y0 = np.array([2.0, 3.0, 4.0])
    y1 = np.array([5.0, 6.0, 7.0])
    # Treat everyone
    res_all = evaluate_policy_rule(y0, y1, np.ones(3))
    assert res_all["policy_value"] == pytest.approx(6.0)
    assert res_all["treated_count"] == 3
    assert res_all["treated_fraction"] == 1.0

    # Treat none
    res_none = evaluate_policy_rule(y0, y1, np.zeros(3))
    assert res_none["policy_value"] == pytest.approx(3.0)
    assert res_none["treated_count"] == 0

    # Mismatched shapes
    with pytest.raises(ValueError):
        evaluate_policy_rule(y0, y1, np.ones(2))


def test_simulate_capacity_constrained_policy(synthetic_policy_data):
    y0, y1, tau_true, tau_pred = synthetic_policy_data
    n = len(y0)

    res = simulate_capacity_constrained_policy(y0, y1, tau_pred, capacity_fraction=0.4)
    assert res["capacity_fraction"] == 0.4
    assert res["max_capacity_units"] == int(round(0.4 * n))

    strats = res["strategies"]
    assert "Model-Guided" in strats
    assert "Benchmark Oracle (Optimal)" in strats
    assert "Treat None" in strats
    assert "Treat All" in strats
    assert "Random Assignment" in strats

    # Oracle regret should be zero
    assert strats["Benchmark Oracle (Optimal)"]["policy_regret"] == pytest.approx(0.0)

    # Oracle value should be >= Model-guided value >= Treat None value
    v_oracle = strats["Benchmark Oracle (Optimal)"]["policy_value"]
    v_model = strats["Model-Guided"]["policy_value"]
    v_none = strats["Treat None"]["policy_value"]
    assert v_oracle >= v_model
    assert v_model >= v_none

    # Regrets must be non-negative
    for s_name, s_data in strats.items():
        if s_name == "Treat All":
            assert s_data["policy_regret"] is None  # infeasible at this budget
        else:
            assert s_data["policy_regret"] >= 0.0

    # Out of bounds capacity fraction
    with pytest.raises(ValueError):
        simulate_capacity_constrained_policy(y0, y1, tau_pred, capacity_fraction=1.5)


def test_compute_paired_benchmark_differences():
    # Construct synthetic 10-replication detailed benchmark records
    records = []
    for rep in range(10):
        # Baseline: T-Learner (low error)
        records.append({"replication": rep, "model": "T-Learner", "root_pehe": 1.0 + 0.1 * rep})
        # Competitor: Naive (higher error)
        records.append({"replication": rep, "model": "Naive", "root_pehe": 4.0 + 0.1 * rep})
        # Competitor: X-Learner (similar error)
        records.append({"replication": rep, "model": "X-Learner", "root_pehe": 1.1 + 0.1 * rep})
    df = pd.DataFrame(records)

    diff_df = compute_paired_benchmark_differences(df, baseline_model="T-Learner", metric_col="root_pehe")
    assert len(diff_df) == 2
    assert "Naive vs T-Learner" in diff_df["comparison"].values
    assert "X-Learner vs T-Learner" in diff_df["comparison"].values

    naive_row = diff_df[diff_df["competitor"] == "Naive"].iloc[0]
    assert naive_row["mean_difference"] > 2.5
    assert bool(naive_row["statistically_significant_05"]) is True
    assert naive_row["baseline_win_rate_pct"] == 100.0

    # Invalid baseline model raises ValueError
    with pytest.raises(ValueError, match="not found"):
        compute_paired_benchmark_differences(df, baseline_model="NonExistentModel")
