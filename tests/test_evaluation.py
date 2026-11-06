"""Unit and integration tests for CausalLens Phase 7 evaluation & refutation suite."""
from pathlib import Path
import numpy as np
import pytest

from causallens.data.loader import archive_path
from causallens.evaluation.benchmarks import evaluate_replication, run_ihdp_benchmark
from causallens.evaluation.metrics import (
    ate_error,
    evaluate_estimator,
    pehe_score,
    policy_risk,
    policy_value,
)
from causallens.evaluation.robustness import (
    bootstrap_ate_confidence_interval,
    refute_placebo_treatment,
    refute_random_confounder,
)
from causallens.models import DoublyRobustEstimator, SLearner, XLearner


@pytest.fixture
def synthetic_arrays():
    rng = np.random.RandomState(42)
    n = 100
    y0_true = rng.normal(loc=2.0, scale=0.5, size=n)
    tau_true = 3.0 + 0.5 * rng.normal(size=n)
    y1_true = y0_true + tau_true
    return y0_true, y1_true, tau_true


def test_pehe_score_known_values():
    tau_true = np.array([2.0, 4.0])
    tau_pred_exact = np.array([2.0, 4.0])
    assert pehe_score(tau_true, tau_pred_exact) == pytest.approx(0.0)

    tau_pred_off = np.array([3.0, 5.0])
    assert pehe_score(tau_true, tau_pred_off) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Shape mismatch"):
        pehe_score(tau_true, np.array([2.0]))
    with pytest.raises(ValueError, match="finite"):
        pehe_score(tau_true, np.array([np.nan, 2.0]))


def test_ate_error():
    assert ate_error(4.0, 4.25) == pytest.approx(0.25)
    assert ate_error(3.5, 3.5) == pytest.approx(0.0)


def test_policy_value_and_risk(synthetic_arrays):
    y0_true, y1_true, tau_true = synthetic_arrays
    # Oracle policy: tau_pred = tau_true
    risk_oracle = policy_risk(y0_true, y1_true, tau_true)
    assert risk_oracle == pytest.approx(0.0)

    # Inverted policy: assigns opposite of true benefit
    val_bad = policy_value(y0_true, y1_true, -tau_true)
    val_good = policy_value(y0_true, y1_true, tau_true)
    assert val_good >= val_bad


def test_refute_placebo_treatment():
    rng = np.random.RandomState(42)
    n = 150
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 2.0 * X[:, 0] + 5.0 * T + rng.normal(scale=0.1, size=n)

    res = refute_placebo_treatment(SLearner(), X, T, Y, n_simulations=5)
    assert res["original_ate"] == pytest.approx(5.0, abs=0.5)
    assert abs(res["placebo_ate_mean"]) < 1.0
    assert res["passed"] is True


def test_refute_random_confounder():
    rng = np.random.RandomState(42)
    n = 150
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 2.0 * X[:, 0] + 4.0 * T + rng.normal(scale=0.1, size=n)

    res = refute_random_confounder(SLearner(), X, T, Y, n_simulations=5)
    assert res["passed"] is True
    assert res["percent_shift"] < 15.0


def test_bootstrap_ate_confidence_interval():
    rng = np.random.RandomState(42)
    n = 100
    X = rng.normal(size=(n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 1.0 + 3.0 * T + rng.normal(scale=0.1, size=n)

    low, high, ates = bootstrap_ate_confidence_interval(SLearner(), X, T, Y, n_bootstraps=15)
    assert low <= 3.0 <= high
    assert len(ates) == 15


def test_evaluate_replication_with_ihdp():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not all(archive_path(s, data_dir).exists() for s in ("train", "test")):
        pytest.skip("IHDP benchmark not downloaded in data/raw")

    df = evaluate_replication(0, data_dir=data_dir)
    assert len(df) == 6
    assert set(df["model"]) == {
        "Naive (Unadjusted)",
        "S-Learner",
        "T-Learner",
        "X-Learner",
        "IPW",
        "Doubly Robust (AIPW)",
    }
    # Causal models should beat naive in PEHE or ATE error
    naive_pehe = float(df.loc[df["model"] == "Naive (Unadjusted)", "pehe"].iloc[0])
    xlearner_pehe = float(df.loc[df["model"] == "X-Learner", "pehe"].iloc[0])
    assert xlearner_pehe < naive_pehe
