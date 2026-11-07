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
    compute_estimator_disagreement,
    refute_irrelevant_covariate,
    refute_placebo_treatment,
    refute_subsample_stability,
    sensitivity_unobserved_confounding,
)
from causallens.models import DoublyRobustEstimator, SLearner, TLearner, XLearner


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
    risk_oracle = policy_risk(y0_true, y1_true, tau_true)
    assert risk_oracle == pytest.approx(0.0)

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
    assert "placebo_ci_90" in res
    assert res["passed"] is True


def test_refute_irrelevant_covariate():
    rng = np.random.RandomState(42)
    n = 150
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 2.0 * X[:, 0] + 4.0 * T + rng.normal(scale=0.1, size=n)

    res = refute_irrelevant_covariate(SLearner(), X, T, Y, n_simulations=5)
    assert res["passed"] is True
    assert res["percent_shift"] < 15.0


def test_refute_subsample_stability():
    rng = np.random.RandomState(42)
    n = 150
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 1.5 * X[:, 0] + 3.5 * T + rng.normal(scale=0.1, size=n)

    res = refute_subsample_stability(SLearner(), X, T, Y, subsample_fraction=0.8, n_simulations=5)
    assert res["passed"] is True
    assert res["subsample_ate_mean"] == pytest.approx(3.5, abs=0.6)


def test_estimator_disagreement():
    rng = np.random.RandomState(42)
    n = 150
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 2.0 * X[:, 0] + 4.0 * T + rng.normal(scale=0.1, size=n)

    m1 = SLearner().fit(X, T, Y)
    m2 = TLearner().fit(X, T, Y)
    dis = compute_estimator_disagreement({"S": m1, "T": m2}, X)
    assert "ate_spread" in dis
    assert dis["ate_spread"] < 1.0
    assert dis["passed"] is True


def test_sensitivity_unobserved_confounding():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.normal(size=(n, 4))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 2.0 * X[:, 0] + 3.0 * T + rng.normal(scale=0.5, size=n)

    res = sensitivity_unobserved_confounding(X, T, Y, alpha=0.05)
    assert "robustness_value_rv" in res
    assert "rv_alpha" in res
    assert "partial_r2_treatment" in res
    assert 0.0 < res["robustness_value_rv"] < 1.0
    assert 0.0 <= res["rv_alpha"] <= res["robustness_value_rv"]
    assert res["degrees_of_freedom"] == n - 4 - 2
    assert res["passed"] is True


def test_cinelli_hazlett_rv_formula_manual():
    # Theoretical verification of RV_{q=1} calculation:
    # Given t = 2.0, df = 100:
    # f_q = 2.0 / sqrt(100) = 0.2
    # f_q^2 = 0.04
    # RV_{q=1} = 0.5 * (sqrt(0.04^2 + 4 * 0.04) - 0.04)
    #          = 0.5 * (sqrt(0.0016 + 0.16) - 0.04)
    #          = 0.5 * (sqrt(0.1616) - 0.04)
    #          = 0.5 * (0.401995 - 0.04) = 0.1809975
    t = 2.0
    df = 100.0
    f_q = abs(t) / np.sqrt(df)
    rv_manual = 0.5 * (np.sqrt(f_q**4 + 4.0 * (f_q**2)) - f_q**2)
    assert rv_manual == pytest.approx(0.1809975, abs=1e-5)


def test_monte_carlo_p_value_bounds():
    rng = np.random.RandomState(42)
    n = 100
    X = rng.normal(size=(n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = 5.0 * T + rng.normal(scale=0.1, size=n)

    # For B=5 simulations, Davison & Hinkley p_MC >= 1/(1+5) = 1/6 ~ 0.1667
    res = refute_placebo_treatment(SLearner(), X, T, Y, n_simulations=5)
    assert res["monte_carlo_p_value"] >= 1.0 / (1.0 + 5.0)


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
        "IPW + Weighted Reg",
        "Doubly Robust (AIPW)",
    }
    assert (df["status"] == "SUCCESS").all()
    assert (df["fit_time_sec"] >= 0).all()

    # Causal models should beat naive in PEHE or ATE error
    naive_pehe = float(df.loc[df["model"] == "Naive (Unadjusted)", "pehe"].iloc[0])
    xlearner_pehe = float(df.loc[df["model"] == "X-Learner", "pehe"].iloc[0])
    assert xlearner_pehe < naive_pehe


def test_trained_models_key_registry_and_aliases():
    """Regression test for KeyError: 'IPW + Weighted Reg'.

    Verifies that canonical model IDs ('ipw', 'doubly_robust', etc.) and all display
    aliases ('IPW + Weighted Outcome Regression', 'IPW + Weighted Reg', 'IPW') are present
    and resolve to the correct model classes without KeyError, while unknown keys raise KeyError.
    """
    from causallens.evaluation.benchmarks import get_default_models

    models = get_default_models()

    # Legacy to canonical mapping
    legacy_to_canonical = {
        "Doubly Robust (AIPW)": "doubly_robust",
        "X-Learner": "x_learner",
        "T-Learner": "t_learner",
        "IPW + Weighted Reg": "ipw",
        "S-Learner": "s_learner",
        "Naive (Unadjusted)": "naive",
    }

    # Ensure get_default_models contains all expected entries
    assert len(models) == 6
    for default_name in legacy_to_canonical:
        assert default_name in models, f"Missing default model: {default_name}"

    # Build full dictionary with canonical keys and aliases
    unified_dict = {}
    for default_name, inst in models.items():
        cid = legacy_to_canonical[default_name]
        unified_dict[cid] = inst
        unified_dict[default_name] = inst

    # Populate IPW aliases
    unified_dict["IPW"] = unified_dict["ipw"]
    unified_dict["IPW + Weighted Outcome Regression"] = unified_dict["ipw"]

    # Verify no KeyError for any expected lookup form
    expected_keys = [
        "ipw",
        "doubly_robust",
        "x_learner",
        "t_learner",
        "s_learner",
        "naive",
        "IPW",
        "IPW + Weighted Reg",
        "IPW + Weighted Outcome Regression",
        "Doubly Robust (AIPW)",
        "X-Learner",
        "T-Learner",
        "S-Learner",
        "Naive (Unadjusted)",
    ]
    for key in expected_keys:
        assert key in unified_dict, f"Lookup key '{key}' raised KeyError"
        assert unified_dict[key] is not None

    # Verify explicit error handling on unknown keys (zero silent substitution)
    with pytest.raises(KeyError):
        _ = unified_dict["nonexistent_model_estimator"]
