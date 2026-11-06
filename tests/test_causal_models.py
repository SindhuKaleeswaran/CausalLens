"""Unit and integration tests for CausalLens Phase 6 causal estimators."""
from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from causallens.data.loader import archive_path, load_ihdp
from causallens.models import (
    BaseCausalEstimator,
    DoublyRobustEstimator,
    IPWEstimator,
    NaiveEstimator,
    SLearner,
    TLearner,
    XLearner,
)


@pytest.fixture
def synthetic_data():
    """Generates synthetic observational data with known constant treatment effect = 4.0."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.normal(size=(n, 5))
    # Confounded treatment assignment: P(T=1|X) depends on X[:, 0]
    logits = 0.8 * X[:, 0] - 0.3 * X[:, 1]
    prob = 1.0 / (1.0 + np.exp(-logits))
    T = rng.binomial(1, prob).astype(float)
    # Outcome: baseline response + true effect (4.0) * T + noise
    baseline = 1.5 + 2.0 * X[:, 0] - 1.2 * X[:, 1]
    tau_true = 4.0
    noise = rng.normal(scale=0.1, size=n)
    Y = baseline + tau_true * T + noise
    return X, T, Y, tau_true


@pytest.mark.parametrize(
    "estimator_cls",
    [NaiveEstimator, SLearner, TLearner, XLearner, IPWEstimator, DoublyRobustEstimator],
)
def test_unfitted_estimator_raises(estimator_cls):
    model = estimator_cls()
    X_dummy = np.ones((10, 3))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.estimate_ite(X_dummy)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.estimate_ate(X_dummy)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_potential_outcomes(X_dummy)


def test_input_validation_errors():
    model = SLearner()
    # Non-2D X
    with pytest.raises(ValueError, match="X must be 2D"):
        model.fit(np.zeros(10), np.zeros(10), np.zeros(10))
    # Mismatched rows
    with pytest.raises(ValueError, match="Sample size mismatch"):
        model.fit(np.zeros((10, 2)), np.zeros(8), np.zeros(10))
    # Non-binary T
    with pytest.raises(ValueError, match="binary treatment values"):
        model.fit(np.zeros((10, 2)), np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0]), np.zeros(10))
    # Non-finite values
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(np.array([[np.nan, 1.0], [2.0, 3.0]]), np.array([0, 1]), np.array([1.0, 2.0]))
    # Only one arm present
    with pytest.raises(ValueError, match="Both treatment and control"):
        model.fit(np.ones((10, 2)), np.zeros(10), np.zeros(10))


def test_naive_estimator():
    model = NaiveEstimator()
    X = np.ones((4, 2))
    T = np.array([0, 0, 1, 1], dtype=float)
    Y = np.array([2.0, 4.0, 10.0, 12.0])  # mean(Y0)=3.0, mean(Y1)=11.0, diff=8.0
    model.fit(X, T, Y)

    assert model.estimate_ate() == pytest.approx(8.0)
    ite = model.estimate_ite(X)
    np.testing.assert_array_equal(ite, np.full(4, 8.0))

    y0, y1 = model.predict_potential_outcomes(X)
    np.testing.assert_array_equal(y0, np.full(4, 3.0))
    np.testing.assert_array_equal(y1, np.full(4, 11.0))
    assert "unconfoundedness" in model.assumptions


def test_meta_learners_recover_synthetic_effect(synthetic_data):
    X, T, Y, tau_true = synthetic_data

    for est in [SLearner(), TLearner(), XLearner()]:
        est.fit(X, T, Y)
        ate_est = est.estimate_ate(X)
        assert ate_est == pytest.approx(tau_true, abs=0.5)

        ite_est = est.estimate_ite(X)
        assert ite_est.shape == (len(X),)
        assert np.mean(ite_est) == pytest.approx(tau_true, abs=0.5)

        y0, y1 = est.predict_potential_outcomes(X)
        np.testing.assert_allclose(y1 - y0, ite_est, rtol=1e-5)


def test_ipw_estimator(synthetic_data):
    X, T, Y, tau_true = synthetic_data
    model = IPWEstimator(clip_range=(0.05, 0.95))
    model.fit(X, T, Y)

    ate_est = model.estimate_ate()
    assert ate_est == pytest.approx(tau_true, abs=0.6)
    assert model.effective_sample_size_ is not None and model.effective_sample_size_ > 0

    props = model.get_propensity_scores(X)
    assert np.all(props >= 0.05) and np.all(props <= 0.95)


def test_doubly_robust_estimator(synthetic_data):
    X, T, Y, tau_true = synthetic_data
    model = DoublyRobustEstimator()
    model.fit(X, T, Y)

    ate_est = model.estimate_ate()
    assert ate_est == pytest.approx(tau_true, abs=0.5)

    low, high = model.ate_confidence_interval(alpha=0.05)
    assert low < tau_true < high
    assert "double_robustness" in model.assumptions


def test_custom_base_estimators(synthetic_data):
    X, T, Y, _ = synthetic_data
    gbr = GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=42)
    s_learner = SLearner(base_estimator=gbr)
    s_learner.fit(X, T, Y)
    assert s_learner.estimate_ate(X) > 0


def test_integration_with_ihdp_benchmark():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not all(archive_path(s, data_dir).exists() for s in ("train", "test")):
        pytest.skip("IHDP benchmark not present in data/raw")

    train = load_ihdp(0, "train", data_dir=data_dir)
    test = load_ihdp(0, "test", data_dir=data_dir)

    models = [
        NaiveEstimator(),
        SLearner(),
        TLearner(),
        XLearner(),
        IPWEstimator(),
        DoublyRobustEstimator(),
    ]

    for model in models:
        model.fit(train["X"], train["T"], train["Y"])
        summary = model.summary()
        assert summary["n_features_in"] == 25
        assert summary["training_treated"] == 123
        assert summary["training_control"] == 549

        # Evaluate on test split
        ite_test = model.estimate_ite(test["X"])
        assert ite_test.shape == (75,)
        assert np.all(np.isfinite(ite_test))

        ate_test = model.estimate_ate(test["X"])
        assert np.isfinite(ate_test)

        y0_test, y1_test = model.predict_potential_outcomes(test["X"])
        assert y0_test.shape == (75,)
        assert y1_test.shape == (75,)
        assert np.all(np.isfinite(y0_test))
        assert np.all(np.isfinite(y1_test))
