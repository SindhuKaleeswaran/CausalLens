"""Reference cases for the independent scientific audit."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from causallens.evaluation.benchmarks import (
    aggregate_benchmark_results,
    get_default_models,
)
from causallens.evaluation.metrics import pehe_score, policy_risk, policy_value
from causallens.evaluation.policy import (
    compute_paired_benchmark_differences,
    evaluate_policy_rule,
    simulate_capacity_constrained_policy,
)
from causallens.evaluation.robustness import sensitivity_unobserved_confounding
from causallens.models import NaiveEstimator


@pytest.mark.parametrize("capacity", [0, 0.19, 0.5, 1])
@pytest.mark.parametrize("threshold", [-3, 0, 5])
def test_allocation_against_exhaustive_search(capacity, threshold):
    tau = np.array([2.0, 2.0, -2.0, 0.0, 1.0])
    pred = np.array([3.0, 3.0, 6.0, -1.0, 2.0])
    k = int(capacity * len(tau))
    ref = max(
        np.dot(mask, tau) / len(tau)
        for mask in product([0, 1], repeat=len(tau))
        if sum(mask) <= k
    )
    sim = simulate_capacity_constrained_policy(
        np.zeros_like(tau), tau, pred, capacity, threshold
    )
    model = sim["strategies"]["Model-Guided"]
    assert sim["max_capacity_units"] == k
    assert model["treated_count"] <= k
    assert (
        sim["selected_indices"]
        == [
            i
            for i in sorted(range(len(tau)), key=lambda i: -pred[i])
            if pred[i] > threshold
        ][:k]
    )
    assert sim["strategies"]["Benchmark Oracle (Optimal)"][
        "policy_value"
    ] == pytest.approx(ref)
    assert model["policy_regret"] == pytest.approx(ref - model["policy_value"])
    assert (sim["strategies"]["Treat All"]["policy_regret"] is None) == (k < len(tau))


def test_threshold_does_not_change_oracle_objective():
    assert policy_risk([0, 0], [1, 2], [0, 0], threshold=3) == 1.5


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [[1]], [], [1, 2]])
def test_invalid_policy_and_metric_vectors(bad):
    with pytest.raises(ValueError):
        simulate_capacity_constrained_policy([0], [1], bad)
    with pytest.raises(ValueError):
        policy_value([0], [1], bad)
    with pytest.raises(ValueError):
        pehe_score([[1]], [[1]])


def test_policy_requires_binary_decisions():
    with pytest.raises(ValueError):
        evaluate_policy_rule([0], [1], [2])


def test_policy_small_effect_precision():
    res = simulate_capacity_constrained_policy([0], [0.00004], [0], 1)
    assert res["strategies"]["Model-Guided"]["policy_regret"] == pytest.approx(0.00004)


def test_pairing_with_failed_fits_and_duplicate_ids():
    df = pd.DataFrame(
        [
            (i, m, "SUCCESS", float(i + (m == "other")))
            for i in range(8)
            for m in ("T-Learner", "other")
        ],
        columns=["replication", "model", "status", "root_pehe"],
    )
    df.loc[1, ["status", "root_pehe"]] = ["FAILED", np.nan]
    out = compute_paired_benchmark_differences(df).iloc[0]
    assert out.n_replications == 7 and out.mean_difference == 1
    assert out.wilcoxon_holm == pytest.approx(2 / 128)
    with pytest.raises(ValueError, match="uplicate"):
        compute_paired_benchmark_differences(pd.concat([df, df]))
    with pytest.raises(ValueError, match="uplicate"):
        aggregate_benchmark_results(pd.concat([df, df]))


def test_unidentified_treatment_and_redundant_covariate():
    x = np.arange(40.0)
    t = x % 2
    y = 3 * t + np.sin(x)
    with pytest.raises(ValueError, match="identif"):
        sensitivity_unobserved_confounding(t[:, None], t, y)
    assert (
        sensitivity_unobserved_confounding(np.column_stack([x, x]), t, y)[
            "degrees_of_freedom"
        ]
        == 37
    )
    x = np.arange(12.0)
    assert sensitivity_unobserved_confounding(
        x[:, None], x % 2, 1.6 * (x % 2) + np.sin(x)
    )["rv_alpha"] == pytest.approx(0.3144, abs=0.0001)


def test_named_features_and_refit():
    x = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) % 3})
    t = np.arange(10) % 2
    for model in get_default_models().values():
        model.fit(x, t, x.a)
        with pytest.raises(ValueError, match="feature"):
            model.estimate_ite(x[["b", "a"]])
        model.fit(x[["a"]], t, x.a)
        assert model.estimate_ite(x[["a"]]).shape == (10,)
    with pytest.raises(ValueError, match="eature"):
        NaiveEstimator().fit(x, t, x.a).estimate_ate(np.ones((4, 3)))


@pytest.mark.parametrize("name", ["IPW + Weighted Reg", "Doubly Robust (AIPW)"])
def test_invalid_propensity_clipping(name):
    model = get_default_models()[name].set_params(clip_range=(1, 0))
    with pytest.raises(ValueError, match="clip_range"):
        model.fit(np.arange(20.0)[:, None], np.arange(20) % 2, np.arange(20.0))


def test_published_releases_are_separate_and_complete():
    from pathlib import Path
    from causallens.evaluation.benchmarks import load_benchmark_artifacts
    from causallens.models.experimental import get_experimental_models

    root = Path(__file__).resolve().parents[1] / "results"
    original, _ = load_benchmark_artifacts(root, get_default_models())
    extended, _ = load_benchmark_artifacts(
        root / "extended", {**get_default_models(), **get_experimental_models()}
    )
    assert len(original) == 600 and len(extended) == 800
    assert original.status.eq("SUCCESS").all() and extended.status.eq("SUCCESS").all()
    keys = ["replication", "model"]
    cols = ["root_pehe", "ate_error", "policy_value", "policy_risk"]
    baseline = original.set_index(keys)[cols]
    pd.testing.assert_frame_equal(
        baseline, extended.set_index(keys).loc[baseline.index, cols]
    )
    for name in ("Naive (Unadjusted)", "S-Learner"):
        assert original.loc[original.model == name, "cate_pearson_r"].isna().all()
    assert original.loc[original.model == "Doubly Robust (AIPW)", "seed"].eq(42).all()


def test_weighted_fit_internal_error_is_not_silently_retried():
    from sklearn.base import BaseEstimator
    from causallens.models import IPWEstimator

    class BrokenWeightedModel(BaseEstimator):
        def fit(self, x, y, sample_weight=None):
            if sample_weight is not None:
                raise TypeError("internal weighted arithmetic failure")
            return self

    x = np.arange(12.)
    with pytest.raises(TypeError, match="internal weighted"):
        IPWEstimator(outcome_estimator=BrokenWeightedModel()).fit(x[:, None], x % 2, x)
