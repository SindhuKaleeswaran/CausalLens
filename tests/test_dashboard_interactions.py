"""End-to-end headless interaction tests for CausalLens Streamlit dashboard.

Uses Streamlit's official AppTest testing framework to simulate real user interactions
across all six analytical sections without a browser:
1. Data & Confounding
2. Model Evaluation
3. Counterfactual Explorer
4. Treatment Heterogeneity
5. Robustness Analysis
6. Evidence Assessment
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def synthetic_dashboard(monkeypatch):
    """Artificial observed/oracle fixtures; never fall back to private archives."""
    import numpy as np
    import streamlit as st
    from causallens.data import loader
    st.cache_data.clear()
    st.cache_resource.clear()
    def observed(replication=0, split="train", data_dir=None):
        rng = np.random.default_rng(1000 + replication + (100 if split == "test" else 0))
        n = 120 if split == "train" else 45
        X = rng.normal(size=(n, 25))
        X[:, 6:] = rng.integers(0, 2, size=(n, 19))
        X[:, 13] += 1
        T = np.arange(n) % 2
        Y = X[:, 0] + T * (2 + X[:, 1]) + rng.normal(size=n)
        return dict(X=X, T=T, Y=Y)
    def truth(replication=0, split="test", data_dir=None):
        X = observed(replication, split)["X"]
        mu0 = X[:, 0]
        mu1 = mu0 + 2 + X[:, 1]
        return dict(mu0=mu0, mu1=mu1, tau=mu1-mu0, ycf=mu0)
    monkeypatch.setattr(loader, "load_ihdp", observed)
    monkeypatch.setattr(loader, "load_ihdp_truth", truth)
    import causallens.evaluation.benchmarks as benchmarks
    monkeypatch.setattr(benchmarks, "load_ihdp", observed)
    monkeypatch.setattr(benchmarks, "load_ihdp_truth", truth)
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def test_dashboard_initial_render():
    """Verify clean initial render with zero exceptions across all 6 tabs."""
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0, f"Exceptions on startup: {[e.value for e in at.exception]}"
    # Verify app title and header context
    assert any(w.label == "Download eight-model experiment (CSV)" for w in at.get("download_button"))
    assert len(at.tabs) == 6
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == [
        "Data & Confounding",
        "Model Evaluation",
        "Counterfactual Explorer",
        "Treatment Heterogeneity",
        "Robustness Analysis",
        "Evidence Assessment",
    ]


def test_dashboard_model_selection_all_estimators():
    """Verify switching between all 6 canonical estimators without KeyError."""
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0

    estimator_ids = [
        "doubly_robust",
        "x_learner",
        "t_learner",
        "ipw",
        "s_learner",
        "naive",
    ]

    for est_id in estimator_ids:
        at.sidebar.selectbox[0].select(est_id).run()
        assert len(at.exception) == 0, f"Exception selecting {est_id}: {[e.value for e in at.exception]}"


def test_dashboard_switch_replications():
    """Verify changing IHDP replication index."""
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0

    # Switch to replication 1
    at.sidebar.slider[0].set_value(1).run()
    assert len(at.exception) == 0

    # Switch to replication 3
    at.sidebar.slider[0].set_value(3).run()
    assert len(at.exception) == 0


def test_dashboard_observed_profile_and_hypothetical_edits():
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0
    at.number_input(key="profile_index").set_value(3).run()
    assert len(at.exception) == 0
    at.checkbox(key="modify_profile").check().run()
    at.number_input(key="profile_0_3_0").set_value(1.5).run()
    assert len(at.exception) == 0
    assert any("unsupported combinations" in w.value for w in at.warning)
    text = " ".join(m.value for m in at.markdown)
    assert "Cross-Split Difference" in text
    assert "Confounding Bias Gap" not in text
    assert "not an estimate of confounding bias" in text


def test_dashboard_robustness_actions_and_caching():
    """Verify executing refutation tests and session state persistence."""
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0

    # Execute placebo test button
    if len(at.button) > 0:
        at.button[0].click().run()
        assert len(at.exception) == 0

    # Execute noise injection test button
    if len(at.button) > 1:
        at.button[1].click().run()
        assert len(at.exception) == 0

    # Verify session state cached the results
    assert "refutation_results" in at.session_state
    assert len(at.session_state.refutation_results) > 0


def test_dashboard_export_controls():
    """Verify presence and validity of export download buttons."""
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0

    # At least Model Evaluation CSV export and Reliability Dossier JSON export
    assert len(at.get("download_button")) >= 2
    labels = [btn.label for btn in at.get("download_button")]
    assert "Export Model Evaluation (CSV)" in labels
    assert "Export Reliability Dossier (JSON)" in labels


def test_heldout_subject_displays_selected_model_and_replication():
    from causallens.data.loader import load_ihdp, load_ihdp_truth
    from causallens.models import NaiveEstimator, TLearner

    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    for rep, model_id, factory, unit in [(0, "t_learner", TLearner, 4), (1, "naive", NaiveEstimator, 8)]:
        at.sidebar.slider[0].set_value(rep)
        at.sidebar.selectbox[0].select(model_id).run()
        at.selectbox(key="test_unit_selector").select(unit).run()
        assert not at.exception
        train, test = load_ihdp(rep), load_ihdp(rep, "test")
        truth = load_ihdp_truth(rep)
        y0, y1 = factory().fit(**train).predict_potential_outcomes(test["X"][unit:unit+1])
        cards = [m.value for m in at.markdown]
        expected = [
            ("Observed Factual", f'{test["Y"][unit]:.2f}'),
            ("Model unobserved branch", f'{(y0 if test["T"][unit] else y1)[0]:.2f}'),
            ("Estimated Effect tau(x)", f'{(y1-y0)[0]:+.2f}'),
            ("Oracle Ground Truth tau", f'{truth["tau"][unit]:+.2f}'),
        ]
        for label, val in expected:
            assert any(label in card and f">{val}</div>" in card for card in cards), (rep, model_id, unit, label, val, [c for c in cards if label in c])
    radio = at.radio(key="archetype_preset_radio")
    radio.set_value(radio.options[1]).run()
    assert next(w for w in at.selectbox if w.key and w.key.startswith("preset_unit_")).value == int(truth["tau"].argmax())
    at.slider(key="policy_capacity_slider").set_value(0).run()
    assert any("Treating 0/" in m.value for m in at.markdown)
    assert not at.exception


def test_corrupt_benchmark_keeps_explorer_available(monkeypatch):
    import pandas as pd
    original = pd.read_csv

    def read_csv(path, *args, **kwargs):
        if str(path).endswith("ihdp100_unified_results.csv"):
            raise pd.errors.EmptyDataError("empty detailed file")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert any("Full benchmark artifacts unavailable" in m.value for m in at.warning)
    assert at.selectbox(key="test_unit_selector").value == 0


@pytest.mark.parametrize('error', [FileNotFoundError, ValueError])
def test_missing_or_invalid_archive_has_actionable_empty_state(monkeypatch, error):
    from causallens.data import loader
    def unavailable(*args, **kwargs):
        raise error('private local path must not be exposed')
    monkeypatch.setattr(loader, 'load_ihdp', unavailable)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert any('IHDP data unavailable or invalid' in e.value for e in at.error)
    assert any('simulated outcomes' in info.value for info in at.info)
    assert not any('private local path' in e.value for e in at.error)


def test_unidentified_sensitivity_does_not_break_evidence_or_exports(monkeypatch):
    from causallens.evaluation import robustness
    def unavailable(*args, **kwargs):
        raise ValueError('unidentified coefficient')
    monkeypatch.setattr(robustness, 'sensitivity_unobserved_confounding', unavailable)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert any('OLS sensitivity is unavailable' in w.value for w in at.warning)
    assert any('OLS t statistic / residual df**: N/A / N/A' in m.value for m in at.markdown)
    assert any(b.label == 'Export Reliability Dossier (JSON)' for b in at.get('download_button'))


def test_cached_fits_and_evaluation_survive_profile_and_metric_changes(monkeypatch):
    import causallens.evaluation.benchmarks as benchmarks
    original = benchmarks.evaluate_replication
    calls = []
    def tracked(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(benchmarks, 'evaluate_replication', tracked)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.number_input(key='profile_index').set_value(2).run()
    at.selectbox(key='benchmark_metric').select('policy_risk').run()
    assert not at.exception
    assert calls == [0]
    at.sidebar.slider[0].set_value(1).run()
    assert not at.exception
    assert calls == [0, 1]


def test_tied_heterogeneous_predictions_render(monkeypatch):
    import numpy as np
    from causallens.models import TLearner
    def tied(self, X):
        return np.r_[np.zeros(int(len(X)*.7)), np.ones(len(X)-int(len(X)*.7))]
    monkeypatch.setattr(TLearner, 'estimate_ite', tied)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.sidebar.selectbox[0].select('t_learner').run()
    assert not at.exception
    assert any('Tied effects stay together' in c.value for c in at.caption)


def test_nonfinite_hypothetical_profile_and_threshold_have_explicit_fallback(monkeypatch):
    # Streamlit normalizes inf in widget transport; inject at its return boundary
    # to exercise our defense if an invalid value reaches the application.
    import streamlit as st
    original = st.number_input
    def invalid(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs.get('key') in {'profile_0_0_0', 'policy_min_threshold'}:
            return float('inf')
        return value
    monkeypatch.setattr(st, 'number_input', invalid)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.checkbox(key='modify_profile').check().run()
    assert not at.exception
    assert any('finite numeric feature values' in w.value for w in at.warning)
    assert any('finite eligibility threshold' in w.value for w in at.warning)


def test_model_initialization_failure_is_distinct_from_missing_data(monkeypatch):
    from causallens.models import TLearner
    def broken(*args, **kwargs):
        raise RuntimeError('internal model initialization')
    monkeypatch.setattr(TLearner, 'fit', broken)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert any('Model initialization failed' in e.value for e in at.error)


def test_absent_overlap_and_consensus_are_na(monkeypatch):
    from causallens.models import IPWEstimator
    from causallens.evaluation import robustness
    original = IPWEstimator.fit
    def without_diagnostics(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.overlap_diagnostics_ = {}
        return self
    monkeypatch.setattr(IPWEstimator, 'fit', without_diagnostics)
    monkeypatch.setattr(robustness, 'compute_estimator_disagreement', lambda *a, **kw: {})
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert any('Effective sample size**: N/A' in m.value for m in at.markdown)
    assert any('Test mean effect spread**: N/A' in m.value for m in at.markdown)
