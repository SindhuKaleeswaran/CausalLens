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
    at.slider[0].set_value(1).run()
    assert len(at.exception) == 0

    # Switch to replication 3
    at.slider[0].set_value(3).run()
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
