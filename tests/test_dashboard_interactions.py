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


def test_dashboard_counterfactual_presets_and_sliders():
    """Verify applying all three vulnerability presets and moving sliders."""
    at = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert len(at.exception) == 0

    # Test presets
    presets = ["High Vulnerability", "Low Vulnerability", "Default (Mean Profile)"]
    for preset in presets:
        at.radio[0].set_value(preset).run()
        assert len(at.exception) == 0, f"Exception applying {preset}: {[e.value for e in at.exception]}"

    # Move sliders
    if len(at.slider) > 1:
        at.slider[1].set_value(1.5).run()
        assert len(at.exception) == 0


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
    assert len(at.download_button) >= 2
    labels = [btn.label for btn in at.download_button]
    assert "Export Model Evaluation (CSV)" in labels
    assert "Export Reliability Dossier (JSON)" in labels
