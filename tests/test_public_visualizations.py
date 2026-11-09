"""Public visualizations and exports must reflect real synthetic computations."""
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
from public_demo_utils import (
    comparison_interpretation, disagreement_matrix, naive_association, synthetic_balance, synthetic_profiles,
)
from causallens.synthetic_demo import FEATURES, generate_example


def test_balance_and_disagreement_definitions():
    train, _, _ = generate_example()
    result = synthetic_balance(train)
    for j in range(5):
        a = train['X'][train['T'] == 1, j]
        b = train['X'][train['T'] == 0, j]
        assert result['Signed SMD'][j] == pytest.approx((a.mean()-b.mean()) / np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2))
    predictions = {'A': np.array([1., 2., 3.]), 'B': np.array([3., 2., 1.]), 'C': np.zeros(3)}
    matrix = disagreement_matrix(predictions)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 0)
    assert matrix.loc['A', 'B'] == pytest.approx(4/3)
    profiles = synthetic_profiles({'tau': np.array([-1., .1, 3.])}, predictions)
    assert profiles['Lowest oracle effect'] == 0
    assert profiles['Near-zero oracle effect'] == 1
    assert profiles['Highest oracle effect'] == 2


def test_naive_association_matches_arm_means():
    outcome = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    treatment = np.array([0, 0, 0, 1, 1, 1])
    naive = naive_association(outcome, treatment)
    assert naive == pytest.approx(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


@pytest.mark.parametrize('naive,estimated,oracle,expected_phrase', [
    (5.0, 2.1, 2.0, 'is closer to the simulated effect than the naive association'),
    (2.05, 5.0, 2.0, 'is not closer to the simulated effect than the naive association'),
    (1.0, 3.0, 2.0, "matches the naive association's distance from the simulated effect"),
])
def test_comparison_interpretation_verdict_and_no_overclaiming(naive, estimated, oracle, expected_phrase):
    text = comparison_interpretation(naive, estimated, oracle, 'X-Learner')
    assert expected_phrase in text
    assert 'X-Learner' in text
    # Must not assert that confounding is the sole or proven cause of any gap.
    assert 'entirely' not in text.lower()
    assert 'proves' not in text.lower() and 'proof that' in text.lower()


def test_kpi_cards_share_evaluation_population_and_update_correctly():
    at = AppTest.from_file(str(ROOT / 'deploy/streamlit_app.py'), default_timeout=60).run()
    assert not at.exception

    def metric_value(label):
        return float(next(m for m in at.metric if m.label == label).value)

    def caption_with(substring):
        return next(c.value for c in at.caption if substring in c.value)

    estimator = at.sidebar.selectbox[1].value
    assert any(m.label == f'Estimated effect · {estimator}' for m in at.metric)
    naive_before = metric_value('Naive observed association')
    oracle_before = metric_value('True simulated effect (oracle)')
    interpretation = caption_with('Naive association')
    assert f'{naive_before:.3f}' in interpretation and f'{oracle_before:.3f}' in interpretation
    assert 'entirely' not in interpretation.lower()

    # Switching estimators must not change the naive association or the oracle effect:
    # all three KPI cards must stay pinned to the same held-out population.
    at.sidebar.selectbox[1].select('Naive (Unadjusted)').run()
    assert not at.exception
    assert metric_value('Naive observed association') == pytest.approx(naive_before)
    assert metric_value('True simulated effect (oracle)') == pytest.approx(oracle_before)
    assert any(m.label == 'Estimated effect · Naive (Unadjusted)' for m in at.metric)

    # Switching the seed changes the held-out population, so the naive/oracle values may change.
    at.sidebar.selectbox[0].select(3).run()
    assert not at.exception
    naive_seed3 = metric_value('Naive observed association')
    oracle_seed3 = metric_value('True simulated effect (oracle)')
    assert (naive_seed3, oracle_seed3) != (naive_before, oracle_before)


def test_guided_workflow_and_provenance_note_present_once():
    at = AppTest.from_file(str(ROOT / 'deploy/streamlit_app.py'), default_timeout=60).run()
    assert not at.exception
    assert any('Explore Association' in c.value for c in at.markdown)
    provenance = [e.label for e in at.expander if 'Research vs. public demonstration' in (e.label or '')]
    assert len(provenance) == 1
    # The lengthy provenance explanation should live in one place, not be repeated verbatim on every tab.
    occurrences = sum('IHDP-100' in m.value for m in at.markdown)
    assert occurrences == 1


def test_all_public_views_controls_and_download_payloads(monkeypatch):
    downloads = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        downloads[label] = data
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, 'download_button', capture)
    at = AppTest.from_file(str(ROOT / 'deploy/streamlit_app.py'), default_timeout=60).run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == [
        'Data & Overview', 'Model Evaluation', 'Counterfactual Explorer',
        'Policy Simulator', 'Robustness & Sensitivity', 'Evidence Summary']
    assert [len(tab.get('plotly_chart')) for tab in at.tabs] == [3, 7, 2, 1, 4, 0]
    for tab in at.tabs:
        for plot in tab.get('plotly_chart'):
            spec = json.loads(plot.proto.spec)
            assert spec['data']
            assert spec['layout']['height'] == 400

    def select(label, value):
        next(widget for widget in at.selectbox if widget.label == label).select(value).run()
        assert not at.exception

    def slide(label, value):
        next(widget for widget in at.slider if widget.label == label).set_value(value).run()
        assert not at.exception

    for metric in ['Root-PEHE', 'ATE error', 'Policy regret']:
        select('Compare models by', metric)
        assert at.dataframe[1].value[metric].is_monotonic_increasing
    for feature in FEATURES:
        select('Heterogeneity feature', feature)
    for profile in ['Lowest oracle effect', 'Near-zero oracle effect', 'Highest oracle effect',
                    'Largest estimator disagreement', 'Choose by index']:
        select('Synthetic profile', profile)
        assert at.slider[0].disabled == (profile != 'Choose by index')
    slide('Synthetic unit index', 149)
    for capacity in [0, 100, 40]:
        slide('Treatment capacity (%)', capacity)
        policy = at.dataframe[3].value.set_index('Strategy')
        assert policy.loc['Model-Guided', 'treated_count'] <= int(150*capacity/100)
    for noise in [0., 3., 1.5]:
        slide('Prediction noise standard deviation', noise)
    select('Estimator', 'Naive (Unadjusted)')  # tied CATE groups must still render
    select('Example seed', 9)
    csv = pd.read_csv(io.StringIO(downloads['Download synthetic evaluation (CSV)']))
    evidence = json.loads(downloads['Download synthetic evidence (JSON)'])
    assert len(csv) == len(evidence['evaluation']) == 6
    pd.testing.assert_frame_equal(csv, pd.DataFrame(evidence['evaluation']))
    assert evidence['seed'] == 9 and evidence['estimator'] == 'Naive (Unadjusted)'
    assert evidence['synthetic_profile']['unit_index'] == 149
    assert evidence['prediction_stress']['noise_sd'] == 1.5
    assert 'fully synthetic' in evidence['data_kind']
    assert 'linear_sensitivity' in evidence
    assert len(at.get('download_button')) == 2
