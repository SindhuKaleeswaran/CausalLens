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
from public_demo_utils import disagreement_matrix, synthetic_balance, synthetic_profiles
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
