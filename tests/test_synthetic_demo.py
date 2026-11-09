"""Public deployment must remain functional without access to research data."""
import builtins
import json
from pathlib import Path
import numpy as np
import pytest
from causallens.synthetic_demo import generate_example, fit_example


def test_generator_reproducible_independent_and_split():
    a, b, truth = generate_example(0)
    again, _, _ = generate_example(0)
    assert a['X'].shape == (600, 5) and b['X'].shape == (150, 5)
    assert set(a) == {'X', 'T', 'Y'}
    assert not np.shares_memory(a['X'], b['X'])
    np.testing.assert_array_equal(a['X'], again['X'])
    np.testing.assert_allclose(truth['mu1']-truth['mu0'], truth['tau'])
    assert not np.array_equal(a['X'], generate_example(1)[0]['X'])
    assert set(np.unique(a['T'])) == {0, 1}


@pytest.mark.parametrize('seed', [-1, 10, 1.1, True])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError):
        generate_example(seed)


def test_all_available_scenarios_have_finite_model_results():
    for seed in range(10):
        _, _, _, predictions, records = fit_example(seed)
        assert len(predictions) == len(records) == 6
        assert all(np.isfinite(p).all() and p.shape == (150,) for p in predictions.values())
        json.dumps(records, allow_nan=False)


def test_public_ui_never_reads_research_data_or_downloads(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from causallens.data import acquisition, loader
    st.cache_data.clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('Public demo attempted research data access')
    monkeypatch.setattr(acquisition, 'download_ihdp', forbidden)
    monkeypatch.setattr(loader, 'load_ihdp', forbidden)
    monkeypatch.setattr(loader, 'load_ihdp_truth', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    # Even old deployment settings cannot opt this entrypoint into raw data.
    monkeypatch.setenv('CAUSALLENS_ENABLE_IHDP_DOWNLOAD', 'true')
    monkeypatch.setenv('CAUSALLENS_DATA_DIR', str(tmp_path / 'private-data'))
    real_open = builtins.open
    def guarded_open(file, *args, **kwargs):
        path = str(file)
        if '/data/raw/' in path or '/results/' in path or path.endswith('.npz'):
            forbidden()
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    entry = Path(__file__).resolve().parents[1] / 'deploy' / 'streamlit_app.py'
    at = AppTest.from_file(str(entry), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 6
    assert 'SYNTHETIC DEMONSTRATION' in at.info[0].value
    assert len(at.get('download_button')) == 2
    for name in at.sidebar.selectbox[1].options:
        at.sidebar.selectbox[1].select(name).run()
        assert not at.exception
    at.slider[1].set_value(0).run()
    assert not at.exception
    strategy = at.dataframe[3].value.set_index('Strategy')
    assert strategy.loc['Model-Guided', 'treated_count'] == 0
    at.slider[0].set_value(149).run()
    at.slider[2].set_value(3.0).run()
    at.sidebar.selectbox[0].select(9).run()
    assert not at.exception
    assert not (tmp_path / 'private-data').exists()
    st.cache_data.clear()
