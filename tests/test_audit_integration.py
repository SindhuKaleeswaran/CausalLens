"""Separate original-archive QA and artificial loader-to-model integration."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from causallens.data.acquisition import sha256
from causallens.data.loader import archive_path, validate_archive, load_ihdp, load_ihdp_truth
from causallens.data.preprocess import load_preprocessed_ihdp
from causallens.data.eda import balance_table
from causallens.evaluation.benchmarks import evaluate_replication


@pytest.fixture
def real_archives():
    root = Path(os.environ.get('CAUSALLENS_REAL_DATA_DIR', 'data/raw'))
    if not all(archive_path(s, root).exists() for s in ('train', 'test')):
        pytest.skip('Original IHDP archives unavailable')
    return root


def test_real_all_replication_schema_checksum_encoding(real_archives):
    manifest = json.loads(Path('docs/data_manifest.json').read_text())
    for split, n in [('train', 672), ('test', 75)]:
        path = archive_path(split, real_archives)
        assert sha256(path) == manifest['files'][split]['sha256']
        assert validate_archive(path) == manifest['files'][split]['schema']
        with np.load(path, allow_pickle=False) as a:
            assert a['x'].shape == (n, 25, 100)
            for rep in range(100):
                for j in range(6, 25):
                    assert np.isin(a['x'][:, j, rep], [1, 2] if j == 13 else [0, 1]).all()


@pytest.mark.parametrize('rep', [0, 1, 7, 42, 99])
def test_real_preprocessing_and_archive_immutability(real_archives, rep):
    before = {s: sha256(archive_path(s, real_archives)) for s in ('train', 'test')}
    train = load_ihdp(rep, 'train', real_archives)
    for mode in (False, True):
        processed = load_preprocessed_ihdp(rep, standardize=mode, data_dir=real_archives)
        for split, out in zip(('train', 'test'), processed):
            raw = load_ihdp(rep, split, real_archives)
            assert set(out) == {'X', 'T', 'Y'}
            for key in ('T', 'Y'): np.testing.assert_array_equal(out[key], raw[key])
            np.testing.assert_array_equal(out['X'][:, 6:], raw['X'][:, 6:])
            if mode:
                sd = train['X'][:, :6].std(0)
                sd[sd == 0] = 1
                np.testing.assert_allclose(out['X'][:, :6], (raw['X'][:, :6]-train['X'][:, :6].mean(0))/sd, atol=1e-12)
            else:
                for key in raw:
                    np.testing.assert_array_equal(out[key], raw[key])
                    assert out[key].dtype == raw[key].dtype
    assert before == {s: sha256(archive_path(s, real_archives)) for s in before}


def test_real_eda_replication_zero(real_archives):
    d = load_ihdp(0, 'train', real_archives)
    assert int(d['T'].sum()) == 123
    table = balance_table(d['X'], d['T'])
    assert (table.smd.abs() > .1).sum() == 17
    assert round(table.smd.abs().max(), 3) == .380
    assert round(d['Y'][d['T'] == 1].mean()-d['Y'][d['T'] == 0].mean(), 3) == 4.005


def test_synthetic_archive_to_preprocessor_to_all_models(tmp_path):
    rng = np.random.default_rng(814)
    for split, n in [('train', 672), ('test', 75)]:
        X = rng.normal(size=(n, 25, 100))
        X[:, 6:, :] = rng.integers(0, 2, size=(n, 19, 100))
        X[:, 13, :] += 1
        T = np.broadcast_to((np.arange(n) % 2)[:, None], (n, 100))
        mu0 = X[:, 0, :]
        mu1 = mu0 + 2 + X[:, 1, :]
        Y = np.where(T, mu1, mu0) + rng.normal(size=(n, 100))
        np.savez(archive_path(split, tmp_path), x=X, t=T, yf=Y, ycf=np.where(T, mu0, mu1), mu0=mu0, mu1=mu1)
    from causallens.evaluation.benchmarks import get_default_models
    for mode in (False, True):
        train, test = load_preprocessed_ihdp(0, standardize=mode, data_dir=tmp_path)
        for model in get_default_models().values():
            model.fit(**train)
            assert model.estimate_ite(test['X']).shape == (75,)
    result = evaluate_replication(0, data_dir=tmp_path)
    assert (result.status == 'SUCCESS').all()
    target = load_ihdp_truth(0, 'test', tmp_path)['tau'].mean()
    np.testing.assert_allclose(result.ate_true, round(target, 4))


def test_benchmark_fitting_never_receives_test_or_oracle(monkeypatch):
    import causallens.evaluation.benchmarks as b
    from causallens.models.naive import NaiveEstimator
    train = dict(X=np.ones((8, 25)), T=np.arange(8) % 2, Y=np.arange(8.))
    test = dict(X=np.full((3, 25), 77.), T=np.array([0, 1, 0]), Y=np.full(3, -999.))
    monkeypatch.setattr(b, 'load_ihdp', lambda replication, split, data_dir: train if split == 'train' else test)
    truth = dict(mu0=np.zeros(3), mu1=np.full(3, 4.), tau=np.full(3, 4.))
    monkeypatch.setattr(b, 'load_ihdp_truth', lambda replication, split, data_dir: truth)
    original_fit = NaiveEstimator.fit
    fits = []
    def audited_fit(self, X, T, Y):
        for key, value in [('X', X), ('T', T), ('Y', Y)]:
            np.testing.assert_array_equal(value, train[key])
        fits.append(True)
        return original_fit(self, X, T, Y)
    monkeypatch.setattr(NaiveEstimator, 'fit', audited_fit)
    first = b.evaluate_replication(models={'naive': NaiveEstimator()})
    test['Y'][:] = 1e9
    truth.update(mu1=np.full(3, 40.), tau=np.full(3, 40.))
    second = b.evaluate_replication(models={'naive': NaiveEstimator()})
    assert len(fits) == 2
    assert first.ate_pred.iloc[0] == second.ate_pred.iloc[0]
    assert first.ate_true.iloc[0] == 4 and second.ate_true.iloc[0] == 40
