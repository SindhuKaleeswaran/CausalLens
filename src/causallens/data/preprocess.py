"""Minimal observed-data preprocessing; raw covariates are the default."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .loader import load_ihdp


def _observed(data):
    """Reject oracle/extra fields and return independent, validated arrays."""
    if set(data) != {'X', 'T', 'Y'}:
        raise ValueError('Expected exactly X, T, Y; oracle and extra fields are forbidden')
    out = {k: np.array(data[k], copy=True) for k in ('X', 'T', 'Y')}
    X, T, Y = out['X'], out['T'], out['Y']
    if X.ndim != 2 or X.shape[1] != 25 or not len(X):
        raise ValueError('X must have shape (n, 25), n > 0')
    if T.shape != (len(X),) or Y.shape != (len(X),):
        raise ValueError('T and Y must have shape (n,) aligned with X')
    for a in out.values():
        if a.dtype.kind not in 'biuf' or not np.isfinite(a).all():
            raise ValueError('All observed values must be finite numeric values; no automatic imputation')
    if not np.isin(T, [0, 1]).all():
        raise ValueError('T must use binary labels 0/1')
    for j in range(6, 25):
        levels = [1, 2] if j == 13 else [0, 1]
        if not np.isin(X[:, j], levels).all():
            raise ValueError(f'Unexpected binary encoding in x{j+1:02d}')
    return out


class IHDPPreprocessor:
    """Fit optional scaling on explicitly supplied training observations only.

    Only X[:, :6] contributes to fitted statistics. T/Y are checked and copied,
    never used in fitting. The caller must supply the correct training subset;
    bare arrays cannot prove their split or replication provenance. Prefer
    load_preprocessed_ihdp for official train/test orchestration.
    """
    def __init__(self, *, standardize=False):
        if not isinstance(standardize, bool):
            raise ValueError('standardize must be a bool')
        self.standardize = standardize

    def fit(self, train):
        # A failed refit must not leave an older usable fitted state.
        self._fitted = False
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize must be a bool")
        data = _observed(train)
        self._standardize = self.standardize
        self.scaler_ = StandardScaler().fit(data['X'][:, :6]) if self._standardize else None
        self._fitted = True
        return self

    def transform(self, observed):
        if not getattr(self, '_fitted', False):
            raise ValueError('Fit on training data before transform')
        if not isinstance(self.standardize, bool) or self.standardize != self._standardize:
            raise ValueError('Configuration changed after fit; refit on training data')
        out = _observed(observed)
        if self.scaler_ is not None:
            X = out['X'].astype(np.float64, copy=True)
            X[:, :6] = self.scaler_.transform(X[:, :6])
            out['X'] = X
        return out

    def fit_transform(self, train):
        return self.fit(train).transform(train)


def load_preprocessed_ihdp(replication=0, *, standardize=False, data_dir=None):
    """Return (train, test), each exactly {'X', 'T', 'Y'}.

    A fresh processor is fitted on this replication's official training split.
    No state is shared across calls/replications. No validation split is created.
    For a future validation subset, split training first, then use the class
    directly to fit only on the reduced training subset.
    """
    train = load_ihdp(replication, 'train', data_dir)
    test = load_ihdp(replication, 'test', data_dir)
    processor = IHDPPreprocessor(standardize=standardize)
    return processor.fit_transform(train), processor.transform(test)
