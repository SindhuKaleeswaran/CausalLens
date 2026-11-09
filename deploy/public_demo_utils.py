"""Presentation diagnostics for independently generated public examples only."""
import numpy as np
import pandas as pd

from causallens.synthetic_demo import FEATURES


def synthetic_balance(train):
    """Signed SMD with pooled within-arm SD, matching the research display."""
    treated = train['X'][train['T'] == 1]
    control = train['X'][train['T'] == 0]
    denominator = np.sqrt((treated.var(axis=0, ddof=1) + control.var(axis=0, ddof=1)) / 2)
    delta = treated.mean(axis=0) - control.mean(axis=0)
    smd = np.divide(delta, denominator, out=np.full_like(delta, np.nan), where=denominator > 0)
    smd[(denominator == 0) & (delta == 0)] = 0
    return pd.DataFrame({'Feature': FEATURES, 'Signed SMD': smd})


def disagreement_matrix(predictions):
    """Mean absolute paired effect differences, in synthetic outcome units."""
    values = np.stack(list(predictions.values()))
    return pd.DataFrame(np.abs(values[:, None, :] - values[None, :, :]).mean(axis=2),
                        index=list(predictions), columns=list(predictions))


def synthetic_profiles(truth, predictions):
    """Select reproducible held-out fictional profiles; no extrapolation or refit."""
    tau = truth['tau']
    return {
        'Choose by index': None,
        'Lowest oracle effect': int(np.argmin(tau)),
        'Near-zero oracle effect': int(np.argmin(np.abs(tau))),
        'Highest oracle effect': int(np.argmax(tau)),
        'Largest estimator disagreement': int(np.argmax(np.std(list(predictions.values()), axis=0))),
    }
