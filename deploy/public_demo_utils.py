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


def naive_association(outcome, treatment):
    """Difference in mean observed outcome between arms, on the same population as the other KPI cards."""
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment)
    return float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def comparison_interpretation(naive, estimated, oracle, estimator_name):
    """One-paragraph educational comparison; does not assert confounding as the sole cause of any gap."""
    naive_gap = naive - oracle
    estimate_gap = estimated - oracle
    if abs(estimate_gap) < abs(naive_gap) - 1e-9:
        verdict = 'is closer to the simulated effect than the naive association'
    elif abs(estimate_gap) > abs(naive_gap) + 1e-9:
        verdict = 'is not closer to the simulated effect than the naive association on this seed'
    else:
        verdict = "matches the naive association's distance from the simulated effect"
    return (
        f'Naive association {naive:.3f} vs. known simulated effect {oracle:.3f} (gap {naive_gap:+.3f}). '
        f'{estimator_name} ({estimated:.3f}) {verdict} (gap {estimate_gap:+.3f}). '
        'This generator has confounding by design, but sampling variability and model specification also shape '
        'the gap — a single seed is not proof that the estimator removed confounding.'
    )
