"""Independent educational simulation: no IHDP inputs, files, or network access."""
import numpy as np
from causallens.models import (DoublyRobustEstimator, IPWEstimator, NaiveEstimator,
                               SLearner, TLearner, XLearner)
from causallens.evaluation.metrics import evaluate_estimator

FEATURES = ['Synthetic feature A', 'Synthetic feature B', 'Synthetic feature C',
            'Synthetic feature D', 'Synthetic feature E']
GENERATOR_VERSION = 'independent-simulation-v1'


def generate_example(seed=0):
    """600 training and 150 held-out units, generated solely from an RNG.

    X ~ N(0,I); p(T=1|X)=0.15+0.7 sigmoid(0.8 A-0.5 B).
    mu0=2+0.8 A+0.5 B^2-0.3 C; tau=0.7+0.9 A-0.6 C.
    Y=mu0+T*tau+epsilon, epsilon~N(0,1). No empirical calibration.
    Oracle arrays are returned separately and never passed to fitting.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 9:
        raise ValueError('Example seed must be an integer from 0 through 9')
    rng = np.random.default_rng(20261109 + int(seed))
    x = rng.normal(size=(750, 5))
    propensity = 0.15 + 0.7 / (1 + np.exp(-(0.8*x[:, 0] - 0.5*x[:, 1])))
    t = rng.binomial(1, propensity).astype(float)
    mu0 = 2 + 0.8*x[:, 0] + 0.5*x[:, 1]**2 - 0.3*x[:, 2]
    tau = 0.7 + 0.9*x[:, 0] - 0.6*x[:, 2]
    y = mu0 + t*tau + rng.normal(size=750)
    train = {'X': x[:600], 'T': t[:600], 'Y': y[:600]}
    test = {'X': x[600:], 'T': t[600:], 'Y': y[600:]}
    truth = {'mu0': mu0[600:], 'mu1': (mu0+tau)[600:], 'tau': tau[600:]}
    return train, test, truth


def fit_example(seed=0):
    train, test, truth = generate_example(seed)
    models = {'Naive (Unadjusted)': NaiveEstimator(), 'S-Learner': SLearner(),
              'T-Learner': TLearner(), 'X-Learner': XLearner(),
              'IPW + Weighted Reg': IPWEstimator(),
              'Doubly Robust (AIPW)': DoublyRobustEstimator()}
    predictions, records = {}, []
    for name, model in models.items():
        model.fit(train['X'], train['T'], train['Y'])
        pred = np.asarray(model.estimate_ite(test['X']), dtype=float)
        predictions[name] = pred
        records.append(evaluate_estimator(name, truth['tau'], pred,
                       float(truth['tau'].mean()), float(pred.mean()),
                       truth['mu0'], truth['mu1']))
    return train, test, truth, predictions, records
