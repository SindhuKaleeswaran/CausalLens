"""Causal estimation models package.

Provides modular causal estimators with explicit identification assumptions:
- Naive difference-in-means baseline
- Meta-learners: S-Learner, T-Learner, X-Learner
- Propensity score weighting: IPWEstimator
- Doubly Robust / AIPW: DoublyRobustEstimator
"""
from causallens.models.base import BaseCausalEstimator
from causallens.models.doubly_robust import DoublyRobustEstimator
from causallens.models.ipw import IPWEstimator
from causallens.models.meta_learners import SLearner, TLearner, XLearner
from causallens.models.naive import NaiveEstimator

__all__ = [
    "BaseCausalEstimator",
    "NaiveEstimator",
    "SLearner",
    "TLearner",
    "XLearner",
    "IPWEstimator",
    "DoublyRobustEstimator",
]
