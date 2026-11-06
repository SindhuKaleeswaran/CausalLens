"""CausalLens evaluation package.

Provides ground-truth benchmark metrics, multi-replication experiment runners,
and refutation tests for causal model validation:
- pehe_score, ate_error, policy_value, policy_risk
- evaluate_replication, run_ihdp_benchmark
- refute_placebo_treatment, refute_random_confounder, bootstrap_ate_confidence_interval
"""
from causallens.evaluation.benchmarks import (
    evaluate_replication,
    get_default_models,
    run_ihdp_benchmark,
)
from causallens.evaluation.metrics import (
    ate_error,
    evaluate_estimator,
    pehe_score,
    policy_risk,
    policy_value,
)
from causallens.evaluation.robustness import (
    bootstrap_ate_confidence_interval,
    refute_placebo_treatment,
    refute_random_confounder,
)

__all__ = [
    "pehe_score",
    "ate_error",
    "policy_value",
    "policy_risk",
    "evaluate_estimator",
    "evaluate_replication",
    "run_ihdp_benchmark",
    "get_default_models",
    "refute_placebo_treatment",
    "refute_random_confounder",
    "bootstrap_ate_confidence_interval",
]
