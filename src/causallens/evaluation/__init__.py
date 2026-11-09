"""CausalLens evaluation package.

Provides ground-truth benchmark metrics, multi-replication experiment runners,
and scientific refutation tests for causal model validation:
- pehe_score, ate_error, policy_value, policy_risk
- evaluate_replication, run_ihdp_benchmark
- refute_placebo_treatment, refute_irrelevant_covariate, refute_subsample_stability
- compute_estimator_disagreement, sensitivity_unobserved_confounding, bootstrap_ate_confidence_interval
"""
from causallens.evaluation.benchmarks import (
    aggregate_benchmark_results,
    compute_bootstrap_ci,
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
from causallens.evaluation.policy import (
    compute_paired_benchmark_differences,
    evaluate_policy_rule,
    simulate_capacity_constrained_policy,
)
from causallens.evaluation.robustness import (
    bootstrap_ate_confidence_interval,
    compute_estimator_disagreement,
    refute_irrelevant_covariate,
    refute_placebo_treatment,
    refute_subsample_stability,
    sensitivity_unobserved_confounding,
)

__all__ = [
    "pehe_score",
    "ate_error",
    "policy_value",
    "policy_risk",
    "evaluate_policy_rule",
    "simulate_capacity_constrained_policy",
    "compute_paired_benchmark_differences",
    "evaluate_estimator",
    "evaluate_replication",
    "run_ihdp_benchmark",
    "aggregate_benchmark_results",
    "compute_bootstrap_ci",
    "get_default_models",
    "refute_placebo_treatment",
    "refute_irrelevant_covariate",
    "refute_subsample_stability",
    "compute_estimator_disagreement",
    "sensitivity_unobserved_confounding",
    "bootstrap_ate_confidence_interval",
]
