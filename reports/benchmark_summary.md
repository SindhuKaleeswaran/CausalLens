# CausalLens — IHDP Benchmark Evaluation Report

Evaluated across **3 replications** (replications 0 to 2).

| model | pehe_summary | ate_error_summary | policy_risk_mean |
| --- | --- | --- | --- |
| T-Learner | 0.526 +/- 0.126 | 0.112 +/- 0.051 | 0.010966666666666666 |
| X-Learner | 0.584 +/- 0.136 | 0.095 +/- 0.083 | 0.010966666666666666 |
| Doubly Robust (AIPW) | 0.628 +/- 0.145 | 0.111 +/- 0.082 | 0.010966666666666666 |
| IPW | 0.656 +/- 0.133 | 0.116 +/- 0.085 | 0.010966666666666666 |
| Naive (Unadjusted) | 0.907 +/- 0.171 | 0.098 +/- 0.053 | 0.010966666666666666 |
| S-Learner | 0.939 +/- 0.163 | 0.248 +/- 0.107 | 0.010966666666666666 |


*Note: PEHE is root mean squared error between predicted individual treatment effect and ground truth (lower is better).*
