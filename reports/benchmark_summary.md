# CausalLens — IHDP Benchmark Evaluation Report

Evaluated across **100 replications** (replications 0 to 99).

| model | n_successful | pehe_summary | ate_error_summary | policy_risk_mean | avg_runtime_sec |
| --- | --- | --- | --- | --- | --- |
| T-Learner | 100 | 2.265 +/- 3.997 | 0.247 +/- 0.380 | 0.077118 | 0.0034859999999999995 |
| X-Learner | 100 | 2.327 +/- 3.994 | 0.272 +/- 0.394 | 0.08213199999999998 | 0.019903 |
| IPW + Weighted Reg | 100 | 2.348 +/- 3.944 | 0.255 +/- 0.348 | 0.08128099999999999 | 0.017203000000000003 |
| Doubly Robust (AIPW) | 100 | 2.460 +/- 3.918 | 0.276 +/- 0.325 | 0.09145199999999999 | 0.09848599999999998 |
| Naive (Unadjusted) | 100 | 5.712 +/- 8.948 | 0.718 +/- 1.671 | 1.262956 | 0.00028300000000000005 |
| S-Learner | 100 | 5.735 +/- 8.855 | 0.797 +/- 1.132 | 1.276817 | 0.0027430000000000006 |

*Note: PEHE is root mean squared error between predicted individual treatment effect and ground truth (lower is better).*
