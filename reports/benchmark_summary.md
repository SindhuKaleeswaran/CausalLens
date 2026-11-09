# CausalLens — IHDP Benchmark Evaluation Report

Evaluated across **100 replications** (replications 0 to 99).

| Model | Fits | Root-PEHE Mean ± SD | Root-PEHE 95% CI (Mean) | Root-PEHE Median [95% CI] | ATE Error Mean ± SD | ATE Error 95% CI (Mean) | ATE Error Median [95% CI] | Policy Regret | Avg Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-Learner | 100 | 2.265 +/- 3.997 | [1.568, 3.183] | 0.915 [0.748, 1.053] | 0.247 +/- 0.380 | [0.182, 0.333] | 0.134 [0.110, 0.169] | 0.077118 | 0.011991000000000003 |
| X-Learner | 100 | 2.327 +/- 3.994 | [1.630, 3.248] | 0.985 [0.849, 1.118] | 0.272 +/- 0.394 | [0.204, 0.364] | 0.147 [0.123, 0.187] | 0.08213199999999998 | 0.047793999999999996 |
| IPW + Weighted Reg | 100 | 2.348 +/- 3.944 | [1.660, 3.260] | 0.964 [0.880, 1.143] | 0.255 +/- 0.348 | [0.195, 0.334] | 0.160 [0.129, 0.186] | 0.08128099999999999 | 0.038991000000000005 |
| Doubly Robust (AIPW) | 100 | 2.460 +/- 3.918 | [1.776, 3.375] | 1.095 [1.004, 1.277] | 0.276 +/- 0.325 | [0.219, 0.352] | 0.203 [0.147, 0.237] | 0.09145199999999999 | 0.20837599999999998 |
| Naive (Unadjusted) | 100 | 5.712 +/- 8.948 | [4.109, 7.797] | 2.530 [2.065, 3.090] | 0.718 +/- 1.671 | [0.452, 1.106] | 0.275 [0.191, 0.392] | 1.262956 | 0.000834 |
| S-Learner | 100 | 5.735 +/- 8.855 | [4.138, 7.807] | 2.541 [2.061, 3.191] | 0.797 +/- 1.132 | [0.605, 1.031] | 0.322 [0.251, 0.475] | 1.276817 | 0.010046 |

*Note on Statistical Methodology:*
- Root-PEHE: Root mean squared error between predicted individual treatment effect and ground truth.
- Confidence intervals for the mean and median are computed via 1,000-resample nonparametric percentile bootstrap.
- Policy Regret: Non-negative excess loss V(pi*) - V(pi) relative to the oracle decision rule.
