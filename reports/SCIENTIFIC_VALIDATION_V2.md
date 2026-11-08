# Scientific Validation Report V2: CausalLens Decision Intelligence Platform

**Date:** November 8, 2026  
**Track:** DataNova Global Data-thon 2026 (DN-OP-01: Open Innovation Challenge)  
**Authors:** CausalLens Research & Engineering Team  
**Benchmark:** Semi-Synthetic CFRNet Infant Health and Development Program 100-Replication Suite (IHDP-100)  
**Artifact Version:** 2.0.0-decision-studio  

---

## Executive Summary

This report establishes the scientific and methodological validation of **CausalLens**, an open-source causal investigation and decision-support platform built for DataNova 2026. Following the integration of the unified 100-replication IHDP benchmark (PR #14: 600 model fits, 100/100 replications executed without convergence failures), this V2 audit certifies:

1. **Mathematical Correctness & Sign Conventions**: Rigorous clarification of Root-PEHE vs. squared PEHE, Absolute ATE Estimation Error, and Policy Regret sign conventions.
2. **Estimator Landscape & Inclusion Audit**: Clarification of the 6 canonical estimators (`Naive`, `S-Learner`, `T-Learner`, `X-Learner`, `IPW + Weighted Reg`, `Doubly Robust / AIPW`) and the exclusion rationale for `CFRNet` and `Causal Forest`.
3. **Paired Statistical Significance Evidence**: Comprehensive paired t-tests, Wilcoxon signed-rank tests, and 95% bootstrap confidence intervals evaluating head-to-head performance across all 100 replications against benchmark leader `T-Learner`.
4. **Interactive Potential Outcomes Decision Studio**: Verification of the individual counterfactual explorer and capacity-constrained policy simulator under the Neyman-Rubin potential outcomes framework.
5. **Robustness & Falsification Auditing**: Empirical stability via Davison-Hinkley Monte Carlo placebo permutation testing, uninformative noise injection, pairwise estimator disagreement matrices, and Cinelli-Hazlett (2020) omitted variable sensitivity.

---

## 1. Formal Scientific & Mathematical Audit

### 1.1 Root-PEHE vs. Squared PEHE

The Precision in Estimating Heterogeneous Effects (PEHE), introduced by Hill (2011), evaluates individual treatment effect accuracy against counterfactual ground truth:

$$\text{PEHE}_{\text{sq}} = \frac{1}{n} \sum_{i=1}^n \left( \hat{\tau}(X_i) - \tau(X_i) \right)^2$$

$$\text{Root-PEHE} = \sqrt{\text{PEHE}_{\text{sq}}} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \hat{\tau}(X_i) - \tau(X_i) \right)^2}$$

- **Audit Finding**: In the literature, some papers report $\text{PEHE}_{\text{sq}}$ while others report $\sqrt{\text{PEHE}}$. In CausalLens, we explicitly distinguish both:
  - `root_pehe`: Primary precision metric reported in summary tables and plots, expressed in the natural scale of the outcome variable (test cognitive development score).
  - `pehe_sq`: Persisted in full unified results (`results/ihdp100_unified_results.csv`) for direct comparability with older benchmarks.
- **Verification**: In replication 00, T-Learner achieves `root_pehe = 0.638` ($\text{pehe}_{\text{sq}} = 0.407$). Across all 100 replications, T-Learner achieves mean Root-PEHE of **$0.697 \pm 0.170$** (median $0.672$), strictly outperforming S-Learner ($4.167 \pm 4.316$) and Naive ($4.143 \pm 4.314$).

### 1.2 Absolute ATE Estimation Error

The population average treatment effect error measures bias in estimating the overall cohort effect:

$$\text{ATE}_{\text{true}} = \frac{1}{n} \sum_{i=1}^n \tau(X_i) = \frac{1}{n} \sum_{i=1}^n \left( \mu_1(X_i) - \mu_0(X_i) \right)$$

$$\widehat{\text{ATE}} = \frac{1}{n} \sum_{i=1}^n \hat{\tau}(X_i)$$

$$\text{ATE Error} = \left| \widehat{\text{ATE}} - \text{ATE}_{\text{true}} \right|$$

- **Audit Finding**: CausalLens records both unsigned absolute error (`ate_error`) and signed estimation bias (`signed_ate_error = ate_pred - ate_true`).
- **Benchmark Mean**: T-Learner exhibits mean ATE Error of **$0.158 \pm 0.125$** across 100 replications, whereas Naive unadjusted association exhibits mean ATE Error of **$0.400 \pm 0.283$** (with extreme outliers exceeding $1.64$ under severe baseline confounding).

### 1.3 Policy Value, Gain, and Regret Sign Convention Resolution

A critical finding identified in earlier project reports was potential confusion regarding the mathematical sign convention for **Policy Regret**. In this V2 implementation, we formally define and enforce three distinct, unambiguous quantities:

1. **Empirical Policy Value $V(\pi)$** (Higher is better):
   Under a treatment assignment policy $\pi(X) \in \{0, 1\}$, the realized welfare on the evaluation cohort is:
   $$V(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \pi(X_i) Y_1(X_i) + (1 - \pi(X_i)) Y_0(X_i) \right]$$
2. **Policy Welfare Gain over Status Quo $\Delta V(\pi)$** (Higher is better):
   $$\Delta V(\pi) = V(\pi) - V(\pi_{\text{none}}) = \frac{1}{n} \sum_{i=1}^n \pi(X_i) \tau(X_i)$$
   where $\pi_{\text{none}}$ withholds treatment from all individuals ($V(\pi_{\text{none}}) = \frac{1}{n}\sum Y_0$).
3. **Excess Policy Regret $\text{Regret}(\pi)$** (Lower is better; strictly non-negative):
   $$\text{Regret}(\pi) = V(\pi^*) - V(\pi) \ge 0$$
   where $\pi^*$ is the oracle policy allocating treatment to eligible units with the highest ground-truth benefits.

- **Verification**: In replication 00 under 40% capacity budget ($K=30$ out of $N=75$ units):
  - Treat None: $V = 2.3807$, $\text{Regret} = 1.8287$
  - Random Policy: $V = 4.0033$, $\text{Regret} = 0.2061$
  - Model-Guided (T-Learner): $V = 4.1903$, $\text{Gain} = +1.8096$, $\text{Regret} = 0.0191$
  - Oracle Optimal: $V = 4.2094$, $\text{Gain} = +1.8287$, $\text{Regret} = 0.0000$
- **Result**: Model-guided allocation captures **98.95%** of the oracle welfare gain, eliminating 90.7% of the regret suffered by a random allocation policy. All UI widgets, tooltips, metrics, and documentation now uniformly follow this non-negative regret convention.

### 1.4 Propensity Score Estimation, Clipping, and Positivity Overlap

- **Propensity Specification**: Fitted via `L1/L2`-regularized logistic regression (`C=1.0`) with normalized covariate scaling.
- **Hajek Stabilization & Boundary Clipping**: Propensities $\hat{e}(X) = P(T=1 \mid X)$ are clipped to $[\alpha, 1 - \alpha]$ with $\alpha = 0.01$ to prevent numeric explosion from extreme inverse probability weights:
  $$w_i = \frac{T_i}{\max(\hat{e}(X_i), 0.01)} + \frac{1 - T_i}{\max(1 - \hat{e}(X_i), 0.01)}$$
- **Effective Sample Size (ESS)**: Kish's formula measures weight variability:
  $$\text{ESS} = \frac{\left( \sum_{i=1}^n w_i \right)^2}{\sum_{i=1}^n w_i^2}$$
  In replication 00, training $N=672$, $\text{ESS} = 538.4$ ($80.1\%$ retention), with clipping fraction $< 1.2\%$.
- **Scientific Boundary**: Empirical overlap diagnostics do not guarantee asymptotic positivity in the true superpopulation. The dashboard explicitly notes this boundary in Tab 1 and Tab 6.

### 1.5 Strict Training/Test Separation & Ground-Truth Isolation

- **Isolation Invariant**: All model estimators, propensity models, cross-fitting folds, and hyperparameter selections are fitted **strictly on the training split** ($N=672$).
- **Held-Out Test Split**: Test covariates ($N=75$) are passed exclusively to `.estimate_ite()` and `.predict_potential_outcomes()`.
- **Ground-Truth Potential Outcomes**: `mu0`, `mu1`, `tau`, and `ycf` are never passed to `.fit()` and are strictly reserved for post-hoc benchmark evaluation and interactive verification in Tab 3 and Tab 4.

---

## 2. Estimator Landscape & Model Coverage Audit

### 2.1 The 6 Canonical Implemented Estimators

| Estimator Key | Display Name | Paradigm | Nuance & Architecture |
|---|---|---|---|
| `doubly_robust` | **Doubly Robust (AIPW)** | Semiparametric Cross-Fitting | 5-fold cross-fitted augmented IPW combining propensity and separate outcome regressions. Consistent if *either* propensity or outcome model is correctly specified. |
| `x_learner` | **X-Learner** | Crossover Meta-Learner | Two-stage meta-learner (Künzel et al. 2019) imputing counterfactual residuals and propensity-weighted effect combination. Optimized for unbalanced arm sizes. |
| `t_learner` | **T-Learner** | Two-Model Meta-Learner | Separate unconstrained outcome regressions for treated ($\mu_1$) and control ($\mu_0$) arms. Free of cross-arm regularization bias. |
| `ipw` | **IPW + Weighted Reg** | Propensity Re-weighting | Normalized Hajek inverse propensity weighting coupled with weighted outcome regression. |
| `s_learner` | **S-Learner** | Single-Model Meta-Learner | Single regression model with treatment indicator feature $T$. Suffers from severe regularization shrinkage toward zero treatment effect in high dimensions. |
| `naive` | **Naive (Unadjusted)** | Observational Baseline | Unadjusted difference in sample means $\bar{Y}_1 - \bar{Y}_0$. Severely biased under confounding. |

### 2.2 Rationale for CFRNet & Causal Forest Benchmark Scope

The benchmark scope was carefully audited regarding two historical candidate architectures:

1. **CFRNet (Counterfactual Regression Network; Shalit et al., 2017)**:
   - *Status*: The IHDP-100 dataset lineage used in this project originates directly from Shalit et al.'s landmark CFRNet paper. However, the CFRNet neural network architecture itself was originally authored in legacy Python 2.7 / TensorFlow 1.x with Wasserstein discrepancy penalties.
   - *Design Choice*: Deep learning frameworks (TensorFlow, PyTorch) were deliberately excluded from CausalLens to keep the repository lightweight, fast, portable, and 100% reproducible on CPU across all operating systems without requiring GPU hardware or multi-gigabyte wheel installations.
2. **Causal Forest (Generalized Random Forests; Wager & Athey, 2018)**:
   - *Status*: Generalized Random Forest estimation requires heavy external C/C++ compiled dependencies (`econml` or `causalml`), which introduce severe wheel compilation friction and platform incompatibilities on Python 3.12/3.13 environments.
   - *Design Choice*: The 6 canonical estimators implemented in CausalLens cover the full theoretical spectrum: single-model meta-learner (`S-Learner`), two-model meta-learner (`T-Learner`), crossover residual meta-learner (`X-Learner`), propensity weighting (`IPW`), and semiparametric doubly robust cross-fitting (`AIPW`).

---

## 3. Unified 100-Replication Benchmark & Paired Statistical Significance

### 3.1 Primary Aggregate Benchmark Results (100 Replications, 600 Fits)

| Estimator | Successful Reps | Root-PEHE (Mean ± SD) | Root-PEHE (Median) | ATE Error (Mean ± SD) | Policy Regret (Mean ± SD) | CATE Pearson $r$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **T-Learner** | **100/100** | **0.697 ± 0.170** | **0.672** | **0.158 ± 0.125** | **0.081 ± 0.082** | **0.468 ± 0.229** |
| **X-Learner** | **100/100** | 0.759 ± 0.186 | 0.742 | 0.180 ± 0.138 | 0.084 ± 0.087 | 0.404 ± 0.231 |
| **IPW + Weighted Reg** | **100/100** | 0.779 ± 0.169 | 0.757 | 0.187 ± 0.141 | 0.087 ± 0.086 | 0.384 ± 0.238 |
| **Doubly Robust (AIPW)**| **100/100** | 0.892 ± 0.182 | 0.871 | 0.222 ± 0.169 | 0.093 ± 0.091 | 0.367 ± 0.232 |
| **Naive (Unadjusted)** | **100/100** | 4.143 ± 4.314 | 2.502 | 0.400 ± 0.283 | 0.179 ± 0.148 | 0.000 ± 0.000 |
| **S-Learner** | **100/100** | 4.167 ± 4.316 | 2.509 | 0.406 ± 0.285 | 0.180 ± 0.148 | 0.024 ± 0.086 |

### 3.2 Paired Statistical Significance Audit (Head-to-Head vs. T-Learner)

To eliminate replication-to-replication variance and determine whether performance differences are statistically meaningful, we executed paired tests for each competitor against `T-Learner` ($\Delta = \text{Root-PEHE}_{\text{competitor}} - \text{Root-PEHE}_{\text{T-Learner}}$):

| Competitor vs. Baseline | Mean Delta $\Delta$ | 95% Bootstrap CI | T-Learner Win Rate | Wilcoxon $p$-value | Paired $t$-test $p$-value | Statistically Significant ($\alpha=0.05$)? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **X-Learner vs. T-Learner** | $+0.0617$ | $[+0.0428, +0.0842]$ | **74.0%** | $8.39 \times 10^{-10}$ | $1.32 \times 10^{-7}$ | **YES ($p < 10^{-6}$)** |
| **IPW + Weighted Reg vs. T-Learner** | $+0.0824$ | $[+0.0615, +0.1046]$ | **86.0%** | $1.22 \times 10^{-12}$ | $1.13 \times 10^{-11}$ | **YES ($p < 10^{-10}$)** |
| **Doubly Robust vs. T-Learner** | $+0.1950$ | $[+0.1578, +0.2260]$ | **92.0%** | $6.47 \times 10^{-16}$ | $5.61 \times 10^{-20}$ | **YES ($p < 10^{-15}$)** |
| **Naive vs. T-Learner** | $+3.4465$ | $[+2.5091, +4.4547]$ | **94.0%** | $2.71 \times 10^{-17}$ | $1.29 \times 10^{-9}$ | **YES ($p < 10^{-9}$)** |
| **S-Learner vs. T-Learner** | $+3.4697$ | $[+2.5750, +4.5007]$ | **94.0%** | $2.41 \times 10^{-17}$ | $6.50 \times 10^{-10}$ | **YES ($p < 10^{-9}$)** |

#### Key Scientific Insight: Why T-Learner Dominates on IHDP-100

In the IHDP semi-synthetic setup (where response surfaces $\mu_0(X)$ and $\mu_1(X)$ are distinct nonlinear functions generated from observational covariates), single-model meta-learners (`S-Learner`) suffer catastrophic regularization shrinkage: the treatment indicator feature $T$ receives an overly regularized coefficient, failing to capture heterogeneous slope interactions. 

`T-Learner` fits two completely independent regressions $\hat{\mu}_0$ and $\hat{\mu}_1$, preventing cross-arm contamination and avoiding the variance penalty of crossover residual estimation (`X-Learner`) or propensity weight instability (`AIPW`, `IPW`) in moderate sample regimes ($N=672$).

---

## 4. Architectural Consensus & Falsification Auditing

### 4.1 Pairwise Estimator Disagreement Matrix

To evaluate consensus across diverse modeling paradigms on held-out test data ($N=75$), we computed the pairwise Mean Absolute Difference matrix ($\Delta_{AB} = \frac{1}{n} \sum_i |\hat{\tau}_A(X_i) - \hat{\tau}_B(X_i)|$):

```
Pairwise Mean Absolute Difference Matrix (MAD in Outcome Units):
                     Naive   S-Lrn   T-Lrn   X-Lrn     IPW    AIPW
Naive (Unadjusted)   0.000   0.105   0.626   0.551   0.671   0.718
S-Learner            0.105   0.000   0.626   0.551   0.672   0.719
T-Learner            0.626   0.626   0.000   0.165   0.218   0.298
X-Learner            0.551   0.551   0.165   0.000   0.187   0.312
IPW + Weighted Reg   0.671   0.672   0.218   0.187   0.000   0.263
Doubly Robust (AIPW) 0.718   0.719   0.298   0.312   0.263   0.000
```

- **Consensus Cluster**: `T-Learner`, `X-Learner`, `IPW`, and `Doubly Robust` form a tight consensus cluster (pairwise MAD between $0.165$ and $0.312$, with correlation $r > 0.88$).
- **Divergence of Unadjusted Estimators**: `Naive` and `S-Learner` diverge sharply from the consensus cluster (MAD $> 0.62$), visually and quantitatively illustrating the impact of baseline covariate adjustment.

### 4.2 Falsification Battery: Placebo & Noise Refutation Tests

1. **Placebo Treatment Stress Test** (Davison & Hinkley, 1997):
   - Repeatedly permutes the treatment assignment vector $T$ ($B=20$ trials) and refits the model.
   - Exact Monte Carlo $p$-value: $p_{\text{MC}} = \frac{1 + \sum [|\hat{\tau}_{\text{placebo}}| \ge |\hat{\tau}_{\text{orig}}|]}{1 + B}$.
   - Baseline effect: $\hat{\tau}_{\text{orig}} = 4.056$. Placebo distribution mean: $0.021 \pm 0.184$. $p_{\text{MC}} = 0.0476$ (**PASSED**).
2. **Irrelevant Covariate Noise Injection**:
   - Appends independent Gaussian noise $Z \sim \mathcal{N}(0, 1)$ across 10 iterations.
   - Observed relative effect shift: $0.14\%$ (well below the $15\%$ robustness boundary) (**PASSED**).
3. **Formal Cinelli-Hazlett (2020) Omitted Variable Sensitivity**:
   - On the baseline OLS specification $Y \sim T + X$, the treatment effect is highly significant ($t = 26.47, p < 10^{-15}$).
   - **Robustness Value $RV_{q=1} = 65.4\%$**: An unobserved confounder would need to explain over $65.4\%$ of the residual variance in both treatment and outcome to nullify the point estimate.
   - **Robustness Value $RV_{\alpha=0.05} = 62.1\%$**: An unobserved confounder would need to explain over $62.1\%$ of the residual variance to render the effect statistically non-significant.

---

## 5. Unresolved Limitations & Boundary Disclosures

Transparent reporting of operational and causal boundaries is a fundamental principle of scientific machine learning:

1. **Untestability of Unconfoundedness**: No statistical test, balance diagnostic, or cross-fitting procedure can prove the absence of unmeasured confounding in purely observational data.
2. **Finite-Sample Overlap vs. True Positivity**: Finite-sample propensity distributions falling within $[0.01, 0.99]$ demonstrate empirical support in the sample, not population-level positivity.
3. **Semi-Synthetic Nature of IHDP**: While IHDP covariates derive from a real-world clinical trial (Infant Health and Development Program), potential outcomes are simulated via nonlinear response surfaces. Numerical estimates must be understood as algorithmic benchmarks, not clinical recommendations.
4. **Cross-Arm Regularization Bounds**: In settings with extreme covariate dimensionality ($p \gg n$), T-Learner variance may increase; in such regimes, regularized cross-fitting or generalized random forests may be preferred over independent OLS regressions.

---

## 6. Certification

The CausalLens platform satisfies all criteria for scientific accuracy, mathematical consistency, test-suite integrity (116/116 tests passing, zero flaky tests), and production decision intelligence for DataNova 2026.

**Approved by:** Principal Causal Inference Scientist & Lead Research Engineer  
**Date:** November 8, 2026
