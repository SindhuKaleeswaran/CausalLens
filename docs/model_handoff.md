# Model Handoff and Experiment Reproduction Specification

## 1. Data Contract & Input Boundaries

The data interface agreed between the data engineering lead (`SindhuKaleeswaran`) and the causal modeling lead (`leninathikam`):

### Observed Data (Model Inputs)
Access strictly through `causallens.data.loader.load_ihdp(replication, split, data_dir)`:
- **`X` (Covariate Matrix)**:
  - Shape: `(672, 25)` for train; `(75, 25)` for test.
  - Dtype: `float64`, finite, non-null.
  - Canonical column names: `x01` through `x25`.
  - Continuous/standardized features: `x01`–`x06`.
  - Binary/categorical features: `x07`–`x25` (binary indicator columns; `x14` uses `{1.0, 2.0}`).
- **`T` (Treatment Vector)**:
  - Shape: `(672,)` for train; `(75,)` for test.
  - Dtype: `float64` with values strictly in `{0.0, 1.0}`.
  - In replication 0: 123 treated (18.3%), 549 control (81.7%) in train; 16 treated, 59 control in test.
- **`Y` (Observed Realized Outcome / `yf`)**:
  - Shape: `(672,)` for train; `(75,)` for test.
  - Dtype: `float64`, continuous.

### Oracle Data (Strict Evaluation-Only Isolation)
Access strictly through `causallens.data.loader.load_ihdp_truth(replication, split, data_dir)`:
- Keys: `mu0`, `mu1`, `ycf`, `tau = mu1 - mu0`.
- **Leakage Restriction**: Never pass `load_ihdp_truth` outputs into estimator fitting, hyperparameter selection, or propensity estimation. It is used exclusively in Phase 7 to calculate ground-truth benchmark metrics ($\sqrt{\epsilon_{PEHE}}$ and $|\hat{\tau} - \tau_{true}|$).

---

## 2. Implemented Causal Estimators (`causallens.models`)

All estimators inherit from `BaseCausalEstimator` and require explicit identification assumptions:

| Estimator | Class | Mechanism | Core Identification Assumptions | Best Suited For |
|---|---|---|---|---|
| **Naive Baseline** | `NaiveEstimator` | $\bar{Y}_1 - \bar{Y}_0$ | Complete randomization (ignores $X$). | Unadjusted baseline demonstrating selection bias. |
| **S-Learner** | `SLearner` | Single model: $\mu(X, T)$ | Unconfoundedness given $X$, Positivity, SUTVA. | Low-dimensional data; simple baselines. |
| **T-Learner** | `TLearner` | Two models: $\mu_0(X), \mu_1(X)$ | Unconfoundedness given $X$, Positivity, SUTVA. | Balanced treatment arms with distinct response surfaces. |
| **X-Learner** | `XLearner` | 2-stage crossover imputation + propensity weighting | Unconfoundedness, Positivity, SUTVA, crossover validity. | **IHDP benchmark** (highly unbalanced arms: 18% treated). |
| **IPW** | `IPWEstimator` | Propensity score weighting ($w_i$) with Hájek normalization & clipping | Positivity ($0 < e(X) < 1$), correct propensity specification, SUTVA. | Reweighting observational cohorts to balance covariates. |
| **Doubly Robust (AIPW)** | `DoublyRobustEstimator` | Augmented IPW pseudo-outcome ($\Gamma_i$) regression | Positivity, SUTVA, **Double Robustness** (either $\mu$ or $e$ correctly specified). | Robust asymptotic efficiency; gold standard semiparametric inference. |

---

## 3. Estimator API Protocol

```python
from causallens.models import SLearner, TLearner, XLearner, IPWEstimator, DoublyRobustEstimator

# 1. Instantiate
model = DoublyRobustEstimator()

# 2. Fit to observational training split
model.fit(X_train, T_train, Y_train)

# 3. Estimate treatment effects
ate_train = model.estimate_ate()            # Scalar float
cate_test = model.estimate_ite(X_test)      # Array of shape (n_test,)

# 4. Predict counterfactual potential outcomes
y0_hat, y1_hat = model.predict_potential_outcomes(X_test)

# 5. Access formal identification assumptions & summary
print(model.assumptions)
print(model.summary())
```

---

## 4. Experiment Reproduction Protocol

When executing experimental runs across replications:
1. Record:
   - Git Commit SHA (`git rev-parse HEAD`)
   - Platform & Python version (`3.13.x`)
   - Seed (default: `42`)
   - Replication index (0 through 99)
   - Base learner parameters
2. Metrics to record against ground truth (Phase 7):
   - $\sqrt{\epsilon_{PEHE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{\tau}_i - \tau_i)^2}$
   - $\text{ATE Error} = |\hat{\tau}_{ATE} - \tau_{ATE}|$
   - Coverage of Doubly Robust 95% Confidence Intervals
