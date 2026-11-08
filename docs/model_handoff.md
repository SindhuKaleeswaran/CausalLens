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

## Phase 5 data handoff — 2026-11-07

### Source, reproduction and ownership

Use the CFRNet-lineage IHDP-100 train/test NPZ files from the author-hosted URLs in `docs/references.md`. Study-derived covariates/assignment and simulated responses must not be interpreted as direct clinical treatment evidence. `docs/data_dictionary.md` defines canonical x01–x25 and flags uncertain clinical mappings. Source hashes and exact schema are in `docs/data_manifest.json`; no publisher-authenticated checksums or separate redistribution license have been verified. Keep archives in ignored `data/raw/`.

From the repository root in the installed environment:

```bash
python scripts/download_ihdp.py --expected-manifest docs/data_manifest.json
python scripts/validate_ihdp.py
python -m pytest -q
```

The validation command audits replication 0, then 1/2/9, then the remainder. Reports contain aggregate records, not patient arrays. Record the commit SHA, environment, hashes, replication, split and preprocessing flag with every experiment. There is no random operation in this data QA; no seed is needed. Estimator seeds remain the modeling developer's responsibility.

### Raw or optional scaling, unchanged estimator interface

```python
from causallens.data.preprocess import load_preprocessed_ihdp
from causallens.data.loader import load_ihdp_truth
from causallens.data.validate import validate_truth

train, test = load_preprocessed_ihdp(0, standardize=False, data_dir="data/raw")
# Existing estimator code uses train['X'], train['T'], train['Y'].
# Enable standardize=True only when required; x01–x06 use training statistics.
# Binary columns, including x14={1,2}, and T/Y remain unchanged.

# Evaluation/QA only, after fitting and tuning decisions are fixed:
truth = load_ihdp_truth(0, "test", data_dir="data/raw")
targets = validate_truth(test, truth)
print(targets['ate_true'], targets['att_true'])
```

Ground-truth validation is available now; earlier references to “Phase 7” describe the model team's evaluation workflow, not an access requirement. Never use truth in preprocessing, estimator fitting, model selection or propensity estimation. If creating validation data, split only the official training partition first and fit transformations on its reduced training subset.

### Ground-truth semantics and alignment

`tau_i = mu1_i - mu0_i` is an expected conditional contrast, not the noisy realized y1−y0. ATE averages tau over the explicitly chosen population; ATT restricts to treated units and ATC to controls. Empty subgroups return None, not zero. Differences across split-specific effects are expected; scalar archive `ate=4` is not a replacement for these targets.

Factual residual is Y−mu_T; counterfactual residual is ycf−mu_(1−T). Noise is retained and no exact equality to expected outcomes is required. Known simulated targets enable evaluation even where observational identification is difficult; they do not establish positivity or clinical generalization.

The audit compares every loaded source field against the exact corresponding archive split and replication. Shapes, dtypes and feature order are verified through this source comparison. Bare arrays have no semantic column names or patient IDs: callers must preserve order. A consistent joint permutation of oracle arrays could pass shape/contrast checks alone; the source-slice checks are therefore essential. Absence of shared exact X/T fingerprints cannot certify patient-level disjointness. Never match row indices across replications or pool simulations as independent cohorts.

### Quality evidence and remaining limits

`reports/data_quality_report.md` and `reports/data_quality_results.json` record all 200 split-level results and archive hashes. Checks cover finite values, exact schema, binary encodings, array/source alignment, expected effects, raw-mode preservation, training-only scaling, unchanged T/Y and immutable archive bytes. Numeric correctness does not settle the model/inference findings in `docs/pre_phase5_audit.md`; those remain Lenin's review responsibility. No estimator mathematics or dashboard code was changed by Phase 5.

### Phase 5 execution ledger

On 2026-11-07, `python scripts/validate_ihdp.py` produced 200 passing split records, zero replication exceptions, and identical before/after archive hashes. Replication-0 ATE targets were 4.011571487963278 (train) and 4.05634575318767 (test); these are simulator targets, not estimated clinical effects.

`python -m pytest -q --ignore=tests/test_dashboard_interactions.py` passed **102 tests**, with no skips. An attempted full-suite run stopped at collection because Streamlit is absent from this local environment; no dashboard validation is claimed for Phase 5. The earlier audit's dashboard evidence remains separately documented. Whitespace checks passed. No raw archives or executed row-level notebook outputs are included in these changes.
