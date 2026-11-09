> Historical six-model report. See the [independent verification and separate eight-model experiment](ASTRA_SCIENTIFIC_VERIFICATION.md) for current results and limitations.

# CausalLens: Integration & Scientific Validation Report

**Project:** CausalLens — Causal ML & Decision Intelligence Platform  
**Lead Contributor / Modeling Owner:** Lenin Athikam  
**Data Lead Contributor:** Sindhu Kaleeswaran  
**Branch:** `feat/unified-ihdp100-benchmark` (created from `main` @ `55f33a4`)  
**Target:** `main`  
**Date:** 2026-11-08  

---

## 1. Scope & Purpose

This report documents the end-to-end integration of Sindhu Kaleeswaran's Phase 5 validated IHDP data pipeline with Lenin Athikam's causal inference estimator suite, evaluation metrics, and interactive Streamlit studio. It addresses open modeling and inference findings from the pre-Phase-5 audit, validates zero data leakage, and establishes reproducible benchmark evidence across all 100 IHDP replications.

---

## 2. Integration Ledger & Resolved Findings

Prior to running the unified benchmark, open review items from [`docs/pre_phase5_audit.md`](../docs/pre_phase5_audit.md) assigned to Lenin were formally resolved and verified:

| Audit Item | File & Component | Issue Description | Resolution & Scientific Justification | Verification Test |
|---|---|---|---|---|
| **F7 (CI Activation)** | `.github/workflows/tests.yml` | Workflow template was prepared in `docs/ci/` but not activated in `.github/workflows/`. | Activated `tests.yml` covering Python 3.11 and 3.13 matrix builds on pushes to `main`, `feat/**`, and pull requests. | Syntax and workflow validation verified. |
| **F9 (Bootstrap Statistic)** | `src/causallens/evaluation/robustness.py:380` | Bootstrap CI called `estimate_ate(X[idx])` (sample mean predicted CATE) without explicit naming or parameterization vs population Hájek. | Added explicit `statistic: {"sample_mean_cate", "training_ate"}` parameter, documented differences, and verified resample stability. | [`tests/test_evaluation.py::test_bootstrap_ate_confidence_interval`](../tests/test_evaluation.py) |
| **F10 (IPW Weighting Fallback)** | `src/causallens/models/ipw.py:145` | When custom outcome estimators rejected `sample_weight`, IPW fell back silently to unweighted regression. | Added explicit `UserWarning` emission and exposed `weights_supported_` metadata in estimator state and `summary()`. | [`tests/test_causal_models.py::test_ipw_fallback_warning_when_sample_weight_unsupported`](../tests/test_causal_models.py) |
| **F11 (AIPW Asymptotic Guarantees)** | `src/causallens/models/doubly_robust.py:46,90` | Docstrings understated conditions required for root-n normality and bias reduction. | Qualified rate double-robustness requirements ($o_P(n^{-1/4})$ nuisance convergence, bounded overlap, propensity clipping limits, and in-sample fallback). | Documentation and assumptions dictionary verified. |
| **F12 (Sensitivity Inversion Rank)** | `src/causallens/evaluation/robustness.py:317` | Direct matrix inversion `np.linalg.inv(M.T @ M)` risked crashing on rank-deficient or collinear designs. | Added rank validation (`np.linalg.matrix_rank`) and graceful fallback to Moore-Penrose pseudo-inverse `np.linalg.pinv(M.T @ M)`. | Tested across rank-deficient covariate designs. |
| **F13 (Feature Guard)** | `src/causallens/models/base.py:60` | Fitted estimators did not enforce feature count matching during prediction, allowing constant naive models to accept invalid dimensions. | Added strict feature-count guard in `BaseCausalEstimator._validate_inputs` when `fitted_` is True. | [`tests/test_causal_models.py::test_feature_count_mismatch_raises_on_predict`](../tests/test_causal_models.py) |

---

## 3. Data Integrity & Leakage Prevention Audit

1. **Split Independence:**
   - Training split ($N=672$) and test split ($N=75$) are loaded independently from the original NPZ archives via `load_ihdp(replication, split)`.
   - As audited in Phase 5, zero shared $(X, T)$ row fingerprints exist between train and test splits.
2. **Oracle Isolation:**
   - Ground truth potential outcomes ($\mu_0, \mu_1, \tau, Y_{cf}$) are accessed exclusively through `load_ihdp_truth(replication, split="test")`.
   - No estimator instance or preprocessing pipeline ever receives oracle arrays during `fit()`, hyperparameter selection, or propensity estimation.
   - Sentinels in `tests/test_audit_integration.py` verify that deliberately corrupted test outcomes and oracle arrays do not alter model fits or predictions.
3. **Training-Only Preprocessing:**
   - When feature standardization is enabled, continuous feature statistics ($\mu, \sigma$) are computed strictly on training rows ($X[:, :6]$) and applied unchanged to test rows. Binary features ($X[:, 6:]$) and responses remain value-preserving.

---

## 4. Verification & Test Suite Summary

The entire suite was executed against the local virtual environment:

```
Platform: Windows (Python 3.13.12, scikit-learn 1.9.1, NumPy 2.5.4, pytest 9.2.0)
Total test items: 110
Passed: 110
Failed: 0
Skipped: 0
Runtime: 104.2 seconds
```

### Component Breakdown
- **Data Pipeline & Acquisition (`tests/test_data_pipeline.py`):** 31 tests passed.
- **Preprocessing Pipeline (`tests/test_preprocess.py`):** 23 tests passed.
- **Ground Truth QA (`tests/test_ground_truth.py`):** 8 tests passed.
- **Causal Estimators (`tests/test_causal_models.py`):** 18 tests passed (including new F10 and F13 tests).
- **Evaluation & Robustness (`tests/test_evaluation.py`):** 13 tests passed (including new F9 test).
- **Audit Integration & Sentinels (`tests/test_audit_integration.py`):** 9 tests passed.
- **Dashboard Headless Interactions (`tests/test_dashboard_interactions.py`):** 6 tests passed.
- **Exploratory Data Analysis (`tests/test_eda.py`):** 2 tests passed.

---

## 5. Unified 100-Replication Benchmark Verification

The unified benchmark was executed across all 100 IHDP replications (`replications 0 through 99`) evaluating 6 causal estimators:
- Naive (Unadjusted)
- S-Learner
- T-Learner
- X-Learner
- IPW + Weighted Regression
- Doubly Robust (AIPW)

### Key Observations:
- **Zero Failures:** 600 out of 600 fits completed successfully (`n_successful=100` for all 6 models).
- **Metric Distinctions:** Root-PEHE ($\sqrt{\epsilon_{PEHE}}$) and squared PEHE ($\epsilon_{PEHE}$) are explicitly distinguished in [`results/ihdp100_summary.csv`](../results/ihdp100_summary.csv) and [`results/ihdp100_unified_results.csv`](../results/ihdp100_unified_results.csv).
- **Heterogeneity Gap:** S-Learner with linear base models predicts constant effects due to the absence of feature-treatment interactions, explaining its Root-PEHE of 5.735 vs T-Learner's 2.265.
- **Decision Value:** Causal estimators achieve policy risks $< 0.092$, whereas unadjusted naive decisions incur a policy risk of $1.263$ ($16.4\times$ higher regret).

---

## 6. Remaining Boundaries & Operational Limitations

1. **IHDP Data Redistribution & Patient Identity:**
   - In accordance with the pre-Phase-5 audit, author-hosted CFRNet lineage files match local SHA-256 baselines, but do not provide publisher cryptographic signatures or verified clinical patient mappings.
2. **Clinical Applicability:**
   - IHDP outcomes are semi-synthetic benchmark responses. Treatment effects reported here evaluate algorithm mathematics, not real clinical medicine.
3. **Nonparametric Double Robustness:**
   - Doubly Robust AIPW point estimates are valid under rate conditions, but point-wise CATE intervals require further non-parametric rate bounding under severe overlap trimming.
