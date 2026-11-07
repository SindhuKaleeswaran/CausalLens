# Phase 4 — minimal configurable preprocessing

Raw covariates are the default. The benchmark already contains finite values and upstream transformed numeric covariates; no additional transformation is scientifically required merely to demonstrate preprocessing.

## Estimator-facing interface

```python
from causallens.data.preprocess import load_preprocessed_ihdp
train, test = load_preprocessed_ihdp(
    replication=0, standardize=False, data_dir="data/raw"
)
# Change only standardize=True if the estimator requires it.
# Each dictionary remains exactly X, T, Y.
# X: (n,25), T: (n,), Y: (n,)
```

A fresh processor is fitted on each selected replication's official training partition and applied unchanged to that replication's test partition. No replications are pooled, partitions reshuffled, or validation split created. Lenin can change the flag without changing estimator inputs.

## Exact behavior

- Raw mode returns independent copies with identical numeric values, dtypes and feature order.
- Optional standardization fits scikit-learn StandardScaler only on training X[:, :6]. It subtracts training means and divides by training population standard deviations (ddof=0). Constant numeric columns use scale 1.
- x07–x25 remain unchanged; x14 retains {1,2}. Numeric X becomes float64 when scaling is enabled. The actual benchmark is already float64.
- T and Y remain unchanged, including their dtypes. Outcomes are never normalized.
- Only the keys X, T, Y are accepted. Oracle and extra fields are rejected. T/Y are validated and copied but do not determine fitted statistics.
- Nonfinite inputs, misaligned shapes, unexpected feature counts, treatment labels or binary encodings raise explicit errors. No imputation, clipping, feature deletion or automatic repair is performed.
- Transform requires a successful fit. Changed configuration requires a new fit. Transform never updates fitted statistics.

## Future validation split

If model tuning requires validation, divide the corresponding official training partition first. Then fit IHDPPreprocessor on only the reduced training subset, and call transform on validation and test. Do not use the convenience function's full-training fit before extracting validation.

```python
from causallens.data.preprocess import IHDPPreprocessor
processor = IHDPPreprocessor(standardize=True)
# reduced_train, validation and test must already be aligned X/T/Y dictionaries.
# train_ready = processor.fit_transform(reduced_train)
# validation_ready = processor.transform(validation)
# test_ready = processor.transform(test)
```

The low-level class cannot infer provenance from bare arrays: callers must supply training rows from the correct replication. The convenience loader enforces official split selection. Ground-truth evaluation remains separate via loader.load_ihdp_truth; no preprocessing code invokes it.

## Historical Phase 4 validation evidence

46 tests passed in the existing Python 3.13.5 environment, including exact raw-mode preservation, independent copies, analytic scaling expectations, shifted-test behavior, binary preservation, T/Y independence, oracle rejection, missing-value rejection, constant columns, reproducibility and per-replication orchestration.

Notebook 03 demonstrates raw versus standardized summaries and explicitly checks parameters against training-only statistics. It perturbs a copy of test features to verify fitted parameters remain fixed. It also exercises both modes on real replications 0 and 1. All four code cells executed successfully. This is Phase 4 implementation validation; full Phase 5 scientific QA across all replications remains pending.

The original archive hashes and upstream preprocessing limitations remain documented in the data manifest and dataset research. Additional train-only scaling cannot undo upstream transformations. No estimator benefit is claimed or chosen using test outcomes.

## Pre-Phase-5 integration clarification

Both `evaluate_replication` and the dashboard call the value-preserving raw loader.
No benchmark configuration has been changed to standardize features. Configuration
must remain a Python bool even after construction; invalid refits invalidate prior fitted state.
Current regression results and limitations are recorded in `pre_phase5_audit.md`.
