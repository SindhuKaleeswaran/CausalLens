# Dataset card — CFRNet IHDP-100

Status: Phases 1 and 2 completed; Phase 2 measurements recorded in [inspection report](data_inspection_report.md). Sources and open provenance gaps: [references](references.md).

| Item | Decision |
|---|---|
| Intended use | Research demonstration of observational imbalance and simulated treatment-effect evaluation |
| Population origin | Low-birth-weight premature infants in the original IHDP trial; selected benchmark subset |
| Real components | Study-derived baseline covariates and treatment assignments |
| Synthetic components | Factual/counterfactual responses and noiseless potential-outcome targets |
| Release | Author-hosted `ihdp_npci_1-100.train.npz` and `.test.npz` |
| Structure | Separate official train/test arrays per simulation; last axis indexes replication |
| Model boundary | Observed X, T, yf only; oracle fields are evaluation-only |
| Transformations | None added during acquisition/loading; preserve archive values and column order |
| Storage | Ignored `data/raw/`; never upload raw benchmark files |
| Licensing | No standalone archive redistribution grant verified; software MIT license is not assumed to license the data |
| Privacy | No re-identification or clinical deployment; study-derived covariates remain sensitive in interpretation |
| Limits | Semi-synthetic outcomes, selection bias, overlap concerns, uncertain detailed feature coding, no direct clinical generalization |

Do not infer 100 independent cohorts or treat test ground truth as tuning data. Use split-specific tau means for evaluation; do not assume an archive-wide effect constant. The original randomized intervention and this selected simulation benchmark answer different questions. See [dataset research](dataset_research.md) for facts, assumptions and source limitations.

## Measured release dimensions

100 realizations; 672 training and 75 test observations per realization, 25 covariates. Replication 0 has 123 treated/549 controls in train and 16 treated/59 controls in test. All required arrays are float64 and finite. There are six multi-valued numeric and 19 two-level columns in replication 0; x14 uses {1,2}. See the inspection report and manifest for exact schema and local-baseline SHA-256 hashes.
