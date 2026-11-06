# Primary sources and evidence ledger

Reviewed 2026-11-06. Source code revisions are recorded for reproducibility; no third-party code or data is redistributed.

| ID | Source | Evidence used |
|---|---|---|
| R1 | [IHDP Research Group, JAMA 1990, doi:10.1001/jama.1990.03440220059030](https://pubmed.ncbi.nlm.nih.gov/2188023/) | Original eight-site trial, eligibility and purpose |
| R2 | [Ramey et al., Pediatrics 1992, doi:10.1542/peds.89.3.454](https://pubmed.ncbi.nlm.nih.gov/1371341/) | Intervention components and original arm sizes |
| R3 | [Hill, 2011, Bayesian Nonparametric Modeling for Causal Inference, JCGS 20(1):217–240](https://doi.org/10.1198/jcgs.2010.08162) | Benchmark origin; publisher full text returned 403 in this environment. Selection and implementation details corroborated with R4, rather than claiming full-text verification |
| R4 | [NPCI data generator](https://github.com/vdorie/npci/blob/ba08244f35575831d81990c505e70f4fc2e69d80/examples/ihdp_sim/data.R) and [transformations](https://github.com/vdorie/npci/blob/ba08244f35575831d81990c505e70f4fc2e69d80/R/transform.R) | Selection rule, covariate order, upstream normalization, response-surface naming; revision ba08244f35575831d81990c505e70f4fc2e69d80 |
| R5 | [Shalit, Johansson & Sontag, ICML 2017, §5.1](https://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf) | 747 units, 139 treated, 25 features; NPCI setting A, noiseless targets, 1000 runs and 63/27/10 protocol |
| R6 | [CFRNet README](https://github.com/clinicalml/cfrnet/blob/e5080335b1d0b66ff30de7129ce4c096e84816a6/README.md) and [loader](https://github.com/clinicalml/cfrnet/blob/e5080335b1d0b66ff30de7129ce4c096e84816a6/cfr/loader.py) | Exact 100-run example links versus full 1000-run experiments; archive field handling; revision e5080335b1d0b66ff30de7129ce4c096e84816a6 |
| R7 | [Fredrik Johansson's Software & Data page](https://www.fredjo.com/) | Author-hosted IHDP-100 and IHDP-1000 links; definitions of six named NPZ arrays |
| R8 | [CFRNet software license](https://github.com/clinicalml/cfrnet/blob/e5080335b1d0b66ff30de7129ce4c096e84816a6/LICENSE) | MIT software license; not treated as a separate dataset redistribution grant |

## Selected download URLs

- Train: https://www.fredjo.com/files/ihdp_npci_1-100.train.npz
- Test: https://www.fredjo.com/files/ihdp_npci_1-100.test.npz

These HTTPS URLs match the author-hosted filenames linked by R6/R7. No alternate variant will be substituted. No published SHA-256 or explicit standalone archive redistribution terms were found in the reviewed README/data page. Record locally observed hashes as a reproducibility baseline, not as publisher-authenticated signatures. Download only to ignored local storage; obtain terms clarification before redistribution or public hosting.
