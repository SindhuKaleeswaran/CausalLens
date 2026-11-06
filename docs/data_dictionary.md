# Data dictionary

Canonical feature identifiers are **x01–x25**, ordered as stored in `x`. The archives do not promise names or clinical units. This table records the upstream NPCI ordered list [R4 in references](references.md), **not a certified NPZ-to-clinical schema**. Do not label plots with clinical meanings without stronger exporter evidence. Mnemonics alone do not verify definitions or which binary value denotes which category.

| ID | Zero-based column | Upstream mnemonic | Upstream type expectation | Evidence limit |
|---|---|---|---|---|
| x01 | 0 | `bw` | multi-valued numeric candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x02 | 1 | `b.head` | multi-valued numeric candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x03 | 2 | `preterm` | multi-valued numeric candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x04 | 3 | `birth.o` | multi-valued numeric candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x05 | 4 | `nnhealth` | multi-valued numeric candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x06 | 5 | `momage` | multi-valued numeric candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x07 | 6 | `sex` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x08 | 7 | `twin` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x09 | 8 | `b.marr` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x10 | 9 | `mom.lths` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x11 | 10 | `mom.hs` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x12 | 11 | `mom.scoll` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x13 | 12 | `cig` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x14 | 13 | `first` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x15 | 14 | `booze` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x16 | 15 | `drugs` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x17 | 16 | `work.dur` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x18 | 17 | `prenatal` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x19 | 18 | `ark` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x20 | 19 | `ein` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x21 | 20 | `har` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x22 | 21 | `mia` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x23 | 22 | `pen` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x24 | 23 | `tex` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |
| x25 | 24 | `was` | indicator candidate | Unverified NPZ mapping; units/clinical coding not supplied |

The six leading upstream mnemonics suggest birth weight, head circumference, prematurity, birth order, neonatal health index and maternal age, respectively; the exact clinical operational definitions, source units and mapping to these NPZ columns remain unverified. Other mnemonics refer to child/maternal characteristics and sites. We deliberately retain their original identifiers rather than expanding ambiguous abbreviations. Empirical two-level values and counts are reported separately by inspection; do not blindly assume all indicators are {0,1}.

## Archive schema

| Key | Meaning | Role | Expected selected-replication shape |
|---|---|---|---|
| x | Covariates, unchanged numeric order | Observed features X | (n,25) |
| t | Assignment, 0 control / 1 treated | Observed treatment T | (n,) |
| yf | Noisy response for assigned arm | Observed outcome Y | (n,) |
| ycf | Response for unassigned arm | Evaluation-only | (n,) |
| mu0 | Expected untreated response | Evaluation-only | (n,) |
| mu1 | Expected treated response | Evaluation-only | (n,) |

Array keys and semantics follow the author data page [R7]. Actual dtypes, additional keys, replication axes, binary-value checks and measured ranges are established by Phase 2, not inferred from filenames. Tau is derived as mu1−mu0; no transformation is applied to yf.

## Verified archive facts (Phase 2)

The six required arrays are float64. x has shape (672,25,100) for train and (75,25,100) for test; other required fields have (n,100). Additional `ate`, `yadd`, `ymul` arrays are int64 scalars (4, 0, 1 respectively), not per-person targets. Do not use scalar `ate` as ground truth: compute the chosen split's mean(mu1−mu0) when evaluation begins.

For replication 0, x01–x06 have more than two observed levels; x07–x25 are two-level in both splits. **x14 uses {1,2}; the other 18 use {0,1}.** These are empirical encodings; x14 is not silently recoded. Zero missing/nonfinite values were found across the complete arrays. The leading six are already centered/scaled-looking numeric values; their original clinical units cannot be recovered from these archives alone. No new standardization or imputation has been applied.
