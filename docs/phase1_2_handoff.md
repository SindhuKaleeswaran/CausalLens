# Phase 1–2 handoff and reproducibility

## Current boundary

Research and initial acquisition/inspection only. Phase 3 requires explicit user go-ahead. No EDA, preprocessing fitting, validation splits, causal estimators or frontend were implemented. `preprocess.py` and `validate.py` remain placeholders for their authorized future phases; structural checks presently live in loader.py.

## Environment and storage

Verified with CPython 3.13.5, Linux x86_64, an isolated virtual environment and editable package installation. `requirements.txt` provides bounded portable dependencies; `requirements-lock-py313.txt` records the exact tested Python 3.13 environment. It may not resolve on Python 3.11. Use Python 3.13.5 for matching this reference environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock-py313.txt
python -m pip install -e . --no-deps
python scripts/download_ihdp.py --expected-manifest docs/data_manifest.json
python scripts/inspect_ihdp.py --replication 0
python -m pytest
python -m pip check
```

The initial environment and local data are outside Git. Default data storage is `data/raw/` relative to the working directory; pass an absolute `data_dir` when working elsewhere. Raw NPZ bytes are not rewritten. `manifest.json` beside the raw files records retrieval times and headers; `docs/data_manifest.json` preserves the initial provenance/hash baseline. Ignore local JSON and raw archives; do not upload them.

## API

```python
from causallens.data.loader import load_ihdp, load_ihdp_truth
train = load_ihdp(replication=0, split="train", data_dir="data/raw")
test = load_ihdp(replication=0, split="test", data_dir="data/raw")
# train/test contain only X, T, Y; shapes (n,25), (n,), (n,).
# Separate access for evaluation; never pass this object into fitting/tuning:
truth = load_ihdp_truth(replication=0, split="test", data_dir="data/raw")
# mu0, mu1, ycf, tau each have shape (n,); tau = mu1 - mu0.
```

Float64 source values and order are preserved; T remains float64 with values 0/1. Returned arrays are copies. Invalid split names and noninteger, boolean, negative or out-of-range replication IDs raise ValueError. IDs are zero-based 0–99. Clinical feature mapping is provisional; canonical feature names are x01–x25. No recoding of x14 ({1,2}) occurs. Full CRC/schema checks run during acquisition and inspection; selected loading validates values/shapes without downloading anything.

## Validation and limits

Offline tests use explicitly synthetic arrays, not third-party benchmarks. Optional integration tests run when the two local archives are present; otherwise they skip honestly. Full-array structural checks cover all 100 realizations during acquisition, but comprehensive per-replication scientific QA belongs to Phase 5. The split check compares exact X+T row fingerprints for replication 0, not patient identifiers. No index matching between realizations is valid. Mean potential outcomes and noisy realized responses are distinct.

Source terms, publisher hashes, exact exporter revision, and clinical column coding remain unverified. The data page makes the files available for benchmark research but does not establish a separate redistribution license. Retain downloads locally; clarify terms before public hosting or redistribution. Test data was inspected only for integrity and initial requested summaries, not preprocessing selection or model tuning.

## Next steps after approval

Phase 3: training-first descriptive EDA, group balance and overlap diagnostics with limitations. Later phases: train-only optional transformations, comprehensive validation and a final model integration contract. Coordinate shape/dtype changes with leninathikam via PR review; no changes to the model branch are made here.

## Recorded validation result

On 2026-11-06, all 31 Phase 2 pytest cases passed in the reference environment, including the local benchmark integration check. All six code cells of notebook 01 executed successfully. A second downloader run reused both archives and matched the checked-in hash baseline. `pip check` reported no broken requirements. No model training, Phase 3 EDA or preprocessing validation is implied by these results.
