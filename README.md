# CausalLens

AI-powered causal investigation and decision intelligence system for identifying misleading correlations and evaluating treatment effects. A collaborative research and engineering project for DataNova 2026.

## Problem and objectives

Observed associations can mislead decisions when treatment assignment and outcomes share common causes. CausalLens aims to investigate those associations, estimate treatment effects under explicit assumptions, and communicate uncertainty and limitations through a reproducible interactive demo.

## Planned methodology

1. Research IHDP benchmark variants, provenance, access terms, and available ground truth.
2. Acquire an approved version locally; inspect schemas, missingness, treatment balance, and outcome definitions.
3. Build reproducible preprocessing with train-only fitting and explicit leakage checks.
4. Develop causal baselines and estimators; evaluate treatment-effect estimates against documented benchmark targets where available.
5. Run robustness experiments and integrate a Phase 1 demo.

Phases 1–2 are implemented: source research, local acquisition, structural inspection, and a value-preserving loader. Training-only EDA (Phase 3) is complete; minimal optional preprocessing (Phase 4) is implemented; full quality assurance (Phase 5) remains pending. Six causal estimators, evaluation and robustness modules, and a Streamlit dashboard are implemented. See [pre-Phase-5 audit](docs/pre_phase5_audit.md) for current verification and limitations.

## Development setup

Requires Python 3.11 or newer. The prepared CI template targets Python 3.11 and 3.13; activation is pending workflow authorization (see [CI setup](docs/ci/README.md)).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -c "import causallens; print(causallens.__version__)"
```

On Windows activate with `.venv\Scripts\activate`. Open the numbered notebooks with `jupyter lab`. Notebooks 01–03 are executable.

## Repository map

- `data/raw`, `data/processed`: ignored local datasets; only directory markers are tracked.
- `notebooks`: executable inspection, training-only EDA, and preprocessing validation (01–03).
- `src/causallens`: data pipeline, six estimators, benchmark metrics, and robustness modules.
- `scripts`: working acquisition and structural-inspection entry points.
- `tests`: offline synthetic-fixture acquisition/loader tests plus optional local-benchmark integration checks.
- `docs`: provenance, methodology, schema, handoff, and reference templates.
- `reports`: historical EDA and 100-replication benchmark evidence; generated figures are ignored.

## Collaboration

`main` is the stable integration branch. SindhuKaleeswaran owns dataset research, EDA, preprocessing, validation, and the data pipeline on `feat/data-pipeline`. The teammate (`leninathikam`, verified GitHub username; invitation sent) owns causal methods, models, estimation, evaluation, and robustness work on `feat/causal-models`.

Use short descriptive commits and pull requests into `main`. Pull the latest `main` before starting work, keep PRs focused, run tests locally, and request the other developer's review. Branches are prepared; no branch protection is imposed. See GitHub issues for the initial work breakdown.

## Reproducibility and data handling

See [dataset provenance](docs/dataset_research.md), [dataset card](docs/dataset_card.md), [preprocessing plan](docs/preprocessing_methodology.md), and [model handoff](docs/model_handoff.md). Record exact source/version, retrieval date, checksums, environment, seeds, split indices, preprocessing configuration, and commands before reporting experiments. Never commit raw benchmarks, secrets, or third-party content without verified terms.

## License

All rights reserved. No open-source license has been granted. See LICENSE. Dataset and third-party licensing must be reviewed separately.

## Reproduce Phases 1–2

Run from the repository root after installing the package:

```bash
python scripts/download_ihdp.py --expected-manifest docs/data_manifest.json
python scripts/inspect_ihdp.py --replication 0
python -m pytest
```

The downloader preserves existing raw files and checks ZIP CRC, schema and SHA-256. The checked-in manifest records this retrieval's hashes, not publisher-authenticated hashes. No alternate release is substituted. Raw files, partial downloads and row-level JSON stay ignored. See [Phase 1–2 handoff](docs/phase1_2_handoff.md) for environment, API and remaining limitations. CI activation is still blocked by OAuth workflow scope; a template is supplied under docs/ci.

Run Phase 3 with `python scripts/eda_ihdp.py`; see reports/eda_summary.md. Phase 4 is implemented; see docs/preprocessing_methodology.md.

Optional preprocessing: `train, test = load_preprocessed_ihdp(0, standardize=False)` from `causallens.data.preprocess`. Change the flag to True to standardize x01–x06 using training-only statistics. Both modes return X/T/Y dictionaries.

## Dashboard and benchmark

`python -m pip install -e '.[dashboard]'` installs the dashboard dependencies.
Run `streamlit run app.py` after placing the original archives in `data/raw`.
For all development tools and offline tests, install `-r requirements.txt`.
Dashboard interaction tests use explicitly artificial fixtures and need no archives.
Real-data checks remain separate: `CAUSALLENS_REAL_DATA_DIR=/path/to/archives python -m pytest tests/test_audit_integration.py`.

The dashboard and benchmark use raw loader features by default; optional scaling is not silently enabled.
Run `python scripts/evaluate_benchmark.py --full --output-dir work/benchmark-new` to preserve historical reports.
Reported test ATE predictions are means of predicted CATE over test covariates.
For IPW/AIPW these differ from training Hájek/AIPW estimates returned by `estimate_ate()` without X.
The AIPW training ATE interval does not cover individual CATE or the test prediction average.
All outcomes are simulated benchmark quantities; no clinical effectiveness is established.
