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

This repository contains setup scaffolding only. No dataset has been downloaded, no IHDP analysis has been run, and no research results or model implementations are claimed.

## Development setup

Requires Python 3.11 or newer. The prepared CI template targets Python 3.11; activation is pending workflow authorization (see [CI setup](docs/ci/README.md)).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -c "import causallens; print(causallens.__version__)"
```

On Windows activate with `.venv\Scripts\activate`. Open the numbered notebooks with `jupyter lab`. They contain planning text only.

## Repository map

- `data/raw`, `data/processed`: ignored local datasets; only directory markers are tracked.
- `notebooks`: ordered inspection, EDA, and preprocessing-validation templates.
- `src/causallens`: importable package with reserved data, model, and evaluation modules.
- `scripts`: acquisition and inspection entry points, currently explicit placeholders.
- `tests`: package/scaffold smoke tests, not data-pipeline validation.
- `docs`: provenance, methodology, schema, handoff, and reference templates.
- `reports`: pending EDA and quality reports; generated figures are ignored.

## Collaboration

`main` is the stable integration branch. SindhuKaleeswaran owns dataset research, EDA, preprocessing, validation, and the data pipeline on `feat/data-pipeline`. The teammate owns causal methods, models, estimation, evaluation, and robustness work on `feat/causal-models`; account and invitation are pending verification.

Use short descriptive commits and pull requests into `main`. Pull the latest `main` before starting work, keep PRs focused, run tests locally, and request the other developer's review. Branches are prepared; no branch protection is imposed. See GitHub issues for the initial work breakdown.

## Reproducibility and data handling

See [dataset provenance](docs/dataset_research.md), [dataset card](docs/dataset_card.md), [preprocessing plan](docs/preprocessing_methodology.md), and [model handoff](docs/model_handoff.md). Record exact source/version, retrieval date, checksums, environment, seeds, split indices, preprocessing configuration, and commands before reporting experiments. Never commit raw benchmarks, secrets, or third-party content without verified terms.

## License

All rights reserved. No open-source license has been granted. See LICENSE. Dataset and third-party licensing must be reviewed separately.
