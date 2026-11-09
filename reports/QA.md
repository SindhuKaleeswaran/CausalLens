# Independent final QA

## Execution evidence

Windows x64, CPython 3.13.12, November 8, 2026. Tests ran against the isolated worktree and locally hash-checked IHDP archives. Commands below assume the corresponding environment is active, the project installed, and the repository root as working directory.

| Check | Command | Observed result |
|---|---|---|
| Baseline | python -m pytest -q at 53c9ca7 | 114 passed, 0 failed, 0 skipped, 53 warnings, 180.38 s. |
| Initial adversarial cases | python -m pytest tests/test_audit_regressions.py -q before fixes | 24 failed, 1 passed. After fixes, the targeted suite with policy and evaluation tests passed 42/42. |
| Data pipeline | python scripts/validate_ihdp.py | 200 split records, 0 failures; hashes unchanged. |
| Original benchmark | python scripts/evaluate_benchmark.py --full | 100/100 replications, 600/600 successes; baseline and audited versions both executed. |
| Independent metrics | python scripts/verify_ihdp100.py | 600 unique records, 12 scientific fields each and 36 distinct bootstrap intervals verified. |
| Extended benchmark | python scripts/evaluate_extended.py --smoke and python scripts/evaluate_extended.py | Smoke 16/16; full 800/800. Original data and outputs stored separately. |
| Independent extended metrics | python scripts/verify_ihdp100.py --extended --results-dir results/extended | 800 unique records, 12 fields each and 48 distinct bootstrap intervals verified. |
| Separate model tests | python -m pytest tests/test_experimental_models.py -q | 2 passed, 0 failed, 0 skipped. Checked clone determinism, heterogeneous signal, honest forest, MMD gradient and ablation. |
| Figures | python scripts/generate_benchmark_figures.py | Plots regenerated, including separate extended error/interval figure. |
| Partial release isolation | python scripts/evaluate_benchmark.py --smoke | Produced runs/ihdp-2; full detail CSV SHA-256 unchanged. |
| Combined environment | python -m coverage run --source=causallens -m pytest -q | Final combined run 146 passed, 0 failed, 0 skipped, 66 warnings in 173.52 s; 84% package statement coverage (1,443 statements, 237 missed). Streamlit and optional models installed together. |
| Dashboard | Streamlit AppTest and Playwright in installed Microsoft Edge | All six tabs, switching model and replication, numeric predictions, high-responder preset, 0% capacity, robustness actions, corrupt file handling, real 600-row CSV download, parsed JSON download, and mobile navigation exercised. |
| Installation | Fresh venv; pip install --upgrade pip; pip install -e ".[eda,dashboard,dev]"; pip check | Clean installation and startup succeeded. Combined environment pip check found no broken requirements. |
| Syntax and style | python -m compileall -q src scripts tests app.py; ruff check app.py src scripts tests --select E9,F63,F7,F82; git diff --check | Passed. Full Ruff lint and formatting checks detected existing compact statements, unused imports and legacy layout. New experiment/verifier/test files passed targeted lint and formatting. |
| Dependency audit | pip-audit --path ENV/Lib/site-packages --format json | Original venv pip 25.3 had five distinct published advisory IDs. Disposable release environment upgraded pip to 26.3; no known vulnerabilities returned among scanned packages. Local causallens and the +cpu Torch build lacked PyPI advisory records for their exact versions. |

## Browser and environment evidence

An unconditional high-responder preset click in the original app displayed unit 0. The fixed real browser showed unit 34, and AppTest checked individual displayed numbers against independently fitted models. The real browser test navigated at 1440×960 and 390×844. It downloaded and parsed the six-model CSV and reliability JSON. The automated selector initially timed out during plot rendering; waiting for the actual widget value passed. The sandboxed Streamlit event loop stalled creating a socket pair; full tests were rerun with loopback available.

## Release decision

Conditionally ready for a local research demo using the verified data and environments. The eight-model report is read-only in the dashboard; the live individual explorer uses the original six. No load testing, full accessibility audit, clinical validation or deployment penetration testing was executed. A Windows CPU benchmark is not a portable latency guarantee. Test datasets in hosted CI may be absent by design. Keep the author-hosted archives out of Git and run the release-data verification locally before submission.
