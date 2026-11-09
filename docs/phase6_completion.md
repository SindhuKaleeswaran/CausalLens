# Phase 6 — frontend improvement and verification

Completed locally November 9, 2026 (America/Los_Angeles). **Ready for review; not published or deployed.** Phase 7 has not started.

## Completed work

- Audited current main and both open integration PRs before editing. The before-change audit is `docs/phase6_frontend_audit.md`, committed separately as `57ef139`.
- Built on `cf536a1` (PR #16), which itself builds on PR #14. Neither dependency was merged or modified. Remote main remains `55f33a4d310a9610e421648b771c5f3297f02b5c` at the closing check.
- Preserved all six tabs, six live estimators, extended read-only benchmark, individual presets, hypothetical profiles, policy simulator, robustness actions and downloads.
- Added an investigation landing section, four-step guide, glossary, assumptions, explicit semi-synthetic outcome disclosure and provisional approval status.
- Replaced the prominent cross-split gap KPI with the selected estimator's test mean predicted effect. The cross-split difference remains in Data, explicitly described as not an estimate of confounding bias.
- Added interactive treated/control outcome distributions, model-error comparisons, and a metric selector for the committed 100-replication distribution (Root-PEHE, absolute ATE error, policy regret). Plot boxes are described as quartiles; summary mean/median bootstrap intervals retain their explicit statistical targets.
- Fixed missing diagnostic values: N/A in the UI and null in strict JSON. Removed green verdicts that could imply identification or clinical readiness. Unavailable OLS sensitivity no longer breaks the evidence page.
- Kept tied predicted effects together and handled fewer than three quantile groups. Added finite-input fallback warnings and separate data/model initialization error messages.
- Bounded data/model caches and cached validated benchmark artifacts and paired comparisons. File stat signatures invalidate replaced artifacts. No benchmark function or mathematical definition changed.
- Fixed dark-on-dark sidebar caption text found during browser inspection, made the sidebar automatic on narrow screens, exposed horizontal tab scrolling, improved headings and responsive cards, and removed the remote font import.

## Files changed

- `app.py`: presentation, input/error states, bounded caching, charts and evidence export.
- `src/causallens/dashboard.py`: presentation-only numeric/JSON formatting and tied-effect display grouping.
- `tests/test_dashboard_interactions.py`: additional failure, input, caching and grouping regressions.
- `tests/test_dashboard_helpers.py`: missing-value and tied-group edge cases.
- `docs/phase6_frontend_audit.md`: prioritized before-change audit.
- `docs/phase6_completion.md`: this report.

No differences from the base in `src/causallens/models`, `src/causallens/evaluation`, committed `results`, or benchmark summary artifacts. No raw archives are tracked. Local ignored symlinks point to existing user archives; their SHA-256 values match `docs/data_manifest.json`.

## Executed checks

Commands ran from the isolated repository, with Python in `../phase6-venv`.

| Check | Command | Exact result |
|---|---|---|
| Fresh environment | `python3 -m venv ../phase6-venv`; `../phase6-venv/bin/python -m pip install -e '.[dashboard,dev]'` | Installed successfully. An initial pytest launch before installation finished failed collection; discarded and restarted after installation. |
| Dependency compatibility | `../phase6-venv/bin/python -m pip check` | No broken requirements found. |
| Baseline before app edits | `OPENBLAS_NUM_THREADS=1 ../phase6-venv/bin/python -m pytest -q` | **144 passed, 2 skipped, 66 warnings in 93.30 s**. |
| Existing dashboard interactions after first edits | `OPENBLAS_NUM_THREADS=1 ../phase6-venv/bin/python -m pytest tests/test_dashboard_interactions.py -q` | **8 passed, 63 warnings in 52.03 s**. |
| New targeted checks, first attempt | `OPENBLAS_NUM_THREADS=1 ../phase6-venv/bin/python -m pytest tests/test_dashboard_helpers.py tests/test_dashboard_interactions.py -q` | **22 passed, 1 failed, 91 warnings in 74.89 s**. The test expected infinity to survive Streamlit transport. Revised it to inject at the widget return boundary, where the app's guard is relevant. |
| Revised edge checks | `OPENBLAS_NUM_THREADS=1 ../phase6-venv/bin/python -m pytest tests/test_dashboard_interactions.py -k 'nonfinite or initialization or absent_overlap' -q` | **3 passed, 13 deselected, 9 warnings in 9.66 s**. |
| Final full suite | `OPENBLAS_NUM_THREADS=1 ../phase6-venv/bin/python -m pytest -q` | **161 passed, 2 skipped, 94 warnings in 86.13 s**. Includes 16 AppTest cases and real-archive integration tests. |
| Syntax | `../phase6-venv/bin/python -m compileall -q app.py src tests` | Passed. |
| Whitespace | `git diff --check` | Passed. |
| Data integrity | `sha256sum data/raw/*.npz` | Both match the manifest. Test: `a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c`; train: `750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240`. |
| Tracking | `git check-ignore data/raw/*.npz`; `git ls-files data/raw` | Both archives ignored; only `.gitkeep` tracked. |

The two skips are optional EconML and Torch model tests. Those models are not fitted in this dashboard environment. The 94 warnings comprise Seaborn/Matplotlib deprecations and existing SciPy precision warnings on near-identical test inputs. Streamlit also logs deprecated `use_container_width` calls inherited throughout the app. They do not fail this tested environment; migration belongs with dependency compatibility review.

Versions: Linux, Python 3.13.5; Streamlit 1.67.0; NumPy 2.5.4; pandas 2.3.3; SciPy 1.18.1; scikit-learn 1.9.1; Matplotlib 3.11.2; Seaborn 0.13.2; Plotly 7.1.0; pytest 9.2.0. The delivered environment inventory records the complete installed set, not a portable deployment lockfile.

## Real-browser checks and screenshots

Used the in-app browser with local verified IHDP archives, at 1440×1000 and 390×844. Inspected all six tabs. Verified benchmark metric selection, held-out high-responder preset (unit 34 in replication 0), actual noise-injection result, zero-capacity policy (CONTROL assignments), responsive evidence stacking and horizontal mobile tab navigation. Desktop sidebar captions are visible after the contrast correction. Temporary viewport override was reset.

Downloaded the full six-model benchmark CSV through its browser control, parsed **600 rows / 600 unique replication-model pairs**, and verified byte equality with the committed CSV. Export controls and strict JSON null conversion are covered by automated tests. A separate end-to-end browser download/parse of the reliability JSON was not performed.

The original development session on port 8501 eventually lost its browser/server connection and left widgets disabled. Browser navigation/download calls also returned with unusually long delays. A fresh server on 8502 with file watching disabled restored interaction; the real-data noise action and zero-capacity check passed there. This is not evidence of production uptime or public deployment reliability. Both local server health endpoints responded when checked.

Actual screenshots are delivered separately (not committed to the source branch):

- `01-overview.jpg`: investigation landing, scientific disclosure and configuration.
- `02-model-evaluation.jpg`: committed full benchmark and distribution.
- `03-counterfactual-explorer.jpg`: held-out unit preset, estimates and oracle.
- `04-policy-zero-capacity.jpg`: policy boundary view.
- `05-robustness.jpg`: real-data noise diagnostic and disagreement view.
- `06-evidence.jpg`: assumptions, diagnostics and paired comparisons.
- `07-mobile-evidence.jpg`: narrow-screen navigation and evidence layout.

Screenshots use provisional results and are demonstration materials, not finalized submission claims.

## Remaining limitations and approval gates

- Modeling-owner approval and review/merge of upstream PRs #14 and #16 remain outstanding. The new work is stacked on #16 to avoid replacing their changes.
- The original six-model and separate eight-model artifact runs are preserved, not rerun or independently re-certified in this frontend phase. Live fits use the recorded local environment and may differ from stored runs across dependency versions.
- The noise helper does not return a completed-simulation count; that field now displays N/A instead of inventing a count. The button still states the requested 10 trials.
- No full accessibility certification, concurrent-user load test, or public-host reliability test was performed. Matplotlib charts remain static; Plotly views provide interactive inspection where added.
- Dataset redistribution permissions are unresolved. Existing local archives are neither committed nor included in the deliverable bundle.
- No branch push, PR creation, merge, deployment, or submission was performed. The user's execution rule 13 requires approval before publishing to shared resources.
- Next: obtain approval to publish the local frontend branch and open a draft PR against `audit/independent-verification`; obtain the user's explicit go for Phase 7. Phase 7 must review reproducible deployment dependencies, permissible dataset loading, hosting authorization and public smoke tests before any deployment.
