# Phase 6 frontend audit — before implementation

## Reviewed revision and boundaries

Reviewed on November 8, 2026 (America/Los_Angeles). Remote main is
`55f33a4`; PR #14 is `53c9ca7`; PR #16 is `cf536a1`. The implementation
branch `feat/phase6-dashboard-polish` starts at PR #16, preserving its fixes
and the six existing tabs. Both upstream PRs remain unmerged. Results remain
provisional for submission until the modeling owner signs off. No estimator,
metric, policy, data-loader, or benchmark-calculation changes are in scope.

This is a source audit of all six tabs, dependencies, tests, artifacts and
open PRs. Browser and execution findings will be recorded separately after
testing. Earlier main-only findings (uncached evaluation, forced sidebar,
unclosed figures and missing-file crashes) are already addressed in #16;
they must not be reported as new fixes here.

## Prioritized findings

| Priority | Area / evidence in app.py at cf536a1 | Planned frontend change |
|---|---|---|
| P1 | Header and KPI ribbon lead with technical terms and a colored cross-split difference without the adjacent caveat. The semi-synthetic limitation is buried in Explorer. | Lead with the investigation question, prominent simulated-outcome disclosure, a four-step reading guide and provisional status. Replace the ribbon's cross-split gap with the selected model's test mean predicted effect; retain the explicitly qualified gap in Data. |
| P1 | Evidence Assessment defaults missing diagnostics to 0 or [0,1], uses green badges for unknown/research-only states and exports those defaults as facts. | Render absent/nonfinite values as N/A; JSON null; use neutral diagnostic labels. Separate model uncertainty from identification uncertainty. |
| P1 | Heterogeneity calls qcut with three fixed labels and duplicates='drop'; tied predictions can produce fewer bins and raise. | Adapt labels to actual quantile edges without splitting ties; use descriptive effect groups rather than clinical benefit tiers. |
| P1 | Hypothetical numeric edits and policy threshold have no finite-value guard. A nonfinite input can stop subsequent tabs. | Validate these presentation inputs and keep an explicit warning with a safe preview fallback. No policy arithmetic changes. |
| P2 | Data explanation asserts observational assignment necessarily induces confounding. Policy and profile wording implies clinical welfare/decision readiness. | Use conditional association language, simulated outcome units, estimated outcomes and research-only policy labels. |
| P2 | Model Evaluation has small overlapping scatter labels, a very wide summary, ambiguous Policy Risk label, and unspecified interval captions. | Interactive hover comparisons using existing Plotly; metric definitions; explicitly label regret and mean/median bootstrap intervals. Retain original CSV values and downloads. |
| P2 | Fixed-width header/ribbon and hidden tab scrollbar harm narrow-screen discoverability. Hundreds of CSS lines impose tiny uppercase labels and external webfonts. | Responsive header, KPI layout, legible labels and visible horizontal tab navigation. Remove remote font dependency. Verify desktop and mobile views. |
| P2 | Resource caches are unbounded across 100 replications. Paired bootstraps and artifact validation run on every rerun. Startup lacks progress context; errors expose local filesystem details. | Bound caches, cache paired comparison by input data, add loading progress and actionable data/model error states. Preserve benchmark computation functions. |
| P2 | Existing AppTest exercises primary paths but lacks missing data, absent diagnostics, tied-group and invalid-input checks. | Add regression coverage, run the whole suite and browser-check all tabs with real hash-checked local archives. |

## Preservation and validation plan

Preserve six tabs, six live estimators, extended read-only experiment, held-out
unit presets, hypothetical profile editing, capacity simulator, refutation
actions and CSV/JSON downloads. Use committed validated artifact pairs and
locally verified archives only. Do not regenerate or edit scientific results.

Run existing suite before edits; then focused AppTest and helper regressions,
full suite, dependency/syntax/diff checks, and real browser desktop/mobile
inspection. Save actual screenshots. Optional experimental model tests may
skip when their separate heavyweight dependencies are absent; report this.

Public deployment belongs to Phase 7 and requires the user's phase go-ahead.
Publishing a branch/PR also requires approval under the supplied execution
rules. Prepare local commits and a reviewable diff first.
