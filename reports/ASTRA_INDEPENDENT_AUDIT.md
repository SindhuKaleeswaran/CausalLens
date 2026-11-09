# Independent repository audit

**Verdict — conditionally ready for a research demonstration.** The original six estimators and separate eight-estimator experiment reproduced across 100 IHDP replications. Clinical deployment and population-level validation are outside the evidence. See [scientific verification](ASTRA_SCIENTIFIC_VERIFICATION.md) and [executed QA](ASTRA_FINAL_QA.md).

## Revision and coordination

GitHub PR #14 targeted main from 53c9ca7bc8b289e695ed1f4f348548b60481fa03. Main was 55f33a4. The local premium redesign c0e652c was the direct parent of the integration head. The older UI work at e91938c was already in main. Git showed one clean checkout and no other registered worktree at intake. The live remote heads and open PR list showed no separate pending frontend branch. Unregistered clones cannot be inspected through Git worktree metadata. The integration commit changes app.py, benchmark aggregation, policy logic, tests and reports on top of the frontend. No merge or cherry-pick was necessary. Audit work uses a separate worktree and branch.

## Findings register

Locations refer to the immutable input revision 53c9ca7. Each row describes the failure, consequence and correction.

| Severity | Location | Reproduction and correction |
|---|---|---|
| High | README.md:45, reports/SCIENTIFIC_VALIDATION_V2.md:36, reports/DEMO_SCRIPT.md:81 | T-Learner mean 0.697 contradicted 600 reproduced rows. Displayed unit 34 had factual 2.23 and error 0.61. Replaced unsupported numbers and clinical assertions with measured examples. |
| High | src/causallens/evaluation/policy.py:103,120,144 and metrics.py:139 | Three units at 50% yielded two slots. A threshold above all true benefits made the oracle worse than a feasible model, then clipping concealed it. Floor capacity, stable ties, maximize raw oracle value, return undefined regret for infeasible Treat All. Exhaustive small-policy tests. |
| High | src/causallens/evaluation/robustness.py:307,346 | Setting X=T returned a meaningful-looking OLS sensitivity result for an unidentified coefficient. Small-sample significance RV was .3543 instead of .3144. Check identifiability, use rank-based df and the sensemakr critical value. |
| High | app.py:1221 | Selecting high responder in rep 0 retained unit 0 in the persistent selectbox. Bind presets to their actual unit and test displayed factual, counterfactual, effect and oracle across replications and models. Real Edge verified unit 34. |
| High | scripts/evaluate_benchmark.py:31, app.py:1047 | Smoke results overwrote full-run files; incomplete pairs and empty CSVs broke dashboard sections. Route partial CLI runs to separate directories; check key grid, statuses, counts and means, guard artifact reads. Report failed fits and exit unsuccessful. |
| Medium | src/causallens/evaluation/policy.py:243,287, app.py:1830 | Failed fits contaminated matched samples, duplicates inflated counts, and p<.05 displayed as p<.001. Reject duplicates, pair successes and show separate raw/Holm tests and matched counts. Cross-checked against statsmodels and SciPy. |
| Medium | src/causallens/evaluation/policy.py:46 | A decision of 2 returned an impossible outcome, multidimensional inputs broadcast and rounding erased a .00004 regret. Validate finite aligned vectors, binary decisions and threshold, retain precision and identify empty groups. |
| Medium | src/causallens/evaluation/benchmarks.py:110,117,270 | Constant effects received numeric zero correlations on 200 fits, and nonexistent per-replication seeds appeared in CSV. Record undefined correlations and counts, actual estimator seeds, valid JSON nulls; validate successful primary metrics and grid. |
| Medium | src/causallens/models/base.py:68, naive.py:73 | Named feature permutations silently changed predictions, valid refits with new width were blocked and naive ATE bypassed validation. Check prediction schema, permit refit. Tested across six estimators. Unnamed arrays require caller-managed order. |
| Medium | src/causallens/models/ipw.py:90,148 and doubly_robust.py:157 | Reversed clipping bounds yielded invalid weights. Internal weighted arithmetic TypeError could trigger an unweighted retry. Validate clip bounds and distinguish declared weight support from internal errors. |
| Medium | app.py:1704,1830 | Fixed disagreement and significance prose contradicted live results (T vs DR mean absolute difference .5485 on rep 9). Removed hardcoded claims and reported evidence per draw. |
| Medium | .streamlit/config.toml:15 | Explicitly disabled both CORS and XSRF safeguards. Restored Streamlit defaults and verified browser interaction and downloads. No demonstrated credential disclosure. |
| Medium | src/causallens/data/loader.py:31, acquisition.py:51,83 | Compressed byte limit did not bound CRC expansion; expected digest checked after parsing. Check total uncompressed bytes before CRC and known digests before inspecting downloads. Added compressed-member regression. |
| Low | app.py:664,855,912,1427 | Auto-expansion fought sidebar collapse, pyplot figures stayed open, six estimators refit on widget reruns, zero capacity inaccessible. Removed forced expansion, close figures, cache evaluation per replication, expose zero. |
| Low | .github/workflows/tests.yml:6, requirements-lock-py313.txt:1, README.md:4 | PRs into the integration branch lacked CI; old Linux snapshot omitted dashboard dependencies and README claimed 116 passing tests without the acquisition step. Added integration target, optional-model CI, tested Windows snapshots and corrected installation. |
| Informational | src/causallens/evaluation/benchmarks.py:29 | No genuine forest or CFRNet at the input revision. New opt-in experiment refits all eight with separate artifacts; the original default registry stays at six. |

## Scope and limitations

Reviewed acquisition, archive schema, observed/oracle access, EDA, preprocessing and validation, all estimator classes, metrics, robustness, policy, scripts, Streamlit workflows, notebook roles, reports, tests, dependency declarations, CI and license. No test or oracle outcomes entered the benchmark fit path. Safe NPZ loading disables pickle. Paths to archive files use a fixed train/test allowlist; HTML uses local templates with controlled model names and numerical values. Source searches for common credential formats, dynamic execution and unsafe deserializers found no matches in scanned executable sources. This was not an exhaustive secret-history or adversarial deployment penetration test.

Performance timings are machine-specific. Archive hashes are locally observed. Split identity cannot be established beyond array and fingerprint checks. No independent clinical cohort, external validity trial, production monitoring, assistive-technology certification or capacity load test was available. Legacy style debt and incomplete advisory coverage are recorded in QA.
