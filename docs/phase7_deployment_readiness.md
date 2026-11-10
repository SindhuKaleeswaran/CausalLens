# Phase 7 — synthetic public demo readiness

Updated November 9, 2026. Supersedes the earlier raw-data runtime proposal at
local commit `63dc1cd`. Owner explicitly approved branch publication and public
synthetic deployment, but prohibited cloud processing/hosting/display of raw
IHDP data. Dataset and derived-output permissions remain unestablished.

## Result

A dedicated six-tab public app uses independent synthetic generation and the
existing six estimators. It includes model evaluation, fictional-unit inspection,
capacity allocation, estimator disagreement, prediction-noise stress, and labeled
CSV/JSON exports. It is explicitly educational, not an IHDP result or clinical
recommendation. There is no IHDP mode switch, acquisition helper, data path, or
secret in this public entrypoint. Aggregate benchmark results are withheld pending
clearance. Research `app.py`, estimators, scientific evaluation functions, and
committed benchmark results are unchanged relative to Phase 6 (`d56650a`).

## Validation

- Full suite: `OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/causallens-public-mpl ../phase6-venv/bin/python -m pytest -q`:
  **168 passed, 2 skipped, 94 warnings in 130.12 seconds**. The two skips are
  optional EconML/Torch dependencies. Warnings are inherited scientific-library
  precision and visualization deprecations.
- Public tests exercise all ten seeds, six estimator outputs, all six tabs,
  model selection, profile bounds, zero capacity, maximum noise, and forbidden
  archive/data-loader calls even with the legacy download flag enabled.
- Dedicated clean source/runtime smoke: no NPZ files, no results directory, no
  editable package. All six tabs and two download controls render without errors.
  Cold start **2.97 s**, warm interaction **0.19 s**, peak RSS **241.0 MiB**.
  These are local AppTest measurements, not cloud latency or load-test guarantees.
- Runtime `pip check`: no broken requirements. Same pinned 48-package environment
  as the earlier clean installation; no new dependency was added.
- Tracked-file security scan: no archive/model binary, secret filename, common
  credential-pattern match or executed notebook output. All three notebook output
  counts are zero. This targeted review cannot prove absence of every secret type.
- `git diff --check`: passed. No raw archives are added or uploaded.
- Local browser server could not bind a socket under the current sandbox
  (`PermissionError`); therefore no new local screenshot or browser download is
  claimed. Public browser smoke/restart checks must follow actual cloud deployment.

## Configuration

Repository `SindhuKaleeswaran/CausalLens`; branch `feat/phase7-deployment-ready`;
entrypoint `deploy/streamlit_app.py`; Python **3.13**; dependency file
`deploy/requirements.txt`; visibility **public**; secrets **none**; static serving
**disabled**. Suggested slug `causallens-synthetic-demo`, subject to availability.
See [deploy README](../deploy/README.md) for exact reproduction and rollback.

Phase 6 PR #17 remains separate and unmerged. Publication uses its commit as the
parent so this deployment work does not replace upstream contributors' branches.
Record the actual published SHA, URL and cloud verification in the delivered
status record; this readiness report alone is not proof of deployment. Phase 8
and final submission are outside the currently authorized scope.

## Observed cloud runtime

The first deployment successfully started at
https://causallens-synthetic-demo.streamlit.app/ on Python **3.13.18**.
Community Cloud selected `deploy/requirements.txt` as intended. Its platform
installer replaced PyArrow 26.0.1 with **24.0.0**, reporting a known segfault in
26.0.1. The runtime pin is now explicitly 24.0.0 to match the running host.
The 168-test local result above used the original local PyArrow 26.0.1; public
UI validation covers the hosted 24.0.0 runtime. The platform also installs its
own logging packages and upgrades pip; these are outside the application lock.
