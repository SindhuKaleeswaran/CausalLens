# CausalLens public synthetic demo

This configuration supersedes the earlier opt-in IHDP deployment proposal.
**Never configure IHDP secrets, raw archives, or acquisition for this app.**

| Setting | Value |
|---|---|
| Provider | Streamlit Community Cloud |
| Repository | SindhuKaleeswaran/CausalLens |
| Branch | feat/phase7-deployment-ready |
| Main file | deploy/streamlit_app.py |
| Python | 3.13 |
| Dependencies | deploy/requirements.txt (48 exact runtime pins) |
| App visibility | Public |
| Suggested URL slug | causallens-synthetic-demo (availability unverified) |
| Secrets | None |
| Data | Independently generated synthetic examples only |
| IHDP downloads / cloud processing / row display | Disabled by construction |
| Static file serving | Disabled |

The entrypoint-directory requirements take precedence over root requirements on
[Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
Use the advanced settings to select Python 3.13. No API tokens are required.
Do not paste development secrets into the deployment form.

## Local reproduction

From the repository root, in a clean Python 3.13 virtual environment:

```sh
python -m pip install -r deploy/requirements.txt
python -m pip check
python -m streamlit run deploy/streamlit_app.py
```

No dataset acquisition step is needed. `app.py` remains the separate authorized
local research dashboard. The public app has six tabs, six estimators, ten seeds,
synthetic profile inspection, capacity allocation, prediction stress, and labeled
CSV/JSON exports. No empirical IHDP result appears in this UI.

See [data-permission review](../docs/phase7_data_permissions.md) for the unresolved
rights boundary and [readiness report](../docs/phase7_deployment_readiness.md) for
validation and deployment status.

## Deployment and validation

1. Sign in to [Community Cloud](https://share.streamlit.io/) as the repository owner.
2. Create an app using the settings above. Review any requested GitHub scopes and
   service terms; restrict repository access where the service permits it.
3. Deploy, record the actual URL and commit, and inspect build/runtime logs.
4. Open the public URL without viewer authentication; verify all six tabs, model
   and seed switches, profile selection, zero/full capacity, and both downloads.
5. Reboot from the owner console and repeat a cold-start smoke check.

Do not call a proposed URL live until deployment is confirmed. If runtime fails,
stop the app or fix the synthetic-only branch; never fall back to IHDP downloads.
Community Cloud follows branch changes, so future pushes to this deployment
branch must retain the same public-data boundary.
