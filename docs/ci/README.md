# Enabling CI

Actions is enabled at repository level, but `.github/workflows` is absent.
The current GitHub OAuth token has `repo`, `read:org`, and `gist`, not `workflow`.
The reviewed template now installs EDA/dashboard/test dependencies and tests
Python 3.11 and 3.13 with artificial offline fixtures. Original-archive tests skip
explicitly when data is absent. Local execution evidence is in the audit report.

A repository owner can create `.github/workflows/tests.yml` using the contents of
[tests.yml.example](tests.yml.example), on a reviewed branch via the GitHub web
editor or a credential with workflow scope. Do not commit directly to main.
No workflow activation or hosted passing run is claimed by this audit.
