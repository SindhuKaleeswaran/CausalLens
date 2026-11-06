# Enabling CI

The prepared workflow is stored as `tests.yml.example` because the current GitHub OAuth authorization cannot create workflow files (missing workflow scope). No Actions workflow is active yet.

A repository owner can use GitHub's web editor to create `.github/workflows/tests.yml` with the contents of this template and commit it to `main`, or authorize a development environment with workflow permission and run:

```bash
mkdir -p .github/workflows
cp docs/ci/tests.yml.example .github/workflows/tests.yml
git add .github/workflows/tests.yml
git commit -m "Enable Python test workflow"
git push origin main
```

The workflow runs package smoke tests on Python 3.11, without downloading benchmark data. GitHub Actions has not yet executed these checks.
