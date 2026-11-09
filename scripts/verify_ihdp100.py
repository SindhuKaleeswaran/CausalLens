"""Refit the selected estimators from the source arrays and independently verify the exports."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causallens.evaluation.benchmarks import get_default_models


def verify(raw, results, models_factory=get_default_models):
    detail = pd.read_csv(results / "ihdp100_unified_results.csv").set_index(
        ["replication", "model"]
    )
    summary = pd.read_csv(results / "ihdp100_summary.csv").set_index("model")
    names = list(models_factory())
    expected = pd.MultiIndex.from_product([range(100), names], names=detail.index.names)
    assert detail.index.is_unique and detail.index.sort_values().equals(
        expected.sort_values()
    )
    assert detail.status.eq("SUCCESS").all()
    assert summary.index.is_unique and set(summary.index) == set(names)
    # Read the archive fields directly, bypassing the loader, metric and aggregate functions.
    with np.load(raw / "ihdp_npci_1-100.train.npz", allow_pickle=False) as archive:
        x, treatment, factual = (archive[key] for key in ("x", "t", "yf"))
    with np.load(raw / "ihdp_npci_1-100.test.npz", allow_pickle=False) as archive:
        test_x, mu0, mu1 = (archive[key] for key in ("x", "mu0", "mu1"))
    assert x.shape == (672, 25, 100) and test_x.shape == (75, 25, 100)
    for rep in range(100):
        tau = mu1[:, rep] - mu0[:, rep]
        oracle = np.maximum(mu1[:, rep], mu0[:, rep]).mean()
        for name, model in models_factory().items():
            model.fit(x[:, :, rep], treatment[:, rep], factual[:, rep])
            effect = model.estimate_ite(test_x[:, :, rep])
            assert effect.shape == tau.shape
            mse = np.dot(effect - tau, effect - tau) / len(tau)
            signed = effect.mean() - tau.mean()
            value = np.where(effect > 0, mu1[:, rep], mu0[:, rep]).mean()
            variable = effect.std() > 1e-12 and tau.std() > 1e-12
            row = dict(
                pehe=np.sqrt(mse),
                root_pehe=np.sqrt(mse),
                pehe_sq=mse,
                ate_pred=effect.mean(),
                ate_true=tau.mean(),
                ate_error=abs(signed),
                signed_ate_error=signed,
                in_sample_ate=model.estimate_ate(),
                policy_value=value,
                policy_risk=oracle - value,
                cate_pearson_r=np.corrcoef(effect, tau)[0, 1] if variable else np.nan,
                cate_spearman_r=stats.spearmanr(effect, tau).statistic
                if variable
                else np.nan,
            )
            actual = detail.loc[(rep, name), list(row)].to_numpy(dtype=float)
            np.testing.assert_allclose(
                actual,
                list(row.values()),
                rtol=0,
                atol=5.0001e-5,
                err_msg=f"replication {rep}, {name}, fields {list(row)}",
            )
    for name, group in detail.groupby(level="model"):
        assert (
            summary.loc[name, "n_successful"] == 100
            and summary.loc[name, "n_failed"] == 0
        )
        for col in ("root_pehe", "ate_error", "policy_risk"):
            values = group.sort_index()[col].to_numpy()
            indices = np.random.RandomState(42).randint(100, size=(1000, 100))
            for label, fn in (("mean", np.mean), ("median", np.median)):
                center = fn(values)
                lo, hi = np.quantile(fn(values[indices], axis=1), [0.025, 0.975])
                fields = [
                    f"{col}_{label}",
                    f"{col}_{label}_ci95_low",
                    f"{col}_{label}_ci95_high",
                ]
                np.testing.assert_allclose(
                    summary.loc[name, fields].to_numpy(float),
                    [center, lo, hi],
                    rtol=0,
                    atol=5.001e-7,
                )
            np.testing.assert_allclose(
                summary.loc[name, col + "_std"], values.std(ddof=1)
            )
    print(
        f"Verified {len(detail)} distinct fits, 12 scientific fields per fit, and {6 * len(names)} bootstrap intervals."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Refit the eight-model experiment on one CPU thread",
    )
    args = parser.parse_args()
    if args.extended:
        import torch
        from threadpoolctl import threadpool_limits
        from causallens.models.experimental import get_experimental_models

        def eight_models():
            return {**get_default_models(), **get_experimental_models()}

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            verify(args.data_dir, args.results_dir, eight_models)
    else:
        verify(args.data_dir, args.results_dir)
