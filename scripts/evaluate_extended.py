"""Run the separate eight-model experiment without changing original benchmark outputs."""

from pathlib import Path
import argparse

from causallens.evaluation.benchmarks import get_default_models, run_ihdp_benchmark
from causallens.models.experimental import get_experimental_models


def main():
    import torch
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    models = {**get_default_models(), **get_experimental_models()}
    out = root / ("runs/extended-smoke" if args.smoke else "results/extended")
    with threadpool_limits(limits=1):
        summary, detail = run_ihdp_benchmark(
            range(2 if args.smoke else 100),
            models=models,
            data_dir=root / "data/raw",
            results_dir=out,
        )
    print(
        summary[
            ["model", "n_successful", "n_failed", "root_pehe_mean", "avg_runtime_sec"]
        ].to_string(index=False)
    )
    if detail.status.ne("SUCCESS").any():
        raise SystemExit(
            "Failed fits recorded. Inspect extended results before making comparisons."
        )


if __name__ == "__main__":
    main()
