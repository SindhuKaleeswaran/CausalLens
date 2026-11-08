"""Execute IHDP benchmark evaluation across replications and generate structured reports."""
import argparse
from pathlib import Path
from causallens.evaluation.benchmarks import run_ihdp_benchmark


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=5, help="Number of replications to evaluate (default: 5)")
    parser.add_argument("--smoke", action="store_true", help="Run fast 2-replication smoke test")
    parser.add_argument("--full", action="store_true", help="Run full 100-replication benchmark")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "reports")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    args = parser.parse_args()

    if args.smoke:
        reps = [0, 1]
    elif args.full:
        reps = list(range(100))
    else:
        reps = list(range(args.replications))

    print(f"Running CausalLens benchmark across {len(reps)} IHDP replications...")
    summary_df, detailed_df = run_ihdp_benchmark(
        replications=reps,
        data_dir=args.data_dir,
        verbose=True,
        export_dir=args.output_dir,
        results_dir=args.results_dir,
    )

    print("\nBenchmark Summary Results (sorted by PEHE):")
    cols = [
        "model",
        "n_successful",
        "pehe_summary",
        "pehe_mean_ci_summary",
        "pehe_median_ci_summary",
        "ate_error_summary",
        "ate_error_mean_ci_summary",
        "ate_error_median_ci_summary",
        "policy_risk_mean",
        "avg_runtime_sec",
    ]
    print(summary_df[cols].to_string(index=False))

    md_content = f"# CausalLens — IHDP Benchmark Evaluation Report\n\n"
    md_content += f"Evaluated across **{len(reps)} replications** (replications {reps[0]} to {reps[-1]}).\n\n"

    rename_map = {
        "model": "Model",
        "n_successful": "Fits",
        "pehe_summary": "Root-PEHE Mean ± SD",
        "pehe_mean_ci_summary": "Root-PEHE 95% CI (Mean)",
        "pehe_median_ci_summary": "Root-PEHE Median [95% CI]",
        "ate_error_summary": "ATE Error Mean ± SD",
        "ate_error_mean_ci_summary": "ATE Error 95% CI (Mean)",
        "ate_error_median_ci_summary": "ATE Error Median [95% CI]",
        "policy_risk_mean": "Policy Regret",
        "avg_runtime_sec": "Avg Runtime (s)",
    }
    sub_df = summary_df[cols].rename(columns=rename_map)
    header = "| " + " | ".join(sub_df.columns) + " |\n"
    separator = "| " + " | ".join(["---"] * len(sub_df.columns)) + " |\n"
    rows = "".join("| " + " | ".join(str(val) for val in row) + " |\n" for row in sub_df.itertuples(index=False))
    md_content += header + separator + rows
    md_content += (
        "\n*Note on Statistical Methodology:*\n"
        "- Root-PEHE: Root mean squared error between predicted individual treatment effect and ground truth.\n"
        "- Confidence intervals for the mean and median are computed via 1,000-resample nonparametric percentile bootstrap.\n"
        "- Policy Regret: Non-negative excess loss V(pi*) - V(pi) relative to the oracle decision rule.\n"
    )

    out_file = args.output_dir / "benchmark_summary.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md_content)
    print(f"\nSaved summary markdown report to: {out_file}")
    print(f"Saved detailed CSV to: {args.output_dir / 'benchmark_detailed.csv'}")


if __name__ == "__main__":
    main()
