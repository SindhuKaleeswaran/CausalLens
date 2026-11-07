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
    )

    print("\nBenchmark Summary Results (sorted by PEHE):")
    cols = ["model", "n_successful", "pehe_summary", "ate_error_summary", "policy_risk_mean", "avg_runtime_sec"]
    print(summary_df[cols].to_string(index=False))

    md_content = f"# CausalLens — IHDP Benchmark Evaluation Report\n\n"
    md_content += f"Evaluated across **{len(reps)} replications** (replications {reps[0]} to {reps[-1]}).\n\n"

    sub_df = summary_df[cols]
    header = "| " + " | ".join(cols) + " |\n"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = "".join("| " + " | ".join(str(val) for val in row) + " |\n" for row in sub_df.itertuples(index=False))
    md_content += header + separator + rows
    md_content += "\n*Note: PEHE is root mean squared error between predicted individual treatment effect and ground truth (lower is better).*\n"

    out_file = args.output_dir / "benchmark_summary.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md_content)
    print(f"\nSaved summary markdown report to: {out_file}")
    print(f"Saved detailed CSV to: {args.output_dir / 'benchmark_detailed.csv'}")


if __name__ == "__main__":
    main()
