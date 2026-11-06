"""Execute IHDP benchmark evaluation across multiple replications and generate report."""
import argparse
from pathlib import Path
from causallens.evaluation.benchmarks import run_ihdp_benchmark


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=5, help="Number of replications to evaluate (default: 5)")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output", type=Path, default=root / "reports" / "benchmark_summary.md")
    args = parser.parse_args()

    reps = list(range(args.replications))
    print(f"Running CausalLens benchmark across {len(reps)} IHDP replications...")
    df = run_ihdp_benchmark(replications=reps, data_dir=args.data_dir, verbose=True)

    print("\nBenchmark Summary Results (sorted by PEHE):")
    print(df[["model", "pehe_summary", "ate_error_summary", "policy_risk_mean"]].to_string(index=False))

    md_content = f"# CausalLens — IHDP Benchmark Evaluation Report\n\n"
    md_content += f"Evaluated across **{len(reps)} replications** (replications {reps[0]} to {reps[-1]}).\n\n"
    cols = ["model", "pehe_summary", "ate_error_summary", "policy_risk_mean"]
    sub_df = df[cols]
    header = "| " + " | ".join(cols) + " |\n"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = "".join("| " + " | ".join(str(val) for val in row) + " |\n" for row in sub_df.itertuples(index=False))
    md_content += header + separator + rows
    md_content += "\n\n*Note: PEHE is root mean squared error between predicted individual treatment effect and ground truth (lower is better).*\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md_content)
    print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
