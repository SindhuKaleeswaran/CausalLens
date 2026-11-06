"""Inspect IHDP archive structure and a selected replication without modifying data."""
import argparse
import json
from pathlib import Path
from causallens.data.inspection import inspect_ihdp, render_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--replication", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=Path("data/processed/inspection.json"))
    p.add_argument("--output-report", type=Path, default=Path("docs/data_inspection_report.md"))
    args = p.parse_args()
    result = inspect_ihdp(args.data_dir, args.replication)
    for path, content in ((args.output_json,json.dumps(result,indent=2)+"\n"),
                          (args.output_report,render_report(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for split,s in result["splits"].items():
        print(f"{split}: n={s['n']}, p={s['n_features']}, treated={s['treated']}, control={s['control']}")
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
