"""Download and validate the exact author-hosted CFRNet IHDP-100 archives."""
import argparse
import json
from pathlib import Path
from causallens.data.acquisition import download_ihdp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--expected-manifest", type=Path,
                        help="Compare against recorded hashes (local baseline, not publisher signature)")
    args = parser.parse_args()
    hashes = None
    if args.expected_manifest:
        m = json.loads(args.expected_manifest.read_text())
        hashes = {s: entry["sha256"] for s, entry in m["files"].items()}
    print(json.dumps(download_ihdp(args.data_dir, hashes), indent=2))


if __name__ == "__main__":
    main()
