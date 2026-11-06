"""Reserved IHDP download entry point; no dataset operation is implemented."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.exit(2, "Not implemented: complete dataset provenance and schema review first.\n")


if __name__ == "__main__":
    main()
