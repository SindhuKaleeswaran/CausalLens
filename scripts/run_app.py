"""Launcher script for CausalLens Streamlit application."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    app_path = root / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching CausalLens Interactive Decision Studio: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
