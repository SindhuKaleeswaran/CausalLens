"""Public, synthetic-only entrypoint. No raw-data paths or acquisition switches."""
import os
from pathlib import Path
import runpy
import sys
import tempfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'causallens-mpl'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'deploy'))
# Deliberately does not execute the research app or read hosting secrets/data paths.
runpy.run_path(str(ROOT / 'deploy' / 'public_demo.py'), run_name='__main__')
