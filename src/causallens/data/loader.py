"""Value-preserving loader for the selected CFRNet IHDP-100 release.

Observed and oracle fields have intentionally separate public accessors.
No scaling, recoding, reshuffling or outcome normalization occurs here.
"""
from numbers import Integral
from pathlib import Path
import zipfile
import numpy as np

REQUIRED_KEYS = ("x", "t", "yf", "ycf", "mu0", "mu1")
SPLITS = ("train", "test")
SPLIT_SIZES = {"train": 672, "test": 75}
FEATURE_NAMES = tuple(f"x{i:02d}" for i in range(1, 26))


def archive_path(split, data_dir=None):
    if split not in SPLITS:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_dir) if data_dir is not None else Path.cwd() / "data" / "raw"
    return root / f"ihdp_npci_1-100.{split}.npz"


def validate_archive(path):
    """Check ZIP CRC, numeric arrays, full-replication shapes and finite values.

    This verifies release structure, not source authenticity or patient identity.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt archive member: {bad}")
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"Missing keys: {sorted(missing)}")
        x = data["x"]
        if x.ndim != 3 or x.shape[1:] != (25, 100) or x.shape[0] == 0:
            raise ValueError(f"Expected x shape (n,25,100), got {x.shape}")
        n = x.shape[0]
        result = {}
        for key in data.files:
            value = data[key]
            if value.dtype.kind not in "biuf":
                raise ValueError(f"Non-numeric array {key}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite values in {key}")
            if key in REQUIRED_KEYS and key != "x" and value.shape != (n, 100):
                raise ValueError(f"Unaligned shape for {key}: {value.shape}")
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "nonfinite": int((~np.isfinite(value)).sum())}
        if not np.isin(data["t"], [0, 1]).all():
            raise ValueError("Treatment must contain only 0 and 1")
    return result


def _selected(replication, split, data_dir, keys):
    path = archive_path(split, data_dir)
    if isinstance(replication, bool) or not isinstance(replication, Integral):
        raise ValueError("replication must be an integer in [0,99]")
    if not 0 <= replication < 100:
        raise ValueError("replication must be in [0,99]")
    # Validate selected arrays here; full CRC/all-replication checks are explicit
    # acquisition/inspection operations, avoiding repeated full-file scans.
    with np.load(path, allow_pickle=False) as data:
        x = data["x"]
        if x.ndim != 3 or x.shape[1:] != (25, 100) or x.shape[0] == 0:
            raise ValueError("Unexpected x shape")
        n = x.shape[0]
        if n != SPLIT_SIZES[split]:
            raise ValueError(f"Unexpected {split} sample count: {n}")
        out = {}
        for key in keys:
            a = data[key]
            expected = (n,25,100) if key == "x" else (n,100)
            if a.shape != expected or a.dtype.kind not in "biuf" or not np.isfinite(a).all():
                raise ValueError(f"Invalid values or shape in {key}")
            out[key] = a[:, :, replication].copy() if key == "x" else a[:, replication].copy()
        if "t" in out and not np.isin(out["t"], [0,1]).all():
            raise ValueError("Treatment must contain only 0 and 1")
        return out


def load_ihdp(replication=0, split="train", data_dir=None):
    """Return only {'X': (n,25), 'T': (n,), 'Y': (n,)} without dtype changes.

    Default storage is data/raw relative to the current working directory.
    Pass data_dir explicitly when calling outside the repository root.
    """
    data = _selected(replication, split, data_dir, ("x", "t", "yf"))
    return {"X": data["x"], "T": data["t"], "Y": data["yf"]}


def load_ihdp_truth(replication=0, split="test", data_dir=None):
    """Explicit evaluation-only access; never pass this object to model fitting."""
    data = _selected(replication, split, data_dir, ("mu0", "mu1", "ycf"))
    data["tau"] = data["mu1"] - data["mu0"]
    return data
