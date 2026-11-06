"""Atomic, cached downloads from the author's exact IHDP-100 URLs."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests
from .loader import SPLITS, SPLIT_SIZES, archive_path, validate_archive

URLS = {s: f"https://www.fredjo.com/files/ihdp_npci_1-100.{s}.npz" for s in SPLITS}
MAX_BYTES = 200 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def download_ihdp(data_dir="data/raw", expected_hashes=None):
    """Return a manifest; existing valid files are reused and never overwritten.

    Optional expected_hashes maps train/test to previously recorded SHA-256.
    No publisher-authenticated checksums are currently known. Local manifests
    detect later changes but cannot authenticate the first retrieval.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "variant": "CFRNet-lineage IHDP-100", "files": {},
        "checksum_authority": "locally observed; no publisher hash located",
        "redistribution": "unverified; local research only, do not upload archives"}
    expected_hashes = expected_hashes or {}
    if set(expected_hashes) - set(SPLITS):
        raise ValueError("Expected hashes must be keyed by train/test")
    for split in SPLITS:
        path = archive_path(split, root)
        previous = manifest["files"].get(split, {})
        expected = expected_hashes.get(split) or previous.get("sha256")
        if path.exists():
            schema = validate_archive(path)
            if schema["x"]["shape"][0] != SPLIT_SIZES[split]:
                raise ValueError(f"Unexpected {split} sample count")
            digest = sha256(path)
            if expected and digest != expected:
                raise ValueError(f"Hash mismatch for existing {path}; refusing replacement")
            if not previous:
                manifest["files"][split] = {
                    "url": URLS[split], "filename": path.name, "bytes": path.stat().st_size,
                    "sha256": digest, "retrieved_at_utc": None,
                    "provenance_note": "Existing file adopted; original retrieval time unknown"}
            atomic_json(manifest_path, manifest)
            continue
        temp = path.with_name(path.name + ".part")
        try:
            with requests.get(URLS[split], stream=True, timeout=(15,120),
                              headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                if response.url != URLS[split]:
                    raise ValueError(f"Unexpected redirect to {response.url}; review provenance")
                size = 0
                with temp.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError("Download exceeded expected safety size limit")
                        f.write(chunk)
                if not size:
                    raise ValueError("Empty download")
                length = response.headers.get("Content-Length")
                if length is not None and size != int(length):
                    raise ValueError("Incomplete download: Content-Length mismatch")
                schema = validate_archive(temp)
                if schema["x"]["shape"][0] != SPLIT_SIZES[split]:
                    raise ValueError(f"Unexpected {split} sample count")
                digest = sha256(temp)
                if expected and digest != expected:
                    raise ValueError(f"SHA-256 mismatch for {split}")
                entry = {"url": URLS[split], "final_url": response.url,
                         "filename": path.name, "bytes": size, "sha256": digest,
                         "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                         "etag": response.headers.get("ETag"),
                         "last_modified": response.headers.get("Last-Modified"),
                         "schema": schema}
                # Never replace an existing raw file.
                if path.exists():
                    raise FileExistsError(path)
                os.replace(temp, path)
                manifest["files"][split] = entry
                atomic_json(manifest_path, manifest)
        finally:
            temp.unlink(missing_ok=True)
    return manifest
