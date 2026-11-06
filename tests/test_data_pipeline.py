"""Scaffold smoke tests only; dataset pipeline tests await implementation."""
from importlib import import_module


def test_package_imports():
    package = import_module("causallens")
    assert package.__version__ == "0.1.0"


def test_data_namespaces_import():
    for name in ("loader", "preprocess", "validate"):
        import_module(f"causallens.data.{name}")
