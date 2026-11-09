"""Presentation helpers only; no estimator or benchmark calculations."""

import numpy as np
import pandas as pd


def format_metric(value, digits=3, suffix=""):
    """Render unavailable diagnostics honestly, including nonfinite results."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def json_safe(value):
    """Use JSON null for missing/nonfinite diagnostics, never fabricated zeros."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def effect_groups(effects):
    """Quantile display groups with tied values kept together, even with <3 bins."""
    values = pd.Series(effects)
    if not np.isfinite(values).all() or values.empty:
        raise ValueError("Effect groups require finite, nonempty predictions")
    if values.nunique() == 1:
        return pd.Categorical(["Same predicted effect"] * len(values))
    codes, edges = pd.qcut(values, q=3, labels=False, retbins=True, duplicates="drop")
    count = len(edges) - 1
    labels = {
        1: ["Combined predicted effects"],
        2: ["Lower predicted effect", "Higher predicted effect"],
        3: ["Lower predicted effect", "Middle predicted effect", "Higher predicted effect"],
    }[count]
    return pd.Categorical.from_codes(codes.astype(int), categories=labels, ordered=True)
