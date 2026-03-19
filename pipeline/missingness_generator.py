"""
pipeline/missingness_generator.py
----------------------------------
Missingness Generator: injects power-law MAR missingness into each key
variable at each of the 7 target proportions and saves one CSV per
(variable × proportion) combination.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    MAR_STRENGTH,
    MISSING_PROPORTIONS,
    PROPORTION_LABELS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_baseline(path: str) -> pd.DataFrame:
    """Load a baseline file — handles .parquet, .csv, .xlsx."""
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    elif p.endswith(".csv"):
        return pd.read_csv(p)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file format: {p}")


# ── Scalar binary search ──────────────────────────────────────────────────────

def _find_scalar(probs: np.ndarray, target: float,
                 tol: float = 0.001, max_steps: int = 50) -> float:
    """Binary search for scalar s such that mean(clip(s*probs, 0, 1)) ≈ target."""
    lo, hi = 0.0, float(len(probs))
    for _ in range(max_steps):
        mid = (lo + hi) / 2.0
        mean_p = np.clip(mid * probs, 0.0, 1.0).mean()
        if abs(mean_p - target) < tol:
            return mid
        if mean_p < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── Public entry ──────────────────────────────────────────────────────────────

def generate_missingness(
    baseline_path: str,
    key_vars: list[str],
    aux_var: str,
    paper_dir: str,
) -> dict[str, dict[str, str]]:
    """Generate power-law MAR datasets for each (key_var, proportion) pair.

    Parameters
    ----------
    baseline_path:
        Path to the baseline dataset (.parquet / .csv / .xlsx).
    key_vars:
        Variables that will receive injected missingness.
    aux_var:
        Auxiliary variable driving the MAR mechanism (must not be in key_vars).
    paper_dir:
        Root directory of the paper; output goes to ``{paper_dir}/missing/``.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{varname: {label: output_path}}`` for all generated files.

    Raises
    ------
    ValueError
        If aux_var is not in data, aux_var is in key_vars, or any key_var is
        not in data.
    """
    df = _load_baseline(baseline_path)

    # ── Validation ────────────────────────────────────────────────────────────
    if aux_var not in df.columns:
        raise ValueError(f"aux_var '{aux_var}' not found in data columns.")
    if aux_var in key_vars:
        raise ValueError(f"aux_var '{aux_var}' must not be in key_vars.")
    missing_kvs = [kv for kv in key_vars if kv not in df.columns]
    if missing_kvs:
        raise ValueError(f"key_vars not found in data: {missing_kvs}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    out_dir = Path(paper_dir) / "missing"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    aux_vals = df[aux_var].values.astype(float)

    # Shift so all values are strictly positive
    if np.any(aux_vals <= 0):
        aux_vals = aux_vals + abs(aux_vals.min()) + 1.0

    # Power-law weights and probabilities
    weights = aux_vals ** MAR_STRENGTH
    probs = weights / weights.sum()

    results: dict[str, dict[str, str]] = {}

    for key_var in key_vars:
        results[key_var] = {}
        for proportion, label in zip(MISSING_PROPORTIONS, PROPORTION_LABELS):
            # Binary-search for scaling scalar
            s = _find_scalar(probs, proportion)
            scaled = np.clip(s * probs, 0.0, 1.0)

            # Reproducible draw — seed before EVERY draw
            np.random.seed(RANDOM_SEED)
            missing_mask = np.random.rand(n) < scaled

            actual_rate = missing_mask.mean()
            if abs(actual_rate - proportion) >= 0.005:
                logger.warning(
                    "[%s] %s MAR_%s: actual_rate=%.4f, target=%.4f — diff=%.4f",
                    paper_dir, key_var, label, actual_rate, proportion,
                    abs(actual_rate - proportion),
                )

            df_out = df.copy()
            df_out.loc[missing_mask, key_var] = np.nan

            out_path = out_dir / f"{key_var}_MAR_{label}.csv"
            df_out.to_csv(str(out_path), index=False)

            results[key_var][label] = str(out_path)
            logger.info(
                "[%s] %s_MAR_%s: target=%.2f actual=%.4f N_missing=%d",
                paper_dir, key_var, label, proportion, actual_rate,
                missing_mask.sum(),
            )

    return results
