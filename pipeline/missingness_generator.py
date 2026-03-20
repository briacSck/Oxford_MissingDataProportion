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


# ── Custom exception ───────────────────────────────────────────────────────────

class MissingnessCalibrationError(RuntimeError):
    """Raised when missingness cannot be calibrated to the target proportion."""
    def __init__(self, var: str, target: float, eligible_n: int,
                 realized_count: int, realized_rate: float):
        self.var = var
        self.target = target
        self.eligible_n = eligible_n
        self.realized_count = realized_count
        self.realized_rate = realized_rate
        super().__init__(
            f"Missingness calibration failed for '{var}': "
            f"target={target:.3f}, eligible_n={eligible_n}, "
            f"realized_count={realized_count}, realized_rate={realized_rate:.4f}"
        )


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

    # NaN-safe shift: use nanmin so NaN rows don't corrupt the shift
    finite_min = np.nanmin(aux_vals)
    if np.isnan(finite_min):
        raise ValueError(
            f"[{paper_dir}] aux_var '{aux_var}' has no finite values — "
            "cannot compute MAR weights."
        )
    if finite_min <= 0:
        aux_vals = aux_vals + abs(finite_min) + 1.0

    # NaN rows get weight 0 (zero probability of missingness)
    weights = np.where(np.isnan(aux_vals), 0.0, aux_vals ** MAR_STRENGTH)
    total_w = weights.sum()
    if total_w == 0.0:
        raise ValueError(
            f"[{paper_dir}] aux_var '{aux_var}': total weight is zero after NaN masking."
        )
    probs = weights / total_w

    results: dict[str, dict[str, str]] = {}

    for key_var in key_vars:
        results[key_var] = {}

        # Eligible rows: key_var is currently observed in baseline
        eligible_mask = ~df[key_var].isna()
        n_eligible = int(eligible_mask.sum())
        if n_eligible == 0:
            raise MissingnessCalibrationError(
                var=key_var, target=0.0, eligible_n=0,
                realized_count=0, realized_rate=0.0,
            )

        prev_count = -1         # for runtime monotonicity assertion
        flat_streak = 0         # consecutive identical counts despite increasing targets

        for proportion, label in zip(MISSING_PROPORTIONS, PROPORTION_LABELS):
            # Binary-search for scaling scalar
            s = _find_scalar(probs, proportion)
            scaled = np.clip(s * probs, 0.0, 1.0)

            # Reproducible draw — seed before EVERY draw
            np.random.seed(RANDOM_SEED)
            rand_vals = np.random.rand(n)
            # Apply only to eligible (non-missing) rows
            missing_mask = (rand_vals < scaled) & eligible_mask.values

            actual_count = int(missing_mask.sum())
            actual_rate = actual_count / n_eligible

            # Runtime: zero missing when target > 0
            if proportion > 0 and actual_count == 0:
                raise MissingnessCalibrationError(
                    var=key_var, target=proportion, eligible_n=n_eligible,
                    realized_count=0, realized_rate=0.0,
                )

            # Runtime: strict monotone — count must never decrease
            if prev_count >= 0 and actual_count < prev_count:
                raise MissingnessCalibrationError(
                    var=key_var, target=proportion, eligible_n=n_eligible,
                    realized_count=actual_count, realized_rate=actual_rate,
                )

            # Runtime: flat streak warning (same count for ≥3 consecutive increasing targets)
            if prev_count >= 0 and actual_count == prev_count:
                flat_streak += 1
                if flat_streak >= 3:
                    logger.warning(
                        "[%s] %s: missing count flat at %d for ≥3 consecutive proportions "
                        "(possible calibration degeneration)",
                        paper_dir, key_var, actual_count,
                    )
            else:
                flat_streak = 0

            prev_count = actual_count

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
                "[%s] %s_MAR_%s: target=%.2f actual=%.4f N_eligible=%d N_missing=%d",
                paper_dir, key_var, label, proportion, actual_rate,
                n_eligible, actual_count,
            )

    return results
