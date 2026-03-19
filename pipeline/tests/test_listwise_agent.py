"""Tests for pipeline/listwise_agent.py"""

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PROPORTION_LABELS
from pipeline.listwise_agent import apply_listwise

# ── Helper ─────────────────────────────────────────────────────────────────────

def _create_mar_files(paper_dir, varname: str, labels=None, n: int = 100,
                      missing_frac: float = 0.2, other_col_nan: bool = False):
    """Write synthetic MAR CSV files to paper_dir/missing/."""
    if labels is None:
        labels = PROPORTION_LABELS
    missing_dir = paper_dir / "missing"
    missing_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for label in labels:
        df = pd.DataFrame({
            varname: rng.normal(size=n).tolist(),
            "other": rng.normal(size=n).tolist(),
        })
        # Inject NaN into target column
        idx = rng.choice(n, size=int(n * missing_frac), replace=False)
        df.loc[idx, varname] = np.nan
        if other_col_nan:
            # Inject NaN into other column on different rows
            other_idx = rng.choice(n, size=5, replace=False)
            df.loc[other_idx, "other"] = np.nan
        df.to_csv(str(missing_dir / f"{varname}_MAR_{label}.csv"), index=False)

    return missing_dir


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_only_target_column_dropped(tmp_path):
    """Rows with NaN only in the non-target column are KEPT; rows with NaN in
    the target column are dropped."""
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    label = "10pct"

    df = pd.DataFrame({
        "x1": [1.0, np.nan, 3.0, 4.0],
        "x2": [np.nan, 2.0, np.nan, 4.0],  # NaN in rows 0 and 2
    })
    # Row 1: x1=NaN → should be dropped
    # Row 0: x2=NaN but x1 ok → should be kept
    # Row 2: x2=NaN but x1 ok → should be kept
    df.to_csv(str(missing_dir / f"x1_MAR_{label}.csv"), index=False)

    result = apply_listwise(str(tmp_path))

    out_path, n_before, n_after = result["x1"][label]
    df_out = pd.read_csv(out_path)

    assert n_before == 4
    assert n_after == 3
    assert len(df_out) == 3
    # Row with NaN in x1 is gone
    assert df_out["x1"].isna().sum() == 0
    # Rows with NaN only in x2 should still be present (x2 NaN rows survive)
    assert df_out["x2"].isna().sum() == 2


def test_n_counts_correct(tmp_path):
    """Returned n_before and n_after match actual CSV lengths."""
    n_total = 200
    missing_frac = 0.25
    _create_mar_files(tmp_path, "x1", labels=["05pct"], n=n_total,
                      missing_frac=missing_frac)

    result = apply_listwise(str(tmp_path))
    _, n_before, n_after = result["x1"]["05pct"]

    assert n_before == n_total
    assert n_after < n_before
    # Verify n_after matches the actual output file
    out_path = result["x1"]["05pct"][0]
    df_out = pd.read_csv(out_path)
    assert len(df_out) == n_after


def test_output_file_naming(tmp_path):
    """Output files are named {varname}_MAR_{label}_LD.csv."""
    _create_mar_files(tmp_path, "x1", labels=["01pct", "10pct"])
    apply_listwise(str(tmp_path))

    listwise_dir = tmp_path / "listwise"
    assert (listwise_dir / "x1_MAR_01pct_LD.csv").exists()
    assert (listwise_dir / "x1_MAR_10pct_LD.csv").exists()


def test_all_files_processed(tmp_path):
    """14 input files (2 vars × 7 proportions) → 14 output files."""
    _create_mar_files(tmp_path, "x1")
    _create_mar_files(tmp_path, "x2")

    result = apply_listwise(str(tmp_path))

    listwise_dir = tmp_path / "listwise"
    output_files = list(listwise_dir.glob("*_MAR_*_LD.csv"))
    assert len(output_files) == 14, f"Expected 14 LD files, got {len(output_files)}"

    # Both vars present in result
    assert "x1" in result and "x2" in result
    assert len(result["x1"]) == 7
    assert len(result["x2"]) == 7
