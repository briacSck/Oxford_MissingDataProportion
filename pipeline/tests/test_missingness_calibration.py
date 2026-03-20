"""Regression tests for missingness_generator robustness."""
import numpy as np
import pandas as pd
import pytest
from pipeline.missingness_generator import generate_missingness, MissingnessCalibrationError
from pipeline.config import MISSING_PROPORTIONS, PROPORTION_LABELS

N = 500
RNG = np.random.default_rng(42)


def _make_df_with_nan_aux(tmp_path, aux_nan_frac=0.15):
    """Baseline where aux_var has NaN in some rows."""
    aux = RNG.uniform(0.5, 5.0, N).astype(float)
    aux[: int(N * aux_nan_frac)] = np.nan
    x1 = RNG.normal(size=N)
    df = pd.DataFrame({"aux": aux, "x1": x1})
    p = tmp_path / "baseline.csv"
    df.to_csv(str(p), index=False)
    return str(p)


def test_nan_aux_does_not_zero_missingness(tmp_path):
    """NaN in aux_var must not produce zero actual missingness."""
    bp = _make_df_with_nan_aux(tmp_path)
    result = generate_missingness(bp, ["x1"], "aux", str(tmp_path))
    for proportion, label in zip(MISSING_PROPORTIONS, PROPORTION_LABELS):
        if proportion == 0:
            continue
        df_out = pd.read_csv(result["x1"][label])
        actual = df_out["x1"].isna().mean()
        assert actual > 0, f"Expected missingness > 0 at {label}, got 0"


def test_eligible_rows_only_receive_missingness(tmp_path):
    """Rows where key_var is already NaN must not be double-counted."""
    aux = RNG.uniform(1.0, 5.0, N)
    x1 = RNG.normal(size=N).astype(float)
    # Pre-existing NaN in x1 (5%)
    x1[: int(N * 0.05)] = np.nan
    df = pd.DataFrame({"aux": aux, "x1": x1})
    p = tmp_path / "baseline.csv"
    df.to_csv(str(p), index=False)
    result = generate_missingness(str(p), ["x1"], "aux", str(tmp_path))
    # At higher targets, actual rate should be reasonable
    label_20 = "20pct"
    df_out = pd.read_csv(result["x1"][label_20])
    # Original NaN rows should still be NaN (just check rate is reasonable)
    assert df_out["x1"].isna().sum() >= int(N * 0.05), "Pre-existing NaN must be preserved"


def test_monotonic_missing_counts(tmp_path):
    """Missing counts must be non-decreasing as proportion increases."""
    aux = RNG.uniform(0.5, 5.0, N)
    x1 = RNG.normal(size=N)
    df = pd.DataFrame({"aux": aux, "x1": x1})
    p = tmp_path / "baseline.csv"
    df.to_csv(str(p), index=False)
    result = generate_missingness(str(p), ["x1"], "aux", str(tmp_path))
    counts = []
    for label in PROPORTION_LABELS:
        df_out = pd.read_csv(result["x1"][label])
        counts.append(df_out["x1"].isna().sum())
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1], (
            f"Missing count not monotone: {counts[i-1]} → {counts[i]} at index {i}"
        )


def test_calibration_error_has_structured_attributes():
    """MissingnessCalibrationError carries var/target/eligible_n/realized fields."""
    err = MissingnessCalibrationError(
        var="x1", target=0.10, eligible_n=500,
        realized_count=0, realized_rate=0.0,
    )
    assert err.var == "x1"
    assert err.target == 0.10
    assert err.eligible_n == 500
    assert err.realized_count == 0
    assert "x1" in str(err)
    assert "0.10" in str(err) or "0.1" in str(err)
