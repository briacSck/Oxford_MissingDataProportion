"""Tests for variable_selector with mixed-dtype baseline DataFrames."""
import numpy as np
import pandas as pd
import pytest
from pipeline.variable_selector import _filter_eligible_vars, _find_baseline_candidates

RNG = np.random.default_rng(13)
N = 300


def _make_mixed_df():
    """DataFrame with both numeric and StringDtype columns (Meyer2024-like)."""
    df = pd.DataFrame({
        "y":    RNG.normal(size=N),
        "x1":   RNG.normal(size=N),
        "x2":   RNG.uniform(0.1, 5.0, size=N),
        # StringDtype column — NOT caught by dtype == object
        "Name": pd.array(["firm_" + str(i % 50) for i in range(N)], dtype="string"),
        "Date": pd.array(["2020-0" + str(i % 9 + 1) for i in range(N)], dtype="string"),
    })
    return df


def _make_spec_absent():
    """Spec whose candidates are entirely absent from the baseline."""
    return {
        "paper_id": "P_Test",
        "dependent_var": "y",
        "key_independent_vars": ["absent_main"],
        "control_vars": ["absent_ctrl"],
        "fixed_effects": [],
        "instrumental_vars": [],
    }


def test_filter_eligible_excludes_string_dtype():
    """Rule 7 must exclude StringDtype columns, not just object dtype."""
    df = _make_mixed_df()
    spec = {
        "paper_id": "P_Test", "dependent_var": "y",
        "key_independent_vars": ["Name"],  # StringDtype column
        "control_vars": ["x1"],
        "fixed_effects": [], "instrumental_vars": [],
    }
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert "Name" not in eligible, "StringDtype column must be excluded"
    assert "Name" in excluded


def test_repair_pass_skips_string_dtype_no_crash():
    """_find_baseline_candidates must not crash on StringDtype columns."""
    df = _make_mixed_df()
    spec = _make_spec_absent()
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert len(eligible) == 0
    # This must not raise TypeError
    repair, source = _find_baseline_candidates(df, spec, eligible, excluded)
    assert "Name" not in repair, "StringDtype column must be excluded from repair"
    assert "Date" not in repair, "StringDtype column must be excluded from repair"
    # Numeric columns should be found
    assert "x1" in repair or "x2" in repair


def test_repair_pass_finds_numeric_columns_in_mixed_df():
    """After filtering non-numeric types, repair pass finds numeric baseline cols."""
    df = _make_mixed_df()
    spec = _make_spec_absent()
    eligible, excluded = _filter_eligible_vars(spec, df)
    repair, source = _find_baseline_candidates(df, spec, eligible, excluded)
    assert len(repair) >= 2, "At least x1 and x2 must be found"
    for col in repair:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"
