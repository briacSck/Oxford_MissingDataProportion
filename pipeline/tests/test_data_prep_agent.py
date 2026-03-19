"""
pipeline/tests/test_data_prep_agent.py
----------------------------------------
Tests for data_prep_agent.py  (5 tests)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.data_prep_agent import (
    prepare_baseline,
    _apply_sample_restrictions,
    _verify_columns,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, data: dict | None = None) -> Path:
    if data is None:
        data = {"y": [1.0, 2.0, None, 4.0], "x1": [10.0, 20.0, 30.0, 40.0]}
    df = pd.DataFrame(data)
    p = tmp_path / "test_data.csv"
    df.to_csv(str(p), index=False)
    return p


def _minimal_spec(**overrides) -> dict:
    base = {
        "paper_id":             "Paper_Test",
        "dependent_var":        "y",
        "key_independent_vars": ["x1"],
        "control_vars":         [],
        "fixed_effects":        [],
        "sample_restrictions":  [],
    }
    base.update(overrides)
    return base


# ── Test 1: CSV loaded ─────────────────────────────────────────────────────────

def test_load_csv(tmp_path):
    csv_path = _make_csv(tmp_path, {"y": [1, 2, 3], "x1": [4, 5, 6], "x2": [7, 8, 9]})
    spec = _minimal_spec(control_vars=["x2"])

    with patch("pipeline.data_prep_agent._append_log"):
        df = prepare_baseline(str(csv_path), spec, str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["y", "x1", "x2"]


# ── Test 2: notna restriction drops NaN rows ───────────────────────────────────

def test_sample_restriction_notna(tmp_path):
    csv_path = _make_csv(tmp_path, {"y": [1.0, None, 3.0], "x1": [10.0, 20.0, 30.0]})
    spec = _minimal_spec(sample_restrictions=["y != ."])

    with patch("pipeline.data_prep_agent._append_log"):
        df = prepare_baseline(str(csv_path), spec, str(tmp_path))

    assert len(df) == 2
    assert df["y"].notna().all()


# ── Test 3: unknown restriction flagged, no rows dropped ──────────────────────

def test_sample_restriction_unknown_flagged(tmp_path):
    csv_path = _make_csv(tmp_path, {"y": [1, 2, 3], "x1": [4, 5, 6]})
    weird_restriction = "bysort firm: some_complex_stata_command"
    spec = _minimal_spec(sample_restrictions=[weird_restriction])

    with patch("pipeline.data_prep_agent._append_log") as mock_log:
        df = prepare_baseline(str(csv_path), spec, str(tmp_path))

    # All 3 rows retained — unparseable restriction is skipped
    assert len(df) == 3

    # The log message should contain the flag about the unparseable restriction
    call_args = mock_log.call_args[0][1]  # message positional arg
    assert "could not be translated" in call_args
    assert weird_restriction in call_args or "some_complex_stata_command" in call_args


# ── Test 4: missing spec variable flagged, no raise ───────────────────────────

def test_missing_variable_flagged(tmp_path):
    csv_path = _make_csv(tmp_path, {"y": [1, 2, 3], "x1": [4, 5, 6]})
    # 'missing_var' is in spec but NOT in the CSV
    spec = _minimal_spec(key_independent_vars=["x1", "missing_var"])

    with patch("pipeline.data_prep_agent._append_log") as mock_log:
        # Should NOT raise even though 'missing_var' is absent
        df = prepare_baseline(str(csv_path), spec, str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    # Log message should mention 'missing_var' and "not found in data"
    call_args = mock_log.call_args[0][1]
    assert "missing_var" in call_args
    assert "not found in data" in call_args


# ── Test 5: baseline.parquet written ──────────────────────────────────────────

def test_parquet_written(tmp_path):
    csv_path = _make_csv(tmp_path, {"y": [1, 2, 3], "x1": [4, 5, 6]})
    spec = _minimal_spec()

    with patch("pipeline.data_prep_agent._append_log"):
        prepare_baseline(str(csv_path), spec, str(tmp_path))

    parquet_path = tmp_path / "baseline.parquet"
    assert parquet_path.exists(), "baseline.parquet should be written to output_dir"

    # Verify it is a valid parquet file
    reloaded = pd.read_parquet(str(parquet_path))
    assert isinstance(reloaded, pd.DataFrame)
    assert len(reloaded) == 3
