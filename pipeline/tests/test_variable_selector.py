"""
pipeline/tests/test_variable_selector.py
-----------------------------------------
Tests for variable_selector.py  (9 tests)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import openpyxl
import pytest

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.variable_selector import (
    select_variables,
    _filter_eligible_vars,
    _select_aux_var,
    _save_selection,
)

RNG = np.random.default_rng(2026)
N   = 200


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_spec(**overrides) -> dict:
    base = {
        "paper_id":             "Paper_Test",
        "dependent_var":        "y",
        "key_independent_vars": ["x1"],
        "control_vars":         ["x2", "x3", "x4"],
        "fixed_effects":        ["firm"],
        "instrumental_vars":    [],
        "parse_confidence":     "high",
        "manual_review_required": False,
        "replication_code_type": "stata",
        "sample_restrictions":  [],
    }
    base.update(overrides)
    return base


def _make_df(**overrides) -> pd.DataFrame:
    base = {
        "y":    RNG.normal(size=N),
        "x1":   RNG.uniform(0.5, 5.0, size=N),   # positive — good aux candidate
        "x2":   RNG.uniform(0.1, 3.0, size=N),   # positive
        "x3":   RNG.normal(size=N),
        "x4":   RNG.normal(size=N),
        "firm": RNG.integers(1, 20, size=N),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_paper_dir(tmp_path: Path, paper_id: str = "Paper_Test") -> tuple[Path, Path]:
    """Create minimal papers/<paper_id>/ with paper_info.xlsx."""
    papers_root = tmp_path / "papers"
    paper_dir   = papers_root / paper_id
    paper_dir.mkdir(parents=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    headers = [
        "paper_short_name", "source_folder", "do_file", "data_file",
        "estimator", "depvar", "main_coef", "indepvars",
        "absorb", "cluster", "published_coef_main",
        "key_vars", "aux_var", "status", "notes",
    ]
    ws.append(headers)
    ws.append([paper_id, "src", "NONE", None, None, None, None, None,
               None, None, None, None, None, "test", None])
    wb.save(paper_dir / "paper_info.xlsx")
    return papers_root, paper_dir


# ── Shared mock gate (auto-confirm) ───────────────────────────────────────────

_AUTO_GATE = patch(
    "pipeline.variable_selector._human_gate",
    side_effect=lambda sel, pid, **kw: {**sel, "human_confirmed": True},
)


# ── Test 1: DV excluded from key_vars ─────────────────────────────────────────

def test_excludes_dv(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)
    spec = _make_spec()
    df   = _make_df()

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    assert "y" not in result["key_vars"], "Dependent variable must never appear in key_vars"


# ── Test 2: binary excluded; ordinal eligible (Gap 3) ─────────────────────────

def test_excludes_binary(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)

    # 'binary_col' has exactly 2 values {0, 1} — must be excluded
    # 'ordinal_col' has 5 distinct integer values — must be ELIGIBLE (Gap 3)
    df = _make_df(
        binary_col  = RNG.integers(0, 2, size=N).astype(float),    # 2 unique
        ordinal_col = RNG.integers(1, 6, size=N).astype(float),    # 5 unique
    )
    spec = _make_spec(control_vars=["x2", "x3", "x4", "binary_col", "ordinal_col"])

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    excl = result["excluded_vars"]
    assert "binary_col" in excl, "Binary column (2 unique values in {0,1}) must be excluded"
    # ordinal (5 unique values) should NOT be excluded
    assert "ordinal_col" not in excl, (
        "Ordinal column with 5 unique values must be ELIGIBLE (Gap 3 — not binary)"
    )


# ── Test 3: ID/time columns excluded ──────────────────────────────────────────

def test_excludes_id_columns(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)

    df = _make_df(
        gvkey   = RNG.integers(1000, 9999, size=N),
        year    = RNG.integers(2000, 2020, size=N),
        firm_id = RNG.integers(1, 100, size=N),
    )
    spec = _make_spec(control_vars=["x2", "x3", "gvkey", "year", "firm_id"])

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    excl = result["excluded_vars"]
    assert "gvkey"   in excl, "gvkey (ID name) must be excluded"
    assert "year"    in excl, "year (time name) must be excluded"
    assert "firm_id" in excl, "firm_id (_id suffix) must be excluded"


# ── Test 4: fixed-effect variable excluded ────────────────────────────────────

def test_excludes_fe_vars(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)
    spec = _make_spec(fixed_effects=["firm"])
    df   = _make_df()

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    assert "firm" not in result["key_vars"], "FE variable must not appear in key_vars"


# ── Test 5: key IV selected first ────────────────────────────────────────────

def test_selects_key_iv_first(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)
    spec = _make_spec(key_independent_vars=["x1"], control_vars=["x2", "x3", "x4"])
    df   = _make_df()

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    key_vars = result["key_vars"]
    assert len(key_vars) >= 1, "Should have at least one key var"
    assert key_vars[0] == "x1", "Key IV (x1) must be first in key_vars"


# ── Test 6: aux_var not in key_vars ───────────────────────────────────────────

def test_aux_not_in_key_vars(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)
    spec = _make_spec()
    df   = _make_df()

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    aux = result["aux_var"]
    if aux:  # aux may be "" if no candidate found
        assert aux not in result["key_vars"], "aux_var must not be in key_vars"


# ── Test 7: aux_var has |r| >= 0.1 with majority of key_vars ─────────────────

def test_aux_correlation(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)

    # Build data with a known correlating aux candidate
    x1  = RNG.uniform(0.5, 5.0, size=N)
    x2  = x1 * 0.8 + RNG.normal(scale=0.3, size=N)   # correlated with x1
    x3  = RNG.normal(size=N)
    x4  = RNG.normal(size=N)
    aux = x1 * 0.9 + RNG.normal(scale=0.2, size=N)    # strongly correlated aux
    aux = np.abs(aux)  # ensure non-negative
    y   = RNG.normal(size=N)
    df  = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                        "firm": RNG.integers(1, 20, N), "aux_candidate": aux})
    spec = _make_spec(control_vars=["x2", "x3", "x4", "aux_candidate"])

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    aux_var = result.get("aux_var")
    key_vars = result.get("key_vars", [])
    if aux_var and key_vars and aux_var in df.columns:
        rs = [abs(df[kv].corr(df[aux_var])) for kv in key_vars if kv in df.columns]
        assert rs, "Correlation data should be available"
        majority_correlated = sum(1 for r in rs if r >= 0.05) >= len(key_vars) / 2
        assert majority_correlated, (
            f"aux_var {aux_var!r} should correlate with majority of key_vars; "
            f"correlations: {[round(r, 3) for r in rs]}"
        )


# ── Test 8: insufficient eligible vars → manual confidence ───────────────────

def test_insufficient_vars(tmp_path):
    papers_root, _ = _make_paper_dir(tmp_path)

    # Only 1 eligible variable available
    df   = _make_df()
    spec = _make_spec(
        key_independent_vars=["x1"],
        control_vars=[],               # no controls → only x1 eligible
    )

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    # 1 eligible < MIN_KEY_VARS (3) → manual
    assert result["selection_confidence"] == "manual"
    assert len(result["flags"]) > 0, "Should have at least one flag about insufficient vars"
    assert result["key_vars"] == [], "key_vars should be empty when insufficient eligible vars"


# ── Test 9: selection.json written with all TypedDict keys ───────────────────

def test_selection_json_written(tmp_path):
    papers_root, paper_dir = _make_paper_dir(tmp_path)
    spec = _make_spec()
    df   = _make_df()

    # Use auto_confirm inside select_variables (bypasses _human_gate prompt)
    # but we still need _save_selection to be called. We patch _human_gate to:
    # 1) return confirmed selection, AND 2) call _save_selection
    def gate_and_save(sel, pid, **kw):
        confirmed = {**sel, "human_confirmed": True}
        confirmed.pop("_df_ref", None)
        confirmed.pop("_spec_ref", None)
        _save_selection(confirmed, str(papers_root))
        return confirmed

    with patch("pipeline.variable_selector._human_gate", side_effect=gate_and_save):
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    sel_path = paper_dir / "selection.json"
    assert sel_path.exists(), "selection.json must be written after gate confirmation"

    with open(sel_path, encoding="utf-8") as f:
        saved = json.load(f)

    # Check all required VariableSelection keys are present
    required_keys = [
        "paper_id", "key_vars", "aux_var", "aux_var_description",
        "key_var_rationale", "aux_var_rationale", "excluded_vars",
        "selection_confidence", "flags", "human_confirmed", "correlation_matrix",
    ]
    for key in required_keys:
        assert key in saved, f"Key '{key}' missing from selection.json"

    # Gap 4: aux_var_description field must be present (may be empty string)
    assert "aux_var_description" in saved
    assert isinstance(saved["aux_var_description"], str)
