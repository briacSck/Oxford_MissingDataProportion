"""
pipeline/tests/test_regression_robustness.py
---------------------------------------------
Regression-robustness tests derived from real-paper failures.

Covers six failure classes not previously tested:
  1. Path-normalization edge cases
  2. Unicode leakage in batch / auto_confirm mode
  3. aux_var strictly separate from key_vars and non-empty
  4. Selector fails clearly (too few vars, no aux due to high missing)
  5. FE verifier robustness (structural edge cases + duplicate DataFrame columns)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import openpyxl
import pytest

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.variable_selector import select_variables
from pipeline.baseline_verifier import verify_baseline


RNG = np.random.default_rng(2026)
N   = 200


# ── Shared helpers ─────────────────────────────────────────────────────────────

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
        "x1":   RNG.uniform(0.5, 5.0, size=N),
        "x2":   RNG.uniform(0.1, 3.0, size=N),
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


def _make_panel_df(n_firms: int = 30, n_periods: int = 10) -> pd.DataFrame:
    """Synthetic panel DataFrame for FE tests."""
    n      = n_firms * n_periods
    firm   = np.repeat(np.arange(n_firms), n_periods)
    period = np.tile(np.arange(n_periods), n_firms)
    x1     = RNG.normal(size=n)
    x2     = RNG.normal(size=n)
    fe     = RNG.normal(size=n_firms)[firm]
    noise  = RNG.normal(scale=0.3, size=n)
    y      = 0.5 * x1 + 0.2 * x2 + fe + noise
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "period": period})


def _make_fe_spec(**overrides) -> dict:
    base = {
        "paper_id":             "Paper_Test",
        "estimator":            "FE",
        "dependent_var":        "y",
        "key_independent_vars": ["x1"],
        "control_vars":         ["x2"],
        "fixed_effects":        ["firm"],
        "cluster_var":          None,
        "instrumental_vars":    [],
        "sample_restrictions":  [],
    }
    base.update(overrides)
    return base


_AUTO_GATE = patch(
    "pipeline.variable_selector._human_gate",
    side_effect=lambda sel, pid, **kw: {**sel, "human_confirmed": True},
)


# ── Group 1 — Path normalization ──────────────────────────────────────────────

@pytest.mark.parametrize("transform_id,path_fn", [
    ("normal",         lambda p: str(p)),
    ("trailing_slash", lambda p: str(p) + "/"),
    ("double_slash",   lambda p: str(p) + "//"),
    ("backslash",      lambda p: str(p).replace("/", "\\")),
])
def test_path_normalization_select_variables(tmp_path, transform_id, path_fn):
    """select_variables must accept path strings with edge-case separators."""
    papers_root, _ = _make_paper_dir(tmp_path)
    spec = _make_spec()
    df   = _make_df()

    transformed = path_fn(papers_root)

    with _AUTO_GATE:
        result = select_variables("Paper_Test", transformed, spec=spec, data=df)

    assert "key_vars" in result, (
        f"[{transform_id}] select_variables must return key_vars for path: {transformed!r}"
    )


# ── Group 2 — No Unicode in auto_confirm mode ─────────────────────────────────

UNICODE_BOX_CHARS = "╔║╠╚╗╣╝⚠"


def test_no_unicode_stdout_in_auto_confirm(tmp_path, capsys):
    """With auto_confirm=True, _print_box must not emit Unicode box-drawing chars."""
    papers_root, _ = _make_paper_dir(tmp_path)
    spec = _make_spec()
    df   = _make_df()

    # Drive through the real _human_gate with auto_confirm=True (no mock needed —
    # we are testing the actual code path that skips _print_box)
    select_variables("Paper_Test", str(papers_root), spec=spec, data=df,
                     auto_confirm=True)

    captured = capsys.readouterr()
    for ch in UNICODE_BOX_CHARS:
        assert ch not in captured.out, (
            f"Unicode char {ch!r} found in stdout during auto_confirm=True run. "
            "_print_box must only be called when auto_confirm=False."
        )


# ── Group 3 — aux_var strictly separate and non-empty ─────────────────────────

def test_aux_strictly_separate_and_nonempty(tmp_path):
    """aux_var must be non-empty and must not overlap with key_vars."""
    papers_root, _ = _make_paper_dir(tmp_path)
    # 5 controls built from a shared latent factor so they are correlated;
    # _pick_aux_first requires majority-correlation >= 0.10 to select aux_var.
    latent = RNG.normal(size=N)
    x2 = np.abs(0.7 * latent + 0.4 * RNG.normal(size=N)) + 0.1   # positive
    x3 = 0.7 * latent + 0.4 * RNG.normal(size=N)
    x4 = 0.6 * latent + 0.4 * RNG.normal(size=N)
    x5 = np.abs(0.7 * latent + 0.4 * RNG.normal(size=N)) + 0.1   # positive
    spec = _make_spec(
        key_independent_vars=["x1"],
        control_vars=["x2", "x3", "x4", "x5"],
    )
    df = _make_df(x2=x2, x3=x3, x4=x4, x5=x5)

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    aux_var  = result.get("aux_var", "")
    key_vars = result.get("key_vars", [])

    assert aux_var != "", "aux_var should be non-empty when enough eligible vars exist"
    assert aux_var not in key_vars, "aux_var must not appear in key_vars"


# ── Group 4 — Selector fails clearly ─────────────────────────────────────────

@pytest.mark.parametrize("scenario,expected_confidence,expected_flag_substr", [
    ("too_few_eligible_vars", "manual",  "need at least"),
    ("no_aux_high_missing",   "partial", "no suitable aux var found"),
])
def test_selector_fails_clearly(tmp_path, scenario, expected_confidence, expected_flag_substr):
    """Selector must set confidence correctly and emit a descriptive flag."""
    papers_root, _ = _make_paper_dir(tmp_path)

    if scenario == "too_few_eligible_vars":
        # Only 1 candidate var total (x1) → after aux reservation, key_pool is empty
        spec = _make_spec(
            key_independent_vars=["x1"],
            control_vars=[],
            fixed_effects=[],
        )
        df = _make_df()

    elif scenario == "no_aux_high_missing":
        # ctrl has 30% NaN → passes rule 8 (<50%) but fails _pick_aux_first 5% filter
        arr = np.empty(N)
        arr[: int(N * 0.3)] = np.nan
        arr[int(N * 0.3) :] = RNG.uniform(0, 1, N - int(N * 0.3))
        spec = _make_spec(
            key_independent_vars=["x1", "x2", "x3"],
            control_vars=["ctrl"],
            fixed_effects=[],
        )
        df = _make_df(ctrl=arr)

    else:
        pytest.fail(f"Unknown scenario: {scenario}")

    with _AUTO_GATE:
        result = select_variables("Paper_Test", str(papers_root), spec=spec, data=df)

    assert result["selection_confidence"] == expected_confidence, (
        f"[{scenario}] expected confidence={expected_confidence!r}, "
        f"got {result['selection_confidence']!r}"
    )

    flags_text = " ".join(result.get("flags", []))
    assert expected_flag_substr in flags_text, (
        f"[{scenario}] expected substring {expected_flag_substr!r} in flags, "
        f"got: {flags_text!r}"
    )


# ── Group 5a — FE verifier structural edge cases ──────────────────────────────

@pytest.mark.parametrize("scenario", [
    "duplicate_fe_in_spec",
    "cluster_var_is_fe_col",
    "multi_column_fe",
    "fe_col_missing_from_data",
])
def test_fe_verifier_structural_edge_cases(scenario):
    """verify_baseline must not crash on structural FE edge cases."""
    df = _make_panel_df()

    if scenario == "duplicate_fe_in_spec":
        # Duplicate FE declaration — _build_matrices deduplicates via set
        spec = _make_fe_spec(fixed_effects=["firm", "firm"])

    elif scenario == "cluster_var_is_fe_col":
        # Real Mapping2024 pattern: cluster_var same column as FE
        spec = _make_fe_spec(fixed_effects=["firm"], cluster_var="firm")

    elif scenario == "multi_column_fe":
        # Two-way FE: firm + period
        spec = _make_fe_spec(fixed_effects=["firm", "period"])

    elif scenario == "fe_col_missing_from_data":
        # Non-existent FE column → filtered by _build_matrices line 112
        spec = _make_fe_spec(fixed_effects=["nonexistent_fe"])

    else:
        pytest.fail(f"Unknown scenario: {scenario}")

    report = verify_baseline(df, spec, published_coef={})

    assert isinstance(report, dict), (
        f"[{scenario}] verify_baseline must return a dict, not raise"
    )
    assert "coef_estimate" in report, (
        f"[{scenario}] report must contain 'coef_estimate' key"
    )


# ── Group 5b — Duplicate DataFrame columns ────────────────────────────────────

def test_fe_verifier_duplicate_df_columns():
    """verify_baseline must return a dict (not raise) when DataFrame has duplicate column names."""
    df_base = _make_panel_df()
    # Introduce a second "x1" column via concat
    df_dup = pd.concat([df_base, df_base[["x1"]]], axis=1)
    # df_dup.columns now contains "x1" twice

    spec = _make_fe_spec(fixed_effects=["firm"])
    report = verify_baseline(df_dup, spec, published_coef={})

    assert isinstance(report, dict), (
        "verify_baseline must return a dict, not raise, when df has duplicate columns"
    )
    assert "coef_estimate" in report, (
        "report must contain 'coef_estimate' key even with duplicate df columns"
    )
