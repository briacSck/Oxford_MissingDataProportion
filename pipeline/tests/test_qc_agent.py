"""Tests for pipeline/qc_agent.py"""

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PROPORTION_LABELS
from pipeline.qc_agent import run_qc

# ── Helper ────────────────────────────────────────────────────────────────────

_RESULT_COLS = [
    "Key Variable",
    "Missing Proportion",
    "Post-LD N",
    "β̂",
    "SE",
    "t-value",
    "p-value",
    "Significance",
    "R²",
    "Consistent with baseline?",
]


def _make_results_df(key_vars=None, n_start=1000, monotone=True):
    """Build a synthetic perfect regression_results DataFrame."""
    if key_vars is None:
        key_vars = ["x1"]

    K = len(key_vars)
    rows = []

    # Baseline row
    rows.append({
        "Key Variable":             key_vars[0],
        "Missing Proportion":       "baseline",
        "Post-LD N":                n_start,
        "β̂":                        2.5,
        "SE":                       0.1,
        "t-value":                  25.0,
        "p-value":                  0.000,
        "Significance":             "***",
        "R²":                       0.85,
        "Consistent with baseline?": "—",
    })

    # LD rows
    rng = np.random.default_rng(7)
    for kv in key_vars:
        for i, label in enumerate(PROPORTION_LABELS):
            if monotone:
                n_row = n_start - (i + 1) * 50
            else:
                # Non-monotone: N increases between proportions
                n_row = n_start + (i + 1) * 10
            rows.append({
                "Key Variable":             kv,
                "Missing Proportion":       label,
                "Post-LD N":                n_row,
                "β̂":                        2.5 + rng.normal() * 0.1,
                "SE":                       0.12,
                "t-value":                  20.0,
                "p-value":                  0.001,
                "Significance":             "***",
                "R²":                       0.83,
                "Consistent with baseline?": "Yes",
            })

    return pd.DataFrame(rows, columns=_RESULT_COLS)


def _write_paper_dir(tmp_path, key_vars=None, monotone=True, drop_rows=0,
                     n_key_vars_for_files=None):
    """Set up a minimal paper_dir with regression_results.xlsx and CSV files."""
    if key_vars is None:
        key_vars = ["x1"]
    if n_key_vars_for_files is None:
        n_key_vars_for_files = len(key_vars)

    paper_dir = tmp_path / "paper"
    paper_dir.mkdir(exist_ok=True)
    missing_dir = paper_dir / "missing"
    listwise_dir = paper_dir / "listwise"
    missing_dir.mkdir(exist_ok=True)
    listwise_dir.mkdir(exist_ok=True)

    K = n_key_vars_for_files
    # Create dummy CSV files
    for kv in key_vars[:K]:
        for label in PROPORTION_LABELS:
            (missing_dir / f"{kv}_MAR_{label}.csv").write_text("x\n1\n")
            (listwise_dir / f"{kv}_MAR_{label}_LD.csv").write_text("x\n1\n")

    # Write results xlsx
    df = _make_results_df(key_vars=key_vars, monotone=monotone)
    if drop_rows > 0:
        df = df.iloc[:-drop_rows].copy()

    df.to_excel(str(paper_dir / "regression_results.xlsx"), sheet_name="Results", index=False)

    return str(paper_dir)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_all_checks_pass(tmp_path):
    """Perfect synthetic results → True and report says 'QC PASSED'."""
    paper_dir = _write_paper_dir(tmp_path, key_vars=["x1"])
    passed = run_qc(paper_dir)
    assert passed is True

    report_text = (tmp_path / "paper" / "qc_report.txt").read_text()
    assert "QC PASSED" in report_text


def test_row_count_fail(tmp_path):
    """Removing rows triggers Check 1 failure → returns False."""
    paper_dir = _write_paper_dir(tmp_path, key_vars=["x1"], drop_rows=3)
    passed = run_qc(paper_dir)
    assert passed is False

    report_text = (tmp_path / "paper" / "qc_report.txt").read_text()
    assert "FAIL" in report_text


def test_n_monotonicity_fail(tmp_path):
    """Non-monotone N triggers WARN but not ERROR → still returns True."""
    paper_dir = _write_paper_dir(tmp_path, key_vars=["x1"], monotone=False)
    passed = run_qc(paper_dir)
    # WARN only → should still pass
    assert passed is True

    report_text = (tmp_path / "paper" / "qc_report.txt").read_text()
    assert "WARN" in report_text
    assert "monotonicity" in report_text.lower() or "N increased" in report_text


def test_report_written(tmp_path):
    """qc_report.txt is always written to {paper_dir}/qc_report.txt."""
    paper_dir = _write_paper_dir(tmp_path, key_vars=["x1"])
    run_qc(paper_dir)

    report_path = tmp_path / "paper" / "qc_report.txt"
    assert report_path.exists()
    assert report_path.stat().st_size > 0
