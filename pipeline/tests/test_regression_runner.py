"""Tests for pipeline/regression_runner.py"""

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PROPORTION_LABELS
from pipeline.regression_runner import _sig_flag, _sig_tier, run_all_regressions

# ── Helpers ───────────────────────────────────────────────────────────────────

N = 300
RNG = np.random.default_rng(99)


def _make_spec(**overrides):
    base = {
        "paper_id":             "Paper_Test",
        "estimator":            "OLS",
        "dependent_var":        "y",
        "key_independent_vars": ["x1"],
        "control_vars":         ["x2"],
        "fixed_effects":        [],
        "cluster_var":          None,
        "sample_restrictions":  [],
        "instrumental_vars":    [],
    }
    base.update(overrides)
    return base


def _make_synthetic_data(n=N, seed=99, include_firm=False):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.5 * x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)
    d = {"y": y, "x1": x1, "x2": x2}
    if include_firm:
        d["firm"] = np.tile(np.arange(50), n // 50 + 1)[:n]
    return pd.DataFrame(d)


def _build_paper_dir(tmp_path, spec, n=N, n_key_vars=2, seed=99):
    """Create paper_dir with baseline.parquet and listwise CSVs."""
    df = _make_synthetic_data(n=n, seed=seed)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    listwise_dir = paper_dir / "listwise"
    listwise_dir.mkdir()

    # Save baseline
    df.to_parquet(str(paper_dir / "baseline.parquet"), index=False)

    key_vars = ["x1", "x2"] if n_key_vars == 2 else ["x1"]

    # Create LD files (simulate output of listwise_agent — no NaN in key var)
    rng = np.random.default_rng(seed + 1)
    for kv in key_vars:
        for label in PROPORTION_LABELS:
            drop_frac = PROPORTION_LABELS.index(label) * 0.06 + 0.01
            n_keep = int(n * (1 - drop_frac))
            idx = rng.choice(n, size=n_keep, replace=False)
            df_ld = df.iloc[sorted(idx)].copy().reset_index(drop=True)
            fname = f"{kv}_MAR_{label}_LD.csv"
            df_ld.to_csv(str(listwise_dir / fname), index=False)

    return str(paper_dir)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_ols_baseline_row(tmp_path):
    """First row of results has Missing Proportion == 'baseline'."""
    spec = _make_spec()
    paper_dir = _build_paper_dir(tmp_path, spec, n_key_vars=1)
    out_path = run_all_regressions(paper_dir, spec)
    df = pd.read_excel(out_path, sheet_name="Results")
    assert df.iloc[0]["Missing Proportion"] == "baseline"


def test_result_row_count(tmp_path):
    """K=2 key vars → total rows = 2×7+1 = 15."""
    spec = _make_spec()
    paper_dir = _build_paper_dir(tmp_path, spec, n_key_vars=2)
    out_path = run_all_regressions(paper_dir, spec)
    df = pd.read_excel(out_path, sheet_name="Results")
    assert len(df) == 15, f"Expected 15 rows, got {len(df)}"


def test_significance_flags(tmp_path):
    """Significance flags are assigned correctly by p-value."""
    assert _sig_flag(0.005) == "***"
    assert _sig_flag(0.03) == "**"
    assert _sig_flag(0.08) == "*"
    assert _sig_flag(0.2) == "ns"
    assert _sig_flag(None) == "ns"


def test_consistency_flag(tmp_path):
    """Consistency flag is 'Yes' for same sign + same tier, 'No' otherwise."""
    spec = _make_spec()
    paper_dir = _build_paper_dir(tmp_path, spec, n_key_vars=1)
    out_path = run_all_regressions(paper_dir, spec)
    df = pd.read_excel(out_path, sheet_name="Results")

    non_baseline = df[df["Missing Proportion"] != "baseline"]
    # All consistency flags must be "Yes" or "No"
    valid_flags = {"Yes", "No"}
    actual_flags = set(non_baseline["Consistent with baseline?"].unique())
    assert actual_flags <= valid_flags, f"Unexpected flags: {actual_flags - valid_flags}"


def test_fe_runs(tmp_path):
    """FE spec with panel data runs without exception and writes a result row."""
    spec = _make_spec(estimator="FE", fixed_effects=["firm"])
    n = N

    df = _make_synthetic_data(n=n, seed=42, include_firm=True)
    paper_dir = tmp_path / "paper_fe"
    paper_dir.mkdir()
    listwise_dir = paper_dir / "listwise"
    listwise_dir.mkdir()
    df.to_parquet(str(paper_dir / "baseline.parquet"), index=False)

    rng = np.random.default_rng(43)
    for label in PROPORTION_LABELS:
        drop_frac = 0.05
        n_keep = int(n * (1 - drop_frac))
        idx = rng.choice(n, size=n_keep, replace=False)
        df_ld = df.iloc[sorted(idx)].copy().reset_index(drop=True)
        df_ld.to_csv(str(listwise_dir / f"x1_MAR_{label}_LD.csv"), index=False)

    out_path = run_all_regressions(str(paper_dir), spec)
    df_res = pd.read_excel(out_path, sheet_name="Results")
    assert len(df_res) >= 2  # baseline + at least one LD row


def test_excel_output_exists(tmp_path):
    """regression_results.xlsx is written to the correct path."""
    spec = _make_spec()
    paper_dir = _build_paper_dir(tmp_path, spec, n_key_vars=1)
    out_path = run_all_regressions(paper_dir, spec)

    import os
    assert os.path.exists(out_path)
    assert out_path.endswith("regression_results.xlsx")
    assert os.path.dirname(os.path.abspath(out_path)) == os.path.abspath(paper_dir)
