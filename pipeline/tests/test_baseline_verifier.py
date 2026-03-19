"""
pipeline/tests/test_baseline_verifier.py
-----------------------------------------
Tests for baseline_verifier.py  (5 tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.baseline_verifier import verify_baseline

RNG = np.random.default_rng(2026)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ols_spec(**overrides) -> dict:
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


def _make_synthetic_df(n: int = 300, beta: float = 0.5) -> pd.DataFrame:
    """Synthetic DataFrame: y = beta*x1 + 0.3*x2 + noise."""
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    noise = RNG.normal(scale=0.5, size=n)
    y = beta * x1 + 0.3 * x2 + noise
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_panel_df(n_firms: int = 30, n_periods: int = 10) -> pd.DataFrame:
    """Synthetic panel DataFrame for FE tests."""
    n = n_firms * n_periods
    firm   = np.repeat(np.arange(n_firms), n_periods)
    period = np.tile(np.arange(n_periods), n_firms)
    x1     = RNG.normal(size=n)
    x2     = RNG.normal(size=n)
    fe     = RNG.normal(size=n_firms)[firm]          # firm fixed effect
    noise  = RNG.normal(scale=0.3, size=n)
    y      = 0.5 * x1 + 0.2 * x2 + fe + noise
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "period": period})


# ── Test 1: OLS runs ───────────────────────────────────────────────────────────

def test_ols_runs():
    df   = _make_synthetic_df()
    spec = _make_ols_spec()
    report = verify_baseline(df, spec, published_coef={})
    assert report["coef_estimate"] is not None, "OLS should produce a coefficient estimate"


# ── Test 2: OLS coefficient close to true β ───────────────────────────────────

def test_ols_coef_close_to_truth():
    true_beta = 0.5
    df   = _make_synthetic_df(n=1000, beta=true_beta)
    spec = _make_ols_spec()
    report = verify_baseline(df, spec, published_coef={})
    assert report["coef_estimate"] is not None
    assert abs(report["coef_estimate"] - true_beta) < 0.1, (
        f"OLS estimate {report['coef_estimate']:.4f} too far from truth {true_beta}"
    )


# ── Test 3: empty published_coef → match=None, no exception ──────────────────

def test_empty_published_coef():
    df   = _make_synthetic_df()
    spec = _make_ols_spec()
    report = verify_baseline(df, spec, published_coef={})

    assert report["match"] is None, "match should be None when published_coef is empty"
    assert isinstance(report["flags"], list)
    assert any("no published coefficient" in f.lower() for f in report["flags"])


# ── Test 4: coefficient mismatch → match=False, discrepancies non-empty ───────

def test_coef_mismatch():
    df   = _make_synthetic_df(n=1000, beta=0.5)
    spec = _make_ols_spec()
    # Deliberately wrong published value
    wrong_published = {"x1": 99.0}
    report = verify_baseline(df, spec, published_coef=wrong_published)

    assert report["match"] is False, "Should be False when estimate ≠ published"
    assert len(report["discrepancies"]) > 0, "Should have at least one discrepancy entry"
    disc = report["discrepancies"][0]
    assert "difference" in disc
    assert disc["difference"] > 1.0  # large difference


# ── Test 5: FE spec runs ───────────────────────────────────────────────────────

def test_fe_runs():
    df = _make_panel_df()
    spec = _make_ols_spec(
        estimator="FE",
        fixed_effects=["firm"],
        key_independent_vars=["x1"],
        control_vars=["x2"],
    )
    report = verify_baseline(df, spec, published_coef={})
    assert report["coef_estimate"] is not None, (
        "FE regression should produce a coefficient estimate (or OLS fallback)"
    )
