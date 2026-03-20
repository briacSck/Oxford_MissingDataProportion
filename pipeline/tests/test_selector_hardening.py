"""Regression tests for variable_selector hardening."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pipeline.variable_selector import _filter_eligible_vars, _find_baseline_candidates

RNG = np.random.default_rng(99)
N = 500


def _make_fang_like_df():
    """DataFrame with prodummy (province code 1-31) and continuous vars."""
    return pd.DataFrame({
        "y":          RNG.normal(size=N),
        "lnasset":    RNG.normal(size=N),
        "prodummy":   RNG.integers(1, 32, size=N),  # province codes
        "lleverage":  RNG.uniform(0.1, 0.9, size=N),
    })


def _make_fang_like_spec():
    return {
        "paper_id": "P_Test",
        "estimator": "OLS",
        "dependent_var": "y",
        "key_independent_vars": ["lnasset"],
        "control_vars": ["prodummy", "lleverage"],
        "fixed_effects": [],
        "instrumental_vars": [],
    }


def test_prodummy_excluded_by_rule_11():
    """prodummy (integer 1-31 named *dummy) must be excluded from eligibility."""
    df = _make_fang_like_df()
    spec = _make_fang_like_spec()
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert "prodummy" not in eligible, "prodummy must be excluded as categorical code"
    assert "prodummy" in excluded


def test_continuous_vars_unaffected_by_rule_11():
    """Continuous numeric controls must not be excluded by Rule 11."""
    df = _make_fang_like_df()
    spec = _make_fang_like_spec()
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert "lleverage" in eligible, "lleverage must remain eligible"


def test_repair_pass_finds_baseline_candidates():
    """When spec vars are absent from baseline, repair finds baseline columns."""
    df = pd.DataFrame({
        "y":      RNG.normal(size=N),
        "colA":   RNG.normal(size=N),
        "colB":   RNG.uniform(0.1, 5.0, size=N),
    })
    spec = {
        "paper_id": "P_Test",
        "dependent_var": "y",
        "key_independent_vars": ["absent_var"],
        "control_vars": ["another_absent"],
        "fixed_effects": [],
        "instrumental_vars": [],
    }
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert len(eligible) == 0, "all spec vars absent → eligible must be empty"
    repair, source = _find_baseline_candidates(df, spec, eligible, excluded)
    assert len(repair) >= 2, "repair must find colA and colB from baseline"
    assert "colA" in repair
    assert "colB" in repair
    assert "y" not in repair  # dep_var excluded


def test_repair_pass_not_invoked_when_eligible_exists():
    """Repair is a fallback only; verify _find_baseline_candidates on non-empty pool."""
    df = _make_fang_like_df()
    spec = _make_fang_like_spec()
    eligible, excluded = _filter_eligible_vars(spec, df)
    # eligible is non-empty, so repair only adds NEW candidates
    repair, source = _find_baseline_candidates(df, spec, eligible, excluded)
    # prodummy already excluded → repair must not re-add it
    assert "prodummy" not in repair


# ── Rule 12: compound time detection ─────────────────────────────────────────

def test_compound_time_quartertime_excluded(tmp_path):
    """'quartertime' must be excluded: ends with time token 'time'."""
    from pipeline.variable_selector import _filter_eligible_vars
    N = 100
    df = pd.DataFrame({
        "dep": np.random.default_rng(1).normal(size=N),
        "quartertime": np.tile(np.arange(1, 15), N // 14 + 1)[:N].astype(float),
        "revenue":     np.random.default_rng(2).uniform(1, 100, size=N),
    })
    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["quartertime", "revenue"],
        "control_vars": [], "fixed_effects": [], "cluster_var": None,
        "instrumental_vars": [],
    }
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert "quartertime" not in eligible, "compound time var must be excluded"
    assert "revenue" in eligible, "unrelated var must remain eligible"


def test_compound_time_not_applied_to_unrelated(tmp_path):
    """'profitmargin' must NOT be excluded: no time token as start/end."""
    from pipeline.variable_selector import _filter_eligible_vars
    N = 100
    df = pd.DataFrame({
        "dep":          np.random.default_rng(3).normal(size=N),
        "profitmargin": np.random.default_rng(4).uniform(0.01, 0.5, size=N),
    })
    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["profitmargin"],
        "control_vars": [], "fixed_effects": [], "cluster_var": None,
        "instrumental_vars": [],
    }
    eligible, _ = _filter_eligible_vars(spec, df)
    assert "profitmargin" in eligible


# ── Rule 13: count variable detection ────────────────────────────────────────

def test_count_var_n_excluded():
    """'n' with integer-like dtype and <200 unique values must be excluded."""
    from pipeline.variable_selector import _filter_eligible_vars
    N = 200
    df = pd.DataFrame({
        "dep":     np.random.default_rng(5).normal(size=N),
        "revenue": np.random.default_rng(6).uniform(1, 100, size=N),
        "n":       np.tile(np.arange(1, 37), N // 36 + 1)[:N].astype(float),
    })
    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["n", "revenue"],
        "control_vars": [], "fixed_effects": [], "cluster_var": None,
        "instrumental_vars": [],
    }
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert "n" not in eligible, "integer-like count var 'n' must be excluded"
    assert "revenue" in eligible


def test_count_var_continuous_float_not_excluded():
    """'n' as a continuous float with many unique values must NOT be excluded."""
    from pipeline.variable_selector import _filter_eligible_vars
    N = 500
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "dep": rng.normal(size=N),
        "n":   rng.uniform(0.001, 9.999, size=N),  # continuous, not integer-like
    })
    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["n"],
        "control_vars": [], "fixed_effects": [], "cluster_var": None,
        "instrumental_vars": [],
    }
    eligible, _ = _filter_eligible_vars(spec, df)
    assert "n" in eligible, "continuous float 'n' must not be excluded by count rule"


# ── Rule 14: dep_var transform detection ─────────────────────────────────────

def test_level_of_log_depvar_excluded():
    """Candidate that is the level form of a log dep_var must be excluded."""
    from pipeline.variable_selector import _filter_eligible_vars
    N = 100
    rng = np.random.default_rng(8)
    sales = rng.uniform(1, 100, size=N)
    df = pd.DataFrame({
        "logSales": np.log(sales),
        "Sales":    sales,
        "revenue":  rng.uniform(1, 50, size=N),
    })
    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "logSales",
        "key_independent_vars": ["Sales", "revenue"],
        "control_vars": [], "fixed_effects": [], "cluster_var": None,
        "instrumental_vars": [],
    }
    eligible, excluded = _filter_eligible_vars(spec, df)
    assert "Sales" not in eligible, "level form of log dep_var must be excluded"
    assert "revenue" in eligible


def test_depvar_transform_suffix_excluded():
    """lav_Visits excluded when dep_var is logTotalVisits (suffix containment)."""
    from pipeline.variable_selector import _is_depvar_transform
    assert _is_depvar_transform("lav_Visits", "logTotalVisits"), (
        "lav_Visits should be detected as transform via suffix 'visits' ⊂ 'totalvisits'"
    )
    assert not _is_depvar_transform("lav_Counties", "logTotalVisits"), (
        "lav_Counties must NOT be flagged as a transform of logTotalVisits"
    )
    assert not _is_depvar_transform("logTotalPIs", "logTotalVisits"), (
        "logTotalPIs must NOT be flagged as a transform of logTotalVisits"
    )
