"""Regression tests for dep_var / column validation in baseline_verifier and regression_runner."""
import numpy as np
import pandas as pd
import pytest
from pipeline.baseline_verifier import _run_ols, _run_fe, verify_baseline, BaselineSpecError
from pipeline.regression_runner import _regress_df

RNG = np.random.default_rng(7)
N = 100


def _make_df():
    firm   = np.repeat(np.arange(10), 10)
    period = np.tile(np.arange(10), 10)
    x1     = RNG.normal(size=N)
    y      = 0.5 * x1 + RNG.normal(scale=0.3, size=N)
    return pd.DataFrame({"y": y, "x1": x1, "firm": firm, "period": period})


def _make_spec(**overrides):
    base = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "y",
        "key_independent_vars": ["x1"], "control_vars": [],
        "fixed_effects": [], "cluster_var": None, "instrumental_vars": [],
    }
    base.update(overrides)
    return base


# --- _run_ols / _run_fe safety guards ---

def test_run_ols_missing_depvar_structured_result():
    """_run_ols safety guard: absent dep_var returns structured dict, not KeyError."""
    df = _make_df()
    result = _run_ols(df, y_col="intravol", X_cols=["x1"], fe_cols=[], cluster_var=None)
    assert result["result"] is None
    assert result["n_obs"] == 0
    assert "intravol" in result.get("flag", "")


def test_run_fe_missing_depvar_structured_result():
    """_run_fe safety guard: absent dep_var returns structured dict, not KeyError."""
    df = _make_df()
    result = _run_fe(
        df, y_col="intravol", X_cols=["x1"], fe_cols=["firm"],
        cluster_var=None, entity_col="firm", time_col=None,
    )
    assert result["result"] is None
    assert result["n_obs"] == 0


# --- _regress_df raises BaselineSpecError ---

def test_regress_df_missing_depvar_raises():
    """_regress_df must raise BaselineSpecError when dep_var absent from data."""
    df = _make_df()
    spec = _make_spec(dependent_var="intravol")  # absent
    with pytest.raises(BaselineSpecError) as exc_info:
        _regress_df(df, spec, "x1")
    assert exc_info.value.missing_dep_var == "intravol"


def test_regress_df_missing_fe_col_raises():
    """_regress_df must raise BaselineSpecError when FE col absent from data."""
    df = _make_df()
    spec = _make_spec(estimator="FE", fixed_effects=["gvkey"])  # gvkey absent
    with pytest.raises(BaselineSpecError) as exc_info:
        _regress_df(df, spec, "x1")
    assert "gvkey" in exc_info.value.missing_fe_cols


# --- verify_baseline still works ---

def test_verify_baseline_missing_depvar_returns_empty_report():
    """verify_baseline existing guard: returns empty report for absent dep_var."""
    df = _make_df()
    spec = _make_spec(dependent_var="intravol")
    report = verify_baseline(df, spec, published_coef={})
    assert report["coef_estimate"] is None
    assert any("intravol" in f for f in report["flags"])


def test_verify_baseline_present_depvar_succeeds():
    """Golden path: verify_baseline produces a non-null coefficient."""
    df = _make_df()
    report = verify_baseline(df, _make_spec(), published_coef={})
    assert report["coef_estimate"] is not None
