"""Tests for regression coefficient extraction correctness."""
import numpy as np
import pandas as pd
import pytest
from pipeline.regression_runner import _regress_df, run_all_regressions
from pipeline.baseline_verifier import _extract_coef

RNG = np.random.default_rng(55)
N = 200


def _make_df():
    x1 = RNG.normal(size=N)
    x2 = RNG.uniform(0.1, 5.0, size=N)
    y  = 0.7 * x1 + 0.3 * x2 + RNG.normal(scale=0.2, size=N)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_spec(**overrides):
    base = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "y",
        "key_independent_vars": ["x1"],
        "control_vars": ["x2"],
        "fixed_effects": [], "cluster_var": None, "instrumental_vars": [],
    }
    base.update(overrides)
    return base


# ── _extract_coef: present key_var ───────────────────────────────────────────

def test_extract_coef_returns_nonnull_when_keyvar_in_params():
    """_extract_coef must return non-None coef when key_var is in model params."""
    import statsmodels.api as sm
    df = _make_df()
    X = sm.add_constant(df[["x1", "x2"]])
    result = sm.OLS(df["y"], X).fit()
    coef, se, tval, pval = _extract_coef(result, "x1")
    assert coef is not None, "coef must be non-None when x1 is in params"
    assert se   is not None
    assert tval is not None
    assert pval is not None


def test_extract_coef_returns_none_when_keyvar_missing():
    """_extract_coef must return (None, None, None, None) when key_var absent."""
    import statsmodels.api as sm
    df = _make_df()
    X = sm.add_constant(df[["x1", "x2"]])
    result = sm.OLS(df["y"], X).fit()
    coef, se, tval, pval = _extract_coef(result, "absent_var")
    assert coef is None
    assert se   is None
    assert tval is None
    assert pval is None


# ── _regress_df: extraction uses key_var ─────────────────────────────────────

def test_regress_df_returns_nonnull_coef_when_keyvar_in_model():
    """_regress_df must return non-None coef when key_var (x1) is in X_cols."""
    df = _make_df()
    spec = _make_spec()
    stats = _regress_df(df, spec, "x1")
    assert stats["coef"] is not None, "coef must be non-None when x1 is in spec and data"
    assert stats["se"]   is not None
    assert stats["r2"]   is not None


def test_regress_df_all_nan_treated_as_failure():
    """If all coef stats are None, the output is distinguishable from success."""
    df = _make_df()
    spec = _make_spec()
    # Pass a key_var that is NOT in the model params
    stats = _regress_df(df, spec, "absent_key_var")
    # coef is None — treat as extraction failure
    assert stats["coef"] is None, "absent key_var must yield None coef (not silent NaN)"
    # But regression still ran, so r2 is present
    assert stats["r2"] is not None, "R² must be present even when coef extraction fails"


# ── run_all_regressions: always extracts baseline_key_var ────────────────────

def test_run_all_regressions_extracts_baseline_key_var(tmp_path):
    """LD rows must report the spec's main key variable coefficient, not varname."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build a simple paper directory
    paper_path = tmp_path
    (paper_path / "listwise").mkdir()

    # Baseline parquet
    df = _make_df()
    pq.write_table(pa.Table.from_pandas(df), str(paper_path / "baseline.parquet"))

    # Simulate 1 LD file where missingness was injected into x2 (a control)
    # This is the scenario that used to fail: varname="x2", but key_var should be "x1"
    df_ld = df.copy()
    df_ld.loc[:10, "x2"] = np.nan  # inject some missingness into x2
    df_ld.dropna().to_csv(str(paper_path / "listwise" / "x2_MAR_10pct_LD.csv"), index=False)

    spec = _make_spec()
    out_path = run_all_regressions(str(paper_path), spec)

    result_df = pd.read_excel(out_path)
    ld_row = result_df[result_df["Missing Proportion"] == "10pct"]
    assert len(ld_row) == 1
    coef = ld_row.iloc[0]["β̂"]
    assert coef is not None and not np.isnan(float(coef)), (
        "LD row must have a non-NaN coefficient when baseline_key_var (x1) is in model"
    )


def test_extract_coef_with_linearmodels_result():
    """_extract_coef must work with linearmodels AbsorbingLS (std_errors/tstats API)."""
    pytest.importorskip("linearmodels")
    from linearmodels import AbsorbingLS
    df = _make_df()
    N = len(df)
    entity = np.repeat(np.arange(20), N // 20 + 1)[:N]
    idx = pd.MultiIndex.from_arrays([entity, np.arange(N)])
    y = pd.Series(df["y"].values, index=idx, name="y")
    X = pd.DataFrame({"x1": df["x1"].values, "x2": df["x2"].values}, index=idx)
    absorb = pd.DataFrame({"entity": pd.Categorical(entity)}, index=idx)
    result = AbsorbingLS(y, X, absorb=absorb).fit()
    coef, se, tval, pval = _extract_coef(result, "x1")
    assert coef is not None, "coef must be non-None for linearmodels AbsorbingLS result"
    assert se   is not None, "se must be non-None (std_errors not bse)"
    assert tval is not None, "tval must be non-None (tstats not tvalues)"
    assert pval is not None
