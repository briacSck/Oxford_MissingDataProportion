"""Tests for run_all_regressions with repair-pass selection (spec key var absent)."""
import json
import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline.regression_runner import run_all_regressions

RNG = np.random.default_rng(42)
N = 150


def _make_df():
    return pd.DataFrame({
        "dep":       RNG.normal(size=N),
        "repair_v1": RNG.uniform(0.1, 5.0, size=N),
        "repair_v2": RNG.uniform(0.1, 3.0, size=N),
        "repair_v3": RNG.uniform(0.1, 2.0, size=N),
        "ctrl":      RNG.normal(size=N),
    })


def _spec_absent():
    return {
        "paper_id": "P_RepairTest", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["absent_key_var"],  # NOT in baseline
        "control_vars": ["ctrl"],
        "fixed_effects": [], "cluster_var": None, "instrumental_vars": [],
    }


def _write_selection(path, key_vars):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "paper_id": "P_RepairTest",
            "key_vars": key_vars,
            "aux_var": "ctrl",
            "selection_repaired": "baseline_columns",
            "human_confirmed": True,
        }, f)


def _write_ld(tmp_path, df, varname, label="10pct"):
    df_ld = df.copy()
    df_ld.loc[:10, varname] = np.nan
    df_ld.dropna().to_csv(
        str(tmp_path / "listwise" / f"{varname}_MAR_{label}_LD.csv"), index=False
    )


def test_repair_pass_produces_nonnull_coef(tmp_path):
    """When spec key var absent, selection.json key_vars yield non-NaN coef."""
    df = _make_df()
    (tmp_path / "listwise").mkdir()
    pq.write_table(pa.Table.from_pandas(df), str(tmp_path / "baseline.parquet"))
    _write_selection(tmp_path / "selection.json", ["repair_v1", "repair_v2", "repair_v3"])
    _write_ld(tmp_path, df, "repair_v1")

    out = run_all_regressions(str(tmp_path), _spec_absent())
    result_df = pd.read_excel(out)
    ld_row = result_df[result_df["Missing Proportion"] == "10pct"]
    assert len(ld_row) == 1
    coef = ld_row.iloc[0]["β̂"]
    assert coef is not None and not np.isnan(float(coef)), (
        "repair-pass LD row must have non-NaN coef"
    )


def test_normal_spec_unaffected_when_key_var_in_baseline(tmp_path):
    """When spec key var IS in baseline, selection.json does not alter behaviour."""
    df = _make_df()
    (tmp_path / "listwise").mkdir()
    pq.write_table(pa.Table.from_pandas(df), str(tmp_path / "baseline.parquet"))
    _write_selection(tmp_path / "selection.json", ["repair_v1", "repair_v2", "repair_v3"])

    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["repair_v1"],  # present in baseline
        "control_vars": ["repair_v2", "ctrl"],
        "fixed_effects": [], "cluster_var": None, "instrumental_vars": [],
    }
    _write_ld(tmp_path, df, "repair_v1")

    out = run_all_regressions(str(tmp_path), spec)
    result_df = pd.read_excel(out)
    ld_row = result_df[result_df["Missing Proportion"] == "10pct"]
    coef = ld_row.iloc[0]["β̂"]
    assert coef is not None and not np.isnan(float(coef))


def test_no_selection_json_does_not_crash(tmp_path):
    """Absent selection.json must not crash run_all_regressions."""
    df = _make_df()
    (tmp_path / "listwise").mkdir()
    pq.write_table(pa.Table.from_pandas(df), str(tmp_path / "baseline.parquet"))
    spec = {
        "paper_id": "P_Test", "estimator": "OLS",
        "dependent_var": "dep",
        "key_independent_vars": ["repair_v1"],
        "control_vars": ["repair_v2"],
        "fixed_effects": [], "cluster_var": None, "instrumental_vars": [],
    }
    _write_ld(tmp_path, df, "repair_v1")

    out = run_all_regressions(str(tmp_path), spec)
    assert out.endswith("regression_results.xlsx")
