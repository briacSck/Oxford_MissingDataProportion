"""
pipeline/regression_runner.py
------------------------------
Regression Runner: re-runs the original regression specification on each
listwise-deleted dataset and collects results into an Excel workbook.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.baseline_verifier import (
    _build_matrices,
    _extract_coef,
    _normalize_estimator,
    _run_regression,
)
from pipeline.config import PROPORTION_LABELS

logger = logging.getLogger(__name__)

# Exact output column order required by qc_agent
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


def _sig_flag(pval: float | None) -> str:
    if pval is None or np.isnan(pval):
        return "ns"
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return "ns"


def _sig_tier(pval: float | None) -> int:
    if pval is None or np.isnan(pval):
        return 0
    if pval < 0.01:
        return 3
    if pval < 0.05:
        return 2
    if pval < 0.10:
        return 1
    return 0


def _get_rsquared(result) -> float | None:
    try:
        return float(result.rsquared)
    except Exception:
        return None


def _regress_df(df: pd.DataFrame, spec: dict, key_var: str) -> dict:
    """Run regression on df and extract stats for key_var."""
    from pipeline.baseline_verifier import BaselineSpecError

    dep_var, X_cols, fe_cols, _entity, _time, flags = _build_matrices(df, spec)
    for f in flags:
        logger.debug("_build_matrices flag: %s", f)

    paper_id = spec.get("paper_id", "UNKNOWN")

    # Validate required columns — raise structured exception rather than
    # silently producing null stats (which would reproduce the AntiDiscrimination
    # limp-forward / all-NaN regression output pattern).
    missing_dep   = dep_var if (not dep_var or dep_var not in df.columns) else None
    missing_xcols = [v for v in X_cols if v not in df.columns]
    # Check raw spec FE cols (before _build_matrices filters them out)
    fe_cols_raw   = [v for v in (spec.get("fixed_effects") or []) if v]
    missing_fe    = [v for v in fe_cols_raw if v not in df.columns]
    cluster_var   = spec.get("cluster_var")
    missing_clust = cluster_var if (cluster_var and cluster_var not in df.columns) else None

    if missing_dep or missing_fe:
        raise BaselineSpecError(
            paper_id=paper_id,
            missing_dep_var=missing_dep,
            missing_xcols=missing_xcols,
            missing_fe_cols=missing_fe,
            missing_cluster=missing_clust,
        )

    if not X_cols:
        raise BaselineSpecError(
            paper_id=paper_id,
            missing_xcols=["(all X_cols absent or empty)"],
        )

    reg = _run_regression(df, dep_var, X_cols, fe_cols, spec)
    result = reg.get("result")
    n_obs = reg.get("n_obs") or len(df)

    coef, se, tval, pval = _extract_coef(result, key_var)
    r2 = _get_rsquared(result)

    if result is not None and coef is None:
        logger.warning(
            "[%s] _regress_df: regression succeeded but _extract_coef returned None "
            "for key_var=%r — key_var may not be in X_cols or model params. "
            "X_cols=%s",
            spec.get("paper_id", "UNKNOWN"), key_var, X_cols,
        )

    return {
        "coef": coef, "se": se, "tval": tval, "pval": pval,
        "r2": r2, "n": n_obs,
    }


def run_all_regressions(
    paper_dir: str,
    spec: dict,
) -> str:
    """Run regressions on all listwise-deleted datasets and write results to Excel.

    Parameters
    ----------
    paper_dir:
        Root directory of the paper (contains ``listwise/`` and ``baseline.parquet``).
    spec:
        Regression specification dict.

    Returns
    -------
    str
        Path to ``{paper_dir}/regression_results.xlsx``.
    """
    paper_path = Path(paper_dir)
    listwise_dir = paper_path / "listwise"
    baseline_parquet = paper_path / "baseline.parquet"

    # ── Baseline row ──────────────────────────────────────────────────────────
    baseline_df = pd.read_parquet(str(baseline_parquet))

    # ── Read selection.json for repair-pass fallback ──────────────────────────
    _selected_key_vars: list[str] = []
    _sel_path = paper_path / "selection.json"
    if _sel_path.exists():
        try:
            with open(_sel_path, encoding="utf-8") as _f:
                _sel_data = json.load(_f)
            _selected_key_vars = _sel_data.get("key_vars") or []
        except Exception as _exc:
            logger.warning("[%s] Could not read selection.json: %s",
                           spec.get("paper_id", "UNKNOWN"), _exc)

    # ── Determine effective spec + extraction target ───────────────────────────
    baseline_key_var = (spec.get("key_independent_vars") or [""])[0]
    if baseline_key_var not in baseline_df.columns and _selected_key_vars:
        # Spec's primary key var is absent (repair-pass paper).
        # Use first selection key_var that exists in baseline.
        _fallback_kv = next(
            (v for v in _selected_key_vars if v in baseline_df.columns), None
        )
        if _fallback_kv:
            baseline_key_var = _fallback_kv
            _aug_keys = _selected_key_vars + [
                v for v in (spec.get("key_independent_vars") or [])
                if v and v not in _selected_key_vars
            ]
            effective_spec = {**spec, "key_independent_vars": _aug_keys}
            logger.info(
                "[%s] run_all_regressions: spec key_var %r absent from baseline; "
                "using selection key_vars=%s, baseline_key_var=%r",
                spec.get("paper_id", "UNKNOWN"),
                (spec.get("key_independent_vars") or [""])[0],
                _selected_key_vars, baseline_key_var,
            )
        else:
            effective_spec = spec
    else:
        effective_spec = spec

    base_stats = _regress_df(baseline_df, effective_spec, baseline_key_var)

    baseline_row = {
        "Key Variable":             baseline_key_var,
        "Missing Proportion":       "baseline",
        "Post-LD N":                base_stats["n"],
        "β̂":                        base_stats["coef"],
        "SE":                       base_stats["se"],
        "t-value":                  base_stats["tval"],
        "p-value":                  base_stats["pval"],
        "Significance":             _sig_flag(base_stats["pval"]),
        "R²":                       base_stats["r2"],
        "Consistent with baseline?": "—",
    }

    base_coef = base_stats["coef"]
    base_pval = base_stats["pval"]

    # ── LD rows ───────────────────────────────────────────────────────────────
    rows: list[dict] = []

    # Sort files deterministically: by varname then by proportion label order
    label_order = {lbl: i for i, lbl in enumerate(PROPORTION_LABELS)}
    ld_files = sorted(
        listwise_dir.glob("*_MAR_*_LD.csv"),
        key=lambda p: (
            p.stem.split("_MAR_")[0],
            label_order.get(p.stem.split("_MAR_")[1].replace("_LD", ""), 999),
        ),
    )

    for csv_path in ld_files:
        stem = csv_path.stem  # e.g. "x1_MAR_01pct_LD"
        # Strip trailing "_LD"
        core = stem[:-3] if stem.endswith("_LD") else stem
        parts = core.split("_MAR_", 1)
        if len(parts) != 2:
            logger.warning("Skipping unrecognised LD filename: %s", csv_path.name)
            continue
        varname, label = parts[0], parts[1]

        df_ld = pd.read_csv(str(csv_path))
        stats = _regress_df(df_ld, effective_spec, baseline_key_var)

        # Consistency check
        coef, pval = stats["coef"], stats["pval"]
        if (
            coef is not None
            and base_coef is not None
            and not np.isnan(coef)
            and not np.isnan(base_coef)
            and np.sign(coef) == np.sign(base_coef)
            and _sig_tier(pval) == _sig_tier(base_pval)
        ):
            consistent = "Yes"
        else:
            consistent = "No"

        rows.append({
            "Key Variable":             varname,
            "Missing Proportion":       label,
            "Post-LD N":                stats["n"],
            "β̂":                        coef,
            "SE":                       stats["se"],
            "t-value":                  stats["tval"],
            "p-value":                  pval,
            "Significance":             _sig_flag(pval),
            "R²":                       stats["r2"],
            "Consistent with baseline?": consistent,
        })

    # ── Assemble and write ────────────────────────────────────────────────────
    all_rows = [baseline_row] + rows
    results_df = pd.DataFrame(all_rows, columns=_RESULT_COLS)

    out_path = paper_path / "regression_results.xlsx"
    results_df.to_excel(str(out_path), sheet_name="Results", index=False)
    logger.info("Regression results written to %s (%d rows)", out_path.name, len(results_df))

    return str(out_path)
