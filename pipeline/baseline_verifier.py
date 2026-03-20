"""
pipeline/baseline_verifier.py
------------------------------
Baseline Verifier: runs the original regression specification on the clean baseline
DataFrame, extracts the key coefficient, compares to the published value (if entered),
and returns a verification report. Gate 1 fires if mismatch or no published value.

Usage
-----
    python pipeline/baseline_verifier.py Paper_Meyer2024
    python pipeline/baseline_verifier.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Custom exception ──────────────────────────────────────────────────────────

class BaselineSpecError(RuntimeError):
    """Raised when required columns are absent from the baseline DataFrame.

    Attributes
    ----------
    missing_dep_var : str | None
    missing_xcols   : list[str]
    missing_fe_cols : list[str]
    missing_cluster : str | None
    """
    def __init__(self, paper_id: str, missing_dep_var=None,
                 missing_xcols=None, missing_fe_cols=None, missing_cluster=None):
        self.missing_dep_var  = missing_dep_var
        self.missing_xcols    = missing_xcols or []
        self.missing_fe_cols  = missing_fe_cols or []
        self.missing_cluster  = missing_cluster
        parts = []
        if missing_dep_var:
            parts.append(f"dep_var={missing_dep_var!r}")
        if missing_xcols:
            parts.append(f"X_cols={missing_xcols}")
        if missing_fe_cols:
            parts.append(f"fe_cols={missing_fe_cols}")
        if missing_cluster:
            parts.append(f"cluster_var={missing_cluster!r}")
        super().__init__(
            f"[{paper_id}] BaselineSpecError — columns absent from data: "
            + "; ".join(parts)
        )


# ── Estimator normalisation ───────────────────────────────────────────────────

_ESTIMATOR_MAP: dict[str, str] = {
    "REG":        "OLS",
    "REGRESS":    "OLS",
    "AREG":       "OLS",
    "REGHDFE":    "FE",
    "XTREG":      "FE",
    "XTLOGIT":    "LOGIT",
    "XTPROBIT":   "PROBIT",
    "IVREGRESS":  "IV",
    "IVREG2":     "IV",
    "XTIVREG":    "IV",
    "XTGLS":      "GLS",
    "XTMIXED":    "HLM",
    "MIXED":      "HLM",
}


def _normalize_estimator(est: str) -> str:
    upper = est.upper().strip()
    # "xtreg fe" or "xtreg, fe"
    if upper.startswith("XTREG"):
        return "RE" if "RE" in upper and "FE" not in upper else "FE"
    first_token = upper.split()[0]
    return _ESTIMATOR_MAP.get(first_token, first_token)


# ── Empty/error report ────────────────────────────────────────────────────────

def _empty_report(**overrides) -> dict:
    base: dict = {
        "match":           None,
        "tolerance":       0.005,
        "key_coef_name":   "",
        "coef_estimate":   None,
        "coef_published":  None,
        "se_estimate":     None,
        "tstat_estimate":  None,
        "pvalue_estimate": None,
        "n_obs":           None,
        "results":         None,
        "discrepancies":   [],
        "flags":           [],
    }
    base.update(overrides)
    return base


# ── Matrix builder ────────────────────────────────────────────────────────────

def _build_matrices(df: pd.DataFrame, spec: dict):
    """Return (dep_var, X_cols, fe_cols, entity_col, time_col, build_flags)."""
    flags: list[str] = []
    dep_var: str = spec.get("dependent_var") or ""

    candidates: list[str] = (
        list(spec.get("key_independent_vars") or [])
        + list(spec.get("control_vars") or [])
    )
    fe_cols_raw: list[str] = list(spec.get("fixed_effects") or [])

    # Build FE set for deduplication
    fe_set = {c for c in fe_cols_raw if c}

    X_cols: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if not c or c in seen:
            continue
        if re.search(r'[*()\[\]]', c):
            flags.append(f"skipped wildcard/malformed column: '{c}'")
            seen.add(c)
            continue
        if c == dep_var or c in fe_set:
            seen.add(c)
            continue
        if c in df.columns:
            X_cols.append(c)
        else:
            flags.append(f"column '{c}' not in data — skipped from regression")
        seen.add(c)

    fe_cols = [c for c in fe_cols_raw if c and c in df.columns]
    entity_col: Optional[str] = fe_cols[0] if fe_cols else None
    time_col:   Optional[str] = fe_cols[1] if len(fe_cols) > 1 else None

    return dep_var, X_cols, fe_cols, entity_col, time_col, flags


# ── Regression runners ────────────────────────────────────────────────────────

def _run_ols(
    df: pd.DataFrame,
    y_col: str,
    X_cols: list[str],
    fe_cols: list[str],
    cluster_var: Optional[str],
) -> dict:
    """OLS with optional FE dummies and clustering."""
    if not y_col or y_col not in df.columns:
        return {"result": None, "work": pd.DataFrame(), "n_obs": 0,
                "flag": f"dep_var '{y_col}' not found in data"}
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    cols_needed = list({y_col} | set(X_cols) | set(fe_cols))
    if cluster_var and cluster_var in df.columns:
        cols_needed.append(cluster_var)
    cols_needed = [c for c in cols_needed if c in df.columns]

    work = df[cols_needed].copy().dropna()
    if work.empty or len(work) <= len(X_cols) + 1:
        return {"result": None, "work": work, "n_obs": 0,
                "flag": "insufficient observations for OLS"}

    # FE dummies
    if fe_cols:
        fe_present = [c for c in fe_cols if c in work.columns]
        if fe_present:
            n_categories = work[fe_present].nunique().max()
            if n_categories > 200:
                flags = [f"FE column has {n_categories} levels — may be slow"]
            dummies = pd.get_dummies(work[fe_present], drop_first=True, prefix_sep="__")
            X_df = pd.concat([work[X_cols].reset_index(drop=True),
                               dummies.reset_index(drop=True)], axis=1)
        else:
            X_df = work[X_cols].copy()
    else:
        X_df = work[X_cols].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_with_const = sm.add_constant(X_df, has_constant="add")

    y_s = work[y_col].reset_index(drop=True)

    model = sm.OLS(y_s, X_with_const.reset_index(drop=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cluster_var and cluster_var in work.columns:
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": work[cluster_var].reset_index(drop=True)},
            )
        else:
            result = model.fit()

    return {"result": result, "work": work, "n_obs": len(work)}


def _run_fe(
    df: pd.DataFrame,
    y_col: str,
    X_cols: list[str],
    fe_cols: list[str],
    cluster_var: Optional[str],
    entity_col: str,
    time_col: Optional[str],
) -> dict:
    """FE via linearmodels AbsorbingLS; falls back to OLS with dummies on failure."""
    if not y_col or y_col not in df.columns:
        return {"result": None, "work": pd.DataFrame(), "n_obs": 0,
                "flag": f"dep_var '{y_col}' not found in data"}
    try:
        from linearmodels import AbsorbingLS  # type: ignore
    except ImportError:
        return _run_ols(df, y_col, X_cols, fe_cols, cluster_var)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    cols_needed = list(set([y_col] + X_cols + fe_cols))
    if cluster_var and cluster_var in df.columns:
        cols_needed.append(cluster_var)
    cols_needed = list(dict.fromkeys(c for c in cols_needed if c in df.columns))

    work = df[cols_needed].copy().dropna()
    if work.empty:
        return {"result": None, "work": work, "n_obs": 0,
                "flag": "no observations after dropna"}

    # Build absorb frame before creating panel index
    absorb_df = pd.DataFrame(index=work.index)
    for c in fe_cols:
        if c in work.columns:
            absorb_df[c] = pd.Categorical(work[c])

    # Build panel MultiIndex
    if time_col and time_col in work.columns:
        time_vals = work[time_col].values
    else:
        time_vals = np.arange(len(work))

    idx = pd.MultiIndex.from_arrays(
        [work[entity_col].values, time_vals],
        names=[entity_col, time_col or "_time_"],
    )

    y_s  = pd.Series(work[y_col].values, index=idx, name=y_col)
    X_df = pd.DataFrame(work[X_cols].values, index=idx, columns=X_cols)
    absorb_indexed = pd.DataFrame(
        {c: absorb_df[c].values for c in absorb_df.columns},
        index=idx,
    )

    try:
        mod = AbsorbingLS(
            y_s, X_df,
            absorb=absorb_indexed if not absorb_indexed.empty else None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cluster_var and cluster_var in work.columns:
                result = mod.fit(
                    cov_type="clustered",
                    clusters=pd.Series(work[cluster_var].values, index=idx),
                )
            else:
                result = mod.fit()
        return {"result": result, "work": work, "n_obs": len(work)}
    except Exception as exc:
        # Fall back to OLS with dummies
        return _run_ols(df, y_col, X_cols, fe_cols, cluster_var)


def _run_logit_probit(
    df: pd.DataFrame,
    y_col: str,
    X_cols: list[str],
    estimator: str,
) -> dict:
    work = df[[y_col] + X_cols].dropna()
    if work.empty:
        return {"result": None, "work": work, "n_obs": 0, "flag": "no observations"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_with_const = sm.add_constant(work[X_cols], has_constant="add")

    ModelCls = sm.Logit if estimator == "LOGIT" else sm.Probit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ModelCls(work[y_col], X_with_const).fit(disp=0, cov_type="HC1")
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ModelCls(work[y_col], X_with_const).fit(disp=0)
    return {"result": result, "work": work, "n_obs": len(work)}


def _run_regression(
    df: pd.DataFrame,
    y_col: str,
    X_cols: list[str],
    fe_cols: list[str],
    spec: dict,
) -> dict:
    est = _normalize_estimator(spec.get("estimator") or "OLS")
    cluster_var: Optional[str] = spec.get("cluster_var")
    entity_col: Optional[str] = fe_cols[0] if fe_cols else None
    time_col:   Optional[str] = fe_cols[1] if len(fe_cols) > 1 else None

    if est == "OLS":
        return _run_ols(df, y_col, X_cols, fe_cols, cluster_var)
    elif est in ("FE", "RE"):
        if not entity_col:
            return _run_ols(df, y_col, X_cols, [], cluster_var)
        return _run_fe(df, y_col, X_cols, fe_cols, cluster_var, entity_col, time_col)
    elif est == "LOGIT":
        return _run_logit_probit(df, y_col, X_cols, "LOGIT")
    elif est == "PROBIT":
        return _run_logit_probit(df, y_col, X_cols, "PROBIT")
    else:
        return {
            "result": None, "work": None, "n_obs": None,
            "flag": f"estimator '{spec.get('estimator')}' not yet implemented",
        }


# ── Coefficient extraction ────────────────────────────────────────────────────

def _extract_coef(result, key_var: str) -> tuple:
    """Return (coef, se, tstat, pvalue) for key_var from a fitted result."""
    if result is None or not key_var:
        return None, None, None, None
    try:
        params = result.params
        if key_var in params.index:
            name = key_var
        else:
            matches = [i for i in params.index if key_var in str(i)]
            if not matches:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "_extract_coef: key_var=%r not found in params %s",
                    key_var, list(params.index)[:10],
                )
                return None, None, None, None
            name = matches[0]

        coef = float(params[name])
        # Dual-API: statsmodels uses .bse / .tvalues; linearmodels uses .std_errors / .tstats
        _bse   = getattr(result, "bse",     getattr(result, "std_errors", None))
        _tvals = getattr(result, "tvalues", getattr(result, "tstats",     None))
        _pvals = getattr(result, "pvalues", None)
        se     = float(_bse[name])   if _bse   is not None else None
        tstat  = float(_tvals[name]) if _tvals is not None else None
        pvalue = float(_pvals[name]) if _pvals is not None else None
        return coef, se, tstat, pvalue
    except Exception:
        return None, None, None, None


# ── Comparison ────────────────────────────────────────────────────────────────

def _compare_coef(
    estimate: Optional[float],
    published: Optional[float],
    tolerance: float = 0.005,
) -> tuple:
    if estimate is None or published is None:
        return None, {}
    diff = abs(estimate - published)
    match = diff <= tolerance
    discrepancy = {} if match else {
        "estimate":   estimate,
        "published":  published,
        "difference": round(diff, 6),
        "tolerance":  tolerance,
    }
    return match, discrepancy


# ── Public entry ──────────────────────────────────────────────────────────────


def verify_baseline(
    baseline_df: pd.DataFrame,
    spec: dict,
    published_coef: dict[str, float],
) -> dict:
    """Run the baseline regression and compare to published coefficients.

    Parameters
    ----------
    baseline_df:
        Clean baseline DataFrame from ``prepare_baseline``.
    spec:
        Regression specification dict from ``parse_paper``.
    published_coef:
        Mapping of variable name → published coefficient value.
        Empty dict is valid; Gate 1 will show estimate only.

    Returns
    -------
    dict
        Verification report with keys: match, tolerance, key_coef_name,
        coef_estimate, coef_published, se_estimate, tstat_estimate,
        pvalue_estimate, n_obs, results, discrepancies, flags.
    """
    flags: list[str] = []

    key_vars = list(spec.get("key_independent_vars") or [])
    key_coef_name = key_vars[0] if key_vars else ""

    dep_var = spec.get("dependent_var") or ""
    if not dep_var or dep_var not in baseline_df.columns:
        return _empty_report(
            key_coef_name=key_coef_name,
            flags=[f"dependent variable '{dep_var}' not found in data"],
        )

    dep_var, X_cols, fe_cols, entity_col, time_col, build_flags = _build_matrices(
        baseline_df, spec
    )
    flags.extend(build_flags)

    if not X_cols:
        flags.append("no valid X columns found — cannot run regression")
        return _empty_report(key_coef_name=key_coef_name, flags=flags)

    # Run regression
    reg_result = _run_regression(baseline_df, dep_var, X_cols, fe_cols, spec)
    if "flag" in reg_result:
        flags.append(reg_result["flag"])

    result = reg_result.get("result")
    n_obs  = reg_result.get("n_obs")

    # Extract key coefficient
    coef, se, tstat, pval = _extract_coef(result, key_coef_name)

    # Compare to published
    if not published_coef:
        flags.append(
            "no published coefficient entered — Gate 1 will show estimate only; "
            "verify manually"
        )
        match_val    = None
        discrepancies: list[dict] = []
    else:
        pub_val = published_coef.get(key_coef_name)
        match_val, disc = _compare_coef(coef, pub_val)
        discrepancies = [disc] if disc else []

    return {
        "match":           match_val,
        "tolerance":       0.005,
        "key_coef_name":   key_coef_name,
        "coef_estimate":   coef,
        "coef_published":  published_coef.get(key_coef_name) if published_coef else None,
        "se_estimate":     se,
        "tstat_estimate":  tstat,
        "pvalue_estimate": pval,
        "n_obs":           n_obs,
        "results":         result,
        "discrepancies":   discrepancies,
        "flags":           flags,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _find_papers_root() -> Path:
    here = Path(__file__).parent.parent
    papers = here / "papers"
    if papers.is_dir():
        return papers
    raise FileNotFoundError(f"papers/ directory not found under {here}")


def _run_paper(paper_id: str, papers_root: Path) -> None:
    paper_dir = papers_root / paper_id
    spec_path     = paper_dir / "spec.json"
    baseline_path = paper_dir / "baseline.parquet"

    if not spec_path.exists():
        print(f"[ERROR] spec.json not found for {paper_id}", file=sys.stderr)
        return
    if not baseline_path.exists():
        print(f"[ERROR] baseline.parquet not found for {paper_id} — run data_prep_agent first",
              file=sys.stderr)
        return

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    baseline_df = pd.read_parquet(str(baseline_path))
    published_coef: dict[str, float] = {}  # loaded from paper_info.xlsx in future

    print(f"[{paper_id}] Running baseline verification ...")
    report = verify_baseline(baseline_df, spec, published_coef)

    print(f"  match:         {report['match']}")
    print(f"  coef_estimate: {report['coef_estimate']}")
    print(f"  se_estimate:   {report['se_estimate']}")
    print(f"  n_obs:         {report['n_obs']}")
    if report["flags"]:
        for f in report["flags"]:
            print(f"  ⚠  {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Verifier")
    parser.add_argument("paper_id", nargs="?", help="Paper ID")
    parser.add_argument("--all", action="store_true", help="Run for all papers")
    args = parser.parse_args()

    papers_root = _find_papers_root()

    if args.all:
        for pd_dir in sorted(papers_root.glob("Paper_*/")):
            _run_paper(pd_dir.name, papers_root)
    elif args.paper_id:
        _run_paper(args.paper_id, papers_root)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
