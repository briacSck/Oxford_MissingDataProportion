"""
pipeline/data_prep_agent.py
---------------------------
Data Prep Agent: loads the raw dataset, applies Stata-style sample restrictions from
the spec, verifies that all spec-referenced variables exist, and saves a clean
baseline DataFrame as baseline.parquet.

Usage
-----
    python pipeline/data_prep_agent.py Paper_Meyer2024
    python pipeline/data_prep_agent.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Raw loader ────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    """Load a .dta / .csv / .xlsx file into a DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    elif suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {suffix!r}")


# ── Sample restrictions ───────────────────────────────────────────────────────

_INLIST_RE  = re.compile(r'inlist\s*\(\s*(\w+)\s*,([^)]+)\)\s*$', re.I)
_INRANGE_RE = re.compile(r'inrange\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)\s*$', re.I)
_NOTMISS_RE = re.compile(r'(\w+)\s*[!~]=\s*\.\s*$', re.I)
_COMPARE_RE = re.compile(r'(\w+)\s*(>=|<=|!=|==|>|<)\s*(-?[\d.]+)\s*$')


def _parse_restriction(df: pd.DataFrame, expr: str):
    """Return a boolean Series mask for *expr*, or None if unparseable."""
    expr = expr.strip()

    m = _INLIST_RE.match(expr)
    if m:
        var, vals_str = m.group(1), m.group(2)
        raw_vals = [v.strip().strip('"').strip("'") for v in vals_str.split(",")]
        try:
            vals = [float(v) for v in raw_vals]
        except ValueError:
            vals = raw_vals
        if var in df.columns:
            return df[var].isin(vals)
        return None

    m = _INRANGE_RE.match(expr)
    if m:
        var, lo_s, hi_s = m.group(1), m.group(2).strip(), m.group(3).strip()
        if var in df.columns:
            try:
                return df[var].between(float(lo_s), float(hi_s))
            except ValueError:
                pass
        return None

    m = _NOTMISS_RE.match(expr)
    if m:
        var = m.group(1)
        if var in df.columns:
            return df[var].notna()
        return None

    m = _COMPARE_RE.match(expr)
    if m:
        var, op, val_s = m.group(1), m.group(2), m.group(3)
        if var in df.columns:
            col = df[var]
            val = float(val_s)
            ops = {
                ">=": col >= val, "<=": col <= val, "!=": col != val,
                "==": col == val, ">":  col > val,  "<":  col < val,
            }
            return ops[op]
        return None

    return None


def _apply_sample_restrictions(
    df: pd.DataFrame,
    restrictions: list,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply each Stata-style restriction string; flag unparseable ones."""
    flags: list[str] = []
    for expr in restrictions:
        if not expr or not str(expr).strip():
            continue
        expr = str(expr).strip()
        mask = _parse_restriction(df, expr)
        if mask is None:
            flags.append(
                f"sample restriction could not be translated: '{expr}' — apply manually"
            )
            continue
        before = len(df)
        df = df[mask].copy()
        after = len(df)
        flags.append(
            f"restriction '{expr}': {before} → {after} rows (dropped {before - after})"
        )
    return df, flags


# ── Column verification ───────────────────────────────────────────────────────

def _verify_columns(df: pd.DataFrame, spec: dict) -> list[str]:
    """Check spec-referenced variables exist in df; return warning strings."""
    warnings_list: list[str] = []
    vars_to_check: list[str] = []

    for key in ("dependent_var",):
        v = spec.get(key)
        if v:
            vars_to_check.append(str(v))

    for key in ("key_independent_vars", "control_vars", "fixed_effects"):
        for v in (spec.get(key) or []):
            if v:
                vars_to_check.append(str(v))

    for v in vars_to_check:
        if re.search(r'[*()\[\]]', v):
            warnings_list.append(f"wildcard '{v}' — cannot verify without expansion")
            continue
        if v not in df.columns:
            warnings_list.append(
                f"variable '{v}' in spec not found in data — check DO file"
            )
    return warnings_list


# ── Log helper ────────────────────────────────────────────────────────────────

def _append_log(paper_id: str, message: str) -> None:
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{paper_id}_log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        ts = datetime.now().isoformat(timespec="seconds")
        f.write(f"\n## [{ts}] data_prep_agent\n")
        f.write(message + "\n")


# ── Public entry ──────────────────────────────────────────────────────────────

def prepare_baseline(
    raw_data_path: str,
    spec: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Load and pre-process the raw dataset into a clean baseline DataFrame.

    Parameters
    ----------
    raw_data_path:
        Absolute path to the raw data file (.dta, .csv, or .xlsx).
        If multiple files exist, pass the primary file; others should be merged
        manually (a flag will be added if spec suggests multiple files).
    spec:
        Regression specification dict produced by ``parse_paper``.
    output_dir:
        Directory where the cleaned baseline dataset will be saved
        (e.g. ``papers/Paper_XXX/``).

    Returns
    -------
    pd.DataFrame
        Clean baseline DataFrame.
    """
    paper_id = spec.get("paper_id") or os.path.basename(output_dir.rstrip("/\\"))
    flags: list[str] = []

    # Step 1: Load raw data
    df = _load_raw(raw_data_path)
    flags.append(f"loaded {raw_data_path!r}: {df.shape[0]} rows × {df.shape[1]} cols")

    # Step 2: Apply sample restrictions
    restrictions = spec.get("sample_restrictions") or []
    df, restriction_flags = _apply_sample_restrictions(df, restrictions)
    flags.extend(restriction_flags)

    # Step 3: Verify columns (warnings only, no raise)
    col_warnings = _verify_columns(df, spec)
    flags.extend(col_warnings)

    # Step 4: Save baseline.parquet
    out_path = Path(output_dir) / "baseline.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False)
    flags.append(f"saved baseline.parquet: {df.shape[0]} rows × {df.shape[1]} cols")

    # Step 5: Log
    log_body = "\n".join(f"  {f}" for f in flags)
    _append_log(paper_id, log_body)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _find_papers_root() -> Path:
    here = Path(__file__).parent.parent
    papers = here / "papers"
    if papers.is_dir():
        return papers
    raise FileNotFoundError(f"papers/ directory not found under {here}")


def _run_paper(paper_id: str, papers_root: Path) -> None:
    paper_dir = papers_root / paper_id
    if not paper_dir.is_dir():
        print(f"[ERROR] Paper directory not found: {paper_dir}", file=sys.stderr)
        return

    # Load spec
    spec_path = paper_dir / "spec.json"
    if not spec_path.exists():
        print(f"[ERROR] spec.json not found for {paper_id} — run parser_agent first.", file=sys.stderr)
        return
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    # Load data file path from spec or paper_info.xlsx
    raw_data_path = spec.get("source_data_file") or ""
    if not raw_data_path or not Path(raw_data_path).exists():
        # Try to find in RA folder
        print(f"[WARN] source_data_file not set or not found for {paper_id}. "
              f"Set 'source_data_file' in paper_info.xlsx and re-run parser_agent.")
        return

    print(f"[{paper_id}] Loading {raw_data_path} ...")
    try:
        df = prepare_baseline(raw_data_path, spec, str(paper_dir))
        print(f"[{paper_id}] baseline.parquet written — {len(df)} rows × {len(df.columns)} cols")
    except Exception as exc:
        print(f"[{paper_id}] ERROR: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Prep Agent")
    parser.add_argument("paper_id", nargs="?", help="Paper ID (e.g. Paper_Meyer2024)")
    parser.add_argument("--all", action="store_true", help="Run for all papers")
    args = parser.parse_args()

    papers_root = _find_papers_root()

    if args.all:
        paper_dirs = sorted(papers_root.glob("Paper_*/"))
        for pd_dir in paper_dirs:
            _run_paper(pd_dir.name, papers_root)
    elif args.paper_id:
        _run_paper(args.paper_id, papers_root)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
