"""pipeline/pipeline_audit.py
------------------------------
Batch preflight audit for all papers.

Run from repo root:
    python pipeline/pipeline_audit.py [--papers-dir papers] [--output-dir outputs]

Outputs:
    outputs/pipeline_audit.csv
    outputs/pipeline_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Re-exports from validators (keeps downstream imports stable) ───────────────
try:
    from pipeline.validators import (  # noqa: F401
        STATUS_CLASSES,
        RUNNER_MAP,
        load_spec,
        load_selection,
        load_baseline,
        get_expected_runner,
        validate_paper_folder,
        validate_spec,
        validate_data_file,
        validate_variable_selection_feasibility,
        validate_fe_structure,
        validate_fe_spec,        # alias — kept for backward compat
        classify_failure,
    )
except ModuleNotFoundError:
    from validators import (  # type: ignore[no-redef]  # noqa: F401
        STATUS_CLASSES,
        RUNNER_MAP,
        load_spec,
        load_selection,
        load_baseline,
        get_expected_runner,
        validate_paper_folder,
        validate_spec,
        validate_data_file,
        validate_variable_selection_feasibility,
        validate_fe_structure,
        validate_fe_spec,        # alias — kept for backward compat
        classify_failure,
    )


# ── Paper audit ───────────────────────────────────────────────────────────────

def audit_paper(paper_dir: Path) -> dict:
    """Run all checks for one paper and return the output row dict."""
    spec      = load_spec(paper_dir)
    selection = load_selection(paper_dir)
    df        = load_baseline(paper_dir)

    folder  = validate_paper_folder(paper_dir)
    spec_v  = validate_spec(spec)
    data    = validate_data_file(spec, paper_dir)
    varsel  = validate_variable_selection_feasibility(selection, spec, df)
    fe      = validate_fe_structure(spec, df)

    status_class = classify_failure(folder, spec_v, data, varsel, fe)

    # depvar_exists — computed inline (not in any single validator)
    depvar_exists = None
    if df is not None and spec is not None:
        depvar = spec.get("dependent_var", "")
        if not depvar:
            depvar_exists = False
        elif depvar in df.columns:
            depvar_exists = True
        else:
            depvar_exists = False

    # Sentinel values matching the pre-existing row schema
    key_vars_valid: bool | str = (
        varsel["key_vars_ok"] if varsel["has_selection"] else "no_selection"
    )
    aux_var_valid: bool | str = (
        varsel["aux_var_ok"] if varsel["has_selection"] else "no_selection"
    )

    if not fe["applicable"]:
        fe_cols_valid: bool | str | None = "not_fe"
    else:
        fe_cols_valid = fe["ok"]  # True / False / None

    # Aggregate all issue strings
    all_issues = (
        folder["issues"]
        + spec_v["issues"]
        + data["issues"]
        + varsel["issues"]
        + fe["issues"]
    )

    # Canonical column order — unchanged schema
    return {
        "paper_id":         paper_dir.name,
        "status_class":     status_class,
        "estimator":        spec_v["estimator"] or (spec or {}).get("estimator", "?"),
        "has_paper_info":   folder["has_paper_info"],
        "has_spec":         folder["has_spec"],
        "data_file_exists": data["source_data_exists"],
        "depvar_exists":    depvar_exists,
        "key_vars_valid":   key_vars_valid,
        "aux_var_valid":    aux_var_valid,
        "fe_cols_valid":    fe_cols_valid,
        "duplicate_columns": fe["duplicate_cluster_fe"],
        "expected_runner":  get_expected_runner(spec),
        "notes":            "|".join(all_issues),
    }


# ── Batch ─────────────────────────────────────────────────────────────────────

def audit_all(papers_dir: Path) -> list[dict]:
    papers = sorted(papers_dir.glob("Paper_*/"))
    return [audit_paper(p) for p in papers]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No papers found.")
        return

    cols = [
        "paper_id", "status_class", "estimator", "expected_runner",
        "has_spec", "data_file_exists", "depvar_exists",
        "key_vars_valid", "aux_var_valid", "fe_cols_valid",
        "duplicate_columns",
    ]
    widths = {
        c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
        for c in cols
    }
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)
    print(f"\nTotal: {len(rows)} papers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preflight audit for all papers")
    parser.add_argument("--papers-dir", default="papers", help="Path to papers/ directory")
    parser.add_argument("--output-dir", default="outputs", help="Path to outputs/ directory")
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    output_dir = Path(args.output_dir)

    if not papers_dir.exists():
        print(f"ERROR: papers dir not found: {papers_dir}")
        return

    rows = audit_all(papers_dir)
    _print_table(rows)

    # Write CSV
    import pandas as pd  # noqa: PLC0415

    csv_path = output_dir / "pipeline_audit.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV written: {csv_path}")

    # Write JSON
    json_path = output_dir / "pipeline_audit.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"JSON written: {json_path}")


if __name__ == "__main__":
    main()
