"""pipeline/pipeline_audit.py
------------------------------
Diagnostic script: prints per-paper precondition status table.

Run from repo root:
    python pipeline/pipeline_audit.py
    python pipeline/pipeline_audit.py --papers-dir papers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_size(path: Path) -> str:
    mb = path.stat().st_size / (1024 * 1024)
    return f"OK({mb:.1f}MB)"


def _check_paper(paper_dir: Path) -> dict:
    result: dict = {"paper": paper_dir.name}

    # 1. spec.json
    spec_path = paper_dir / "spec.json"
    if not spec_path.exists():
        result.update({"conf": "MISSING", "manual_review": "?", "data_file": "NO SPEC"})
        return result

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    result["conf"] = spec.get("parse_confidence", "?")
    result["manual_review"] = str(spec.get("manual_review_required", "?"))

    # 2. source_data_file
    src = spec.get("source_data_file")
    if not src:
        result["data_file"] = "None"
    else:
        p = Path(src)
        if not p.exists():
            result["data_file"] = f"MISSING({p.name})"
        else:
            result["data_file"] = _fmt_size(p)

    # 3. spec_cols_in_data — light check (just confirm depvar + key_vars present)
    spec_cols_ok = "?"
    if src and Path(src).exists():
        try:
            p = Path(src)
            suffix = p.suffix.lower()
            if suffix == ".dta":
                import pandas as pd
                df = pd.read_stata(str(p), iterator=True)
                col_set = set(df.varlist)
            elif suffix == ".csv":
                import pandas as pd
                df5 = pd.read_csv(str(p), nrows=5)
                col_set = set(df5.columns)
            elif suffix in (".xlsx", ".xls"):
                import pandas as pd
                df5 = pd.read_excel(str(p), nrows=5)
                col_set = set(df5.columns)
            else:
                col_set = set()

            dep = spec.get("dependent_var", "")
            keys = spec.get("key_independent_vars") or []
            missing_cols = [c for c in ([dep] + keys) if c and c not in col_set]
            spec_cols_ok = "OK" if not missing_cols else f"MISSING_COLS:{missing_cols}"
        except Exception as exc:
            spec_cols_ok = f"ERR({exc!s:.40})"
    result["spec_cols"] = spec_cols_ok

    # 4–10. Artifacts
    result["baseline"] = "OK" if (paper_dir / "baseline.parquet").exists() else "MISSING"
    result["selection"] = "OK" if (paper_dir / "selection.json").exists() else "MISSING"

    missing_csvs = list(paper_dir.glob("missing/*.csv"))
    result["missing"] = str(len(missing_csvs))

    listwise_csvs = list(paper_dir.glob("listwise/*.csv"))
    result["listwise"] = str(len(listwise_csvs))

    result["results"] = "OK" if (paper_dir / "regression_results.xlsx").exists() else "MISSING"
    result["qc"] = "OK" if (paper_dir / "qc_report.txt").exists() else "MISSING"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pipeline preconditions per paper")
    parser.add_argument("--papers-dir", default="papers", help="Path to papers/ directory")
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    if not papers_dir.exists():
        print(f"ERROR: papers dir not found: {papers_dir}")
        return

    papers = sorted(papers_dir.glob("Paper_*/"))
    if not papers:
        print("No Paper_* directories found.")
        return

    rows = [_check_paper(p) for p in papers]

    # Column widths
    cols = ["paper", "conf", "manual_review", "data_file", "spec_cols",
            "baseline", "selection", "missing", "listwise", "results", "qc"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)

    print(f"\nTotal: {len(rows)} papers")


if __name__ == "__main__":
    main()
