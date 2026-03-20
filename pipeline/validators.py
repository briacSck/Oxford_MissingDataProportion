"""pipeline/validators.py
------------------------
Reusable, structured validation helpers for the pipeline preflight audit.

Every public ``validate_*`` function returns a ``dict`` with at minimum:
    ok      : bool | None  – True iff all checks pass; None when undeterminable
    issues  : list[str]    – human-readable description of every failed check

Loaders, constants, and ``classify_failure`` / ``get_expected_runner`` are also
exported here so that ``pipeline_audit.py`` (and future consumers) only need to
import from this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

STATUS_CLASSES: set[str] = {
    "NEEDS_SPEC_FIX",
    "MANUAL_SPEC",
    "UNSUPPORTED_METHOD",
    "NEEDS_DATA_FIX",
    "FE_STRUCTURE_BUG",
    "SELECTION_INVALID",
    "READY",
}

# Maps spec estimator value → expected_runner label
RUNNER_MAP: dict[str, str] = {
    "OLS": "OLS",
    "Logit": "OLS",
    "Probit": "OLS",
    "IV": "OLS",
    "GLS": "OLS",
    "HLM": "OLS",
    "FE": "FE",
    "SUR": "SUR",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_spec(paper_dir: Path) -> Optional[dict]:
    path = paper_dir / "spec.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_selection(paper_dir: Path) -> Optional[dict]:
    path = paper_dir / "selection.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_baseline(paper_dir: Path):
    """Return pd.DataFrame or None if baseline.parquet is absent."""
    path = paper_dir / "baseline.parquet"
    if not path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_parquet(str(path))
    except Exception:
        return None


# ── Derived helpers ───────────────────────────────────────────────────────────

def get_expected_runner(spec: Optional[dict]) -> str:
    if spec is None:
        return "UNKNOWN"
    estimator = spec.get("estimator", "")
    return RUNNER_MAP.get(estimator, "UNKNOWN")


# ── Validators ────────────────────────────────────────────────────────────────

def validate_paper_folder(paper_dir: Path) -> dict:
    """Check that required and optional files exist in a paper directory.

    ``ok`` is False only when ``paper_info.xlsx`` or ``spec.json`` is missing.
    ``selection.json`` and ``baseline.parquet`` are informational only.
    """
    has_paper_info = (paper_dir / "paper_info.xlsx").exists()
    has_spec       = (paper_dir / "spec.json").exists()
    has_selection  = (paper_dir / "selection.json").exists()
    has_baseline   = (paper_dir / "baseline.parquet").exists()

    issues: list[str] = []
    if not has_paper_info:
        issues.append("paper_info.xlsx missing")
    if not has_spec:
        issues.append("spec.json missing")

    return {
        "ok":             has_paper_info and has_spec,
        "has_paper_info": has_paper_info,
        "has_spec":       has_spec,
        "has_selection":  has_selection,
        "has_baseline":   has_baseline,
        "issues":         issues,
    }


def validate_spec(spec: Optional[dict]) -> dict:
    """Validate the parsed spec against the CONTEXT.md §9 contract.

    Checks that the spec exists, uses a supported replication code type,
    has an acceptable parse confidence, and has the required fields set.
    """
    issues: list[str] = []

    if spec is None:
        return {
            "ok":                   False,
            "estimator":            "",
            "replication_code_type": "",
            "parse_confidence":     "",
            "is_r":                 False,
            "is_none_type":         False,
            "is_low_confidence":    False,
            "issues":               ["spec is None"],
        }

    replication_code_type = spec.get("replication_code_type", "")
    parse_confidence      = spec.get("parse_confidence", "")
    estimator             = spec.get("estimator", "")
    dependent_var         = spec.get("dependent_var", "")

    _SUPPORTED_CODE_TYPES = {"stata"}
    is_r = replication_code_type == "r"   # kept for backward compat
    is_unsupported_code = (
        bool(replication_code_type)
        and replication_code_type not in _SUPPORTED_CODE_TYPES
        and replication_code_type != "none"
    )
    is_none_type      = replication_code_type == "none"
    is_low_confidence = parse_confidence == "low"

    if is_unsupported_code:
        issues.append(f"replication_code_type is {replication_code_type!r} (unsupported)")
    if is_none_type:
        issues.append("replication_code_type is 'none' (manual spec required)")
    if is_low_confidence:
        issues.append("parse_confidence is 'low' (unreliable parse)")
    if not estimator:
        issues.append("estimator is empty")
    if not dependent_var:
        issues.append("dependent_var is empty")

    return {
        "ok":                   len(issues) == 0,
        "estimator":            estimator,
        "replication_code_type": replication_code_type,
        "parse_confidence":     parse_confidence,
        "is_r":                 is_r,
        "is_unsupported_code":  is_unsupported_code,
        "is_none_type":         is_none_type,
        "is_low_confidence":    is_low_confidence,
        "issues":               issues,
    }


def validate_data_file(spec: Optional[dict], paper_dir: Path) -> dict:
    """Check that the source data file and baseline.parquet are present and valid.

    Reads only column metadata from the parquet (no full load).
    """
    issues: list[str] = []

    # Source data file
    source_data_exists = False
    if spec is None:
        issues.append("no spec")
    else:
        src = spec.get("source_data_file")
        if not src:
            issues.append("source_data_file not set in spec")
        else:
            source_data_exists = Path(src).exists()
            if not source_data_exists:
                issues.append(f"source_data_file not found: {Path(src).name}")

    # Baseline parquet
    baseline_path  = paper_dir / "baseline.parquet"
    baseline_exists = baseline_path.exists()
    if not baseline_exists:
        issues.append("baseline.parquet missing (will be generated from source)")

    # Duplicate column check (schema-only read — no full load)
    duplicate_baseline_cols = False
    if baseline_exists:
        try:
            import pyarrow.parquet as pq
            col_names = pq.read_schema(str(baseline_path)).names
            duplicate_baseline_cols = len(col_names) != len(set(col_names))
            if duplicate_baseline_cols:
                issues.append("baseline.parquet has duplicate column names")
        except Exception:
            pass

    # ok=False only when BOTH source and baseline are missing — run_paper auto-generates
    # baseline from source, so a missing baseline alone is not a blocker.
    ok = (source_data_exists or baseline_exists) and not duplicate_baseline_cols

    return {
        "ok":                    ok,
        "source_data_exists":    source_data_exists,
        "baseline_exists":       baseline_exists,
        "duplicate_baseline_cols": duplicate_baseline_cols,
        "issues":                issues,
    }


def validate_variable_selection_feasibility(
    selection: Optional[dict],
    spec: Optional[dict],
    df,
) -> dict:
    """Enforce the CONTEXT.md §5 contract for key_vars and aux_var.

    Returns ``ok=None`` (and ``has_selection=False``) when no selection.json.
    """
    if selection is None:
        return {
            "ok":           None,
            "has_selection": False,
            "key_vars_ok":  None,
            "aux_var_ok":   None,
            "issues":       [],
        }

    try:
        from pipeline.config import MIN_KEY_VARS, MAX_KEY_VARS  # noqa: PLC0415
    except ModuleNotFoundError:
        from config import MIN_KEY_VARS, MAX_KEY_VARS  # noqa: PLC0415

    key_vars: list[str]     = selection.get("key_vars") or []
    aux_var: str            = selection.get("aux_var", "") or ""
    depvar: str             = (spec or {}).get("dependent_var", "")
    fixed_effects: list[str] = (spec or {}).get("fixed_effects") or []

    key_issues: list[str] = []

    # Count constraint
    if not (MIN_KEY_VARS <= len(key_vars) <= MAX_KEY_VARS):
        key_issues.append(
            f"key_vars count {len(key_vars)} not in [{MIN_KEY_VARS}, {MAX_KEY_VARS}]"
        )

    for v in key_vars:
        if depvar and v == depvar:
            key_issues.append(f"key_var '{v}' is the dependent_var")
        if v in fixed_effects:
            key_issues.append(f"key_var '{v}' is in fixed_effects")
        if df is not None:
            if v not in df.columns:
                key_issues.append(f"key_var '{v}' not in baseline columns")
            else:
                unique_vals = set(df[v].dropna().unique())
                if unique_vals <= {0, 1} and len(unique_vals) <= 2:
                    key_issues.append(f"key_var '{v}' appears binary")

    key_vars_ok = len(key_issues) == 0

    # Aux var checks
    aux_issues: list[str] = []

    if not aux_var or not aux_var.strip():
        aux_issues.append("aux_var is empty")
    else:
        if depvar and aux_var == depvar:
            aux_issues.append(f"aux_var '{aux_var}' is the dependent_var")
        if aux_var in key_vars:
            aux_issues.append(f"aux_var '{aux_var}' is also a key_var")
        if df is not None:
            if aux_var not in df.columns:
                aux_issues.append(f"aux_var '{aux_var}' not in baseline columns")
            else:
                unique_vals = set(df[aux_var].dropna().unique())
                if unique_vals <= {0, 1} and len(unique_vals) <= 2:
                    aux_issues.append(f"aux_var '{aux_var}' appears binary")

    aux_var_ok = len(aux_issues) == 0
    all_issues = key_issues + aux_issues

    return {
        "ok":           len(all_issues) == 0,
        "has_selection": True,
        "key_vars_ok":  key_vars_ok,
        "aux_var_ok":   aux_var_ok,
        "issues":       all_issues,
    }


def validate_fe_structure(spec: Optional[dict], df) -> dict:
    """Validate fixed-effects columns against the baseline dataframe.

    Short-circuits with ``applicable=False`` when estimator != 'FE'.
    Returns ``ok=None`` when df is unavailable.
    ``ok`` reflects only missing FE columns; duplicate_cluster_fe is informational only.
    """
    if spec is None or spec.get("estimator") != "FE":
        return {
            "ok":                  True,
            "applicable":          False,
            "fe_cols_missing":     [],
            "fe_cols_expanding":   [],
            "duplicate_cluster_fe": False,
            "issues":              [],
        }

    fixed_effects: list[str] = spec.get("fixed_effects") or []
    cluster_var: str         = spec.get("cluster_var", "") or ""

    # Duplicate cluster/FE check (doesn't require df)
    duplicate_cluster_fe = bool(cluster_var and cluster_var in fixed_effects)
    dup_issues = (
        [f"cluster_var '{cluster_var}' duplicated in fixed_effects"]
        if duplicate_cluster_fe else []
    )

    if df is None:
        return {
            "ok":                  None,
            "applicable":          True,
            "fe_cols_missing":     [],
            "fe_cols_expanding":   [],
            "duplicate_cluster_fe": duplicate_cluster_fe,
            "issues":              dup_issues,
        }

    fe_cols_missing:   list[str] = []
    fe_cols_expanding: list[str] = []
    col_issues:        list[str] = []

    for col in fixed_effects:
        if col not in df.columns:
            fe_cols_missing.append(col)
            col_issues.append(f"fixed_effect col '{col}' not in baseline")
        else:
            # Warn when dtype would expand to multiple dummies (informational only)
            dtype = df[col].dtype
            if hasattr(dtype, "name") and dtype.name in ("object", "category", "string"):
                fe_cols_expanding.append(col)
            elif str(dtype) in ("object", "category"):
                fe_cols_expanding.append(col)

    # Only missing FE columns are blockers; cluster==FE is a valid real-paper pattern
    ok = len(fe_cols_missing) == 0
    issues = col_issues + dup_issues   # dup_issues stay as informational warnings

    return {
        "ok":                  ok,
        "applicable":          True,
        "fe_cols_missing":     fe_cols_missing,
        "fe_cols_expanding":   fe_cols_expanding,
        "duplicate_cluster_fe": duplicate_cluster_fe,
        "issues":              issues,
    }


# Backward-compat alias
validate_fe_spec = validate_fe_structure


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_failure(
    folder: dict,
    spec_v: dict,
    data: dict,
    selection: dict,
    fe: dict,
) -> str:
    """Priority-ordered status classification from the five validator result dicts."""
    # True blocker: missing required files
    if not folder["has_paper_info"] or not folder["has_spec"]:
        return "NEEDS_SPEC_FIX"

    # True blocker: unsupported code type (R, Julia, etc.)
    if spec_v.get("is_unsupported_code"):
        return "UNSUPPORTED_METHOD"

    # Human-intervention required: no estimator/depvar can be determined
    if spec_v["is_none_type"] or spec_v["is_low_confidence"]:
        return "MANUAL_SPEC"

    # True blocker: no runnable data at all (ok=False only when BOTH source and
    # baseline are missing — baseline alone is auto-generated by run_paper)
    if not data["ok"] and not data["source_data_exists"] and not data["baseline_exists"]:
        return "NEEDS_DATA_FIX"

    # True blocker: FE columns missing from data (only when df was available)
    if fe["applicable"] and fe["ok"] is False:
        return "FE_STRUCTURE_BUG"

    # NOT a blocker: stale selection.json — run_paper step 4 always reruns
    # select_variables fresh and overwrites selection.json

    return "READY"
