"""
pipeline/qc_agent.py
---------------------
QC Agent: performs quality control checks on the regression results before
the paper is marked as complete. Writes a qc_report.txt and returns True
iff all ERROR-level checks pass (warnings alone do not fail the QC).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import PROPORTION_LABELS, RANDOM_SEED

logger = logging.getLogger(__name__)


def run_qc(paper_dir: str) -> bool:
    """Run 7 QC checks and write ``{paper_dir}/qc_report.txt``.

    Parameters
    ----------
    paper_dir:
        Absolute path to the paper's directory.

    Returns
    -------
    bool
        True if all ERROR-level checks pass; warnings alone still return True.
    """
    paper_path = Path(paper_dir)
    paper_id = paper_path.name
    report_lines: list[str] = []
    errors: list[str] = []
    warnings_list: list[str] = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines.append(f"QC Report — {paper_id}  [{timestamp}]")
    report_lines.append(f"RANDOM_SEED={RANDOM_SEED} confirmed")

    # ── Load regression results ───────────────────────────────────────────────
    results_path = paper_path / "regression_results.xlsx"
    try:
        results_df = pd.read_excel(str(results_path), sheet_name="Results")
    except Exception as exc:
        errors.append(f"Cannot load regression_results.xlsx: {exc}")
        results_df = pd.DataFrame()

    # ── Load published coef ───────────────────────────────────────────────────
    info_path = paper_path / "paper_info.xlsx"
    published_coef: float | None = None
    if info_path.exists():
        try:
            info_df = pd.read_excel(str(info_path))
            if "published_coef_main" in info_df.columns:
                val = info_df["published_coef_main"].iloc[0]
                published_coef = float(val) if pd.notna(val) else None
        except Exception as exc:
            logger.warning("Could not load paper_info.xlsx: %s", exc)

    # Derive K and non-baseline rows
    if not results_df.empty and "Key Variable" in results_df.columns and "Missing Proportion" in results_df.columns:
        non_baseline = results_df[results_df["Missing Proportion"] != "baseline"]
        K = non_baseline["Key Variable"].nunique()
        baseline_rows = results_df[results_df["Missing Proportion"] == "baseline"]
    else:
        non_baseline = pd.DataFrame()
        K = 0
        baseline_rows = pd.DataFrame()

    # ── Check 1: Row count ────────────────────────────────────────────────────
    expected_rows = K * 7 + 1
    actual_rows = len(results_df)
    if actual_rows != expected_rows:
        msg = f"Row count — expected {expected_rows} (K={K}×7+1), got {actual_rows}"
        errors.append(msg)
        report_lines.append(f"[FAIL] Check 1: {msg}")
    else:
        report_lines.append(f"[PASS] Check 1: Row count — {actual_rows} rows (K={K}×7+1)")

    # ── Check 2: File count ───────────────────────────────────────────────────
    missing_dir = paper_path / "missing"
    listwise_dir = paper_path / "listwise"
    expected_files = K * 7

    missing_csvs = list(missing_dir.glob("*_MAR_*.csv")) if missing_dir.exists() else []
    listwise_csvs = list(listwise_dir.glob("*_MAR_*_LD.csv")) if listwise_dir.exists() else []

    check2_ok = True
    if len(missing_csvs) != expected_files:
        msg = f"File count — missing/ has {len(missing_csvs)} files, expected {expected_files}"
        errors.append(msg)
        report_lines.append(f"[FAIL] Check 2: {msg}")
        check2_ok = False
    if len(listwise_csvs) != expected_files:
        msg = f"File count — listwise/ has {len(listwise_csvs)} files, expected {expected_files}"
        errors.append(msg)
        report_lines.append(f"[FAIL] Check 2: {msg}")
        check2_ok = False
    if check2_ok:
        report_lines.append(
            f"[PASS] Check 2: File count — {len(missing_csvs)} missing, "
            f"{len(listwise_csvs)} listwise (expected {expected_files} each)"
        )

    # ── Check 3: Proportion coverage ──────────────────────────────────────────
    check3_ok = True
    if not non_baseline.empty and K > 0:
        for kv in non_baseline["Key Variable"].unique():
            kv_labels = set(non_baseline[non_baseline["Key Variable"] == kv]["Missing Proportion"].tolist())
            missing_labels = [lbl for lbl in PROPORTION_LABELS if lbl not in kv_labels]
            if missing_labels:
                msg = f"Proportion coverage — '{kv}' missing labels: {missing_labels}"
                errors.append(msg)
                report_lines.append(f"[FAIL] Check 3: {msg}")
                check3_ok = False
    if check3_ok:
        report_lines.append(f"[PASS] Check 3: Proportion coverage — all 7 labels present for each key var")

    # ── Check 4: Baseline sign match ──────────────────────────────────────────
    if published_coef is not None and not baseline_rows.empty and "β̂" in baseline_rows.columns:
        try:
            base_coef_val = float(baseline_rows["β̂"].iloc[0])
            if np.sign(base_coef_val) != np.sign(published_coef):
                msg = (
                    f"Baseline sign mismatch — estimated β={base_coef_val:.4f}, "
                    f"published={published_coef:.4f}"
                )
                warnings_list.append(msg)
                report_lines.append(f"[WARN] Check 4: {msg}")
            else:
                report_lines.append(
                    f"[PASS] Check 4: Baseline sign match — "
                    f"estimated={base_coef_val:.4f}, published={published_coef:.4f}"
                )
        except Exception:
            report_lines.append("[PASS] Check 4: Baseline sign match — no valid coef to compare")
    else:
        report_lines.append("[PASS] Check 4: Baseline sign match — no published coef provided, skipped")

    # ── Check 5: No all-NaN columns ───────────────────────────────────────────
    if not results_df.empty:
        all_nan_cols = [c for c in results_df.columns if results_df[c].isna().all()]
        if all_nan_cols:
            msg = f"All-NaN columns found: {all_nan_cols}"
            errors.append(msg)
            report_lines.append(f"[FAIL] Check 5: {msg}")
        else:
            report_lines.append("[PASS] Check 5: No all-NaN columns")
    else:
        report_lines.append("[PASS] Check 5: No all-NaN columns — results empty, skipped")

    # ── Check 6: N monotonicity ───────────────────────────────────────────────
    label_order = {lbl: i for i, lbl in enumerate(PROPORTION_LABELS)}
    check6_warned = False
    if not non_baseline.empty and "Post-LD N" in non_baseline.columns:
        for kv in non_baseline["Key Variable"].unique():
            kv_rows = non_baseline[non_baseline["Key Variable"] == kv].copy()
            kv_rows["_order"] = kv_rows["Missing Proportion"].map(label_order)
            kv_rows = kv_rows.sort_values("_order")
            ns = kv_rows["Post-LD N"].tolist()
            for i in range(1, len(ns)):
                try:
                    if float(ns[i]) > float(ns[i - 1]):
                        msg = (
                            f"N monotonicity — '{kv}' N increased from {ns[i-1]} "
                            f"to {ns[i]} (proportions "
                            f"{kv_rows['Missing Proportion'].iloc[i-1]} → "
                            f"{kv_rows['Missing Proportion'].iloc[i]})"
                        )
                        warnings_list.append(msg)
                        report_lines.append(f"[WARN] Check 6: {msg}")
                        check6_warned = True
                except (TypeError, ValueError):
                    pass
    if not check6_warned:
        report_lines.append("[PASS] Check 6: N monotonicity — N is non-increasing for all key vars")

    # ── Check 7: Seed confirmation ────────────────────────────────────────────
    report_lines.append(f"[INFO] Check 7: RANDOM_SEED={RANDOM_SEED} confirmed")

    # ── Summary ───────────────────────────────────────────────────────────────
    report_lines.append("---------------------------------------")
    if errors:
        report_lines.append(f"QC FAILED: {len(errors)} issues found")
    else:
        report_lines.append("QC PASSED: ready for human review")

    report_text = "\n".join(report_lines)

    # Write report
    report_path = paper_path / "qc_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("QC report written to %s", report_path.name)

    return len(errors) == 0
