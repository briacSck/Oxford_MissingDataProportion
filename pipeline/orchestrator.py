"""
pipeline/orchestrator.py
-------------------------
Orchestrator: drives the full pipeline for one paper at a time, enforces three
human-approval gates, and optionally processes all papers in sequence or in
parallel.

Pipeline sequence per paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  Parser Agent          → spec dict
    [no gate]
2.  Data Prep Agent       → baseline_df
    [no gate]
3.  Baseline Verifier     → verification report
    *** HUMAN GATE 1 ***  → user confirms baseline regression matches published values
4.  Variable Selector     → selection dict (key_vars, aux_var)
    *** HUMAN GATE 2 ***  → user confirms choice of key variables
5.  Missingness Generator → dict of CSV paths
    [no gate]
6.  Listwise Agent        → nested dict of LD CSV paths
    [no gate]
7.  Regression Runner     → regression_results.xlsx
    [no gate]
8.  QC Agent              → qc_report.txt
    *** HUMAN GATE 3 ***  → user approves QC before paper is marked complete
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Custom exception ──────────────────────────────────────────────────────────

class PipelineHaltedByUser(RuntimeError):
    def __init__(self, gate: int):
        super().__init__(f"Pipeline halted by user at Gate {gate}.")
        self.gate = gate


# ── Human gate helper ─────────────────────────────────────────────────────────

def _human_gate(gate_name: str, summary: str) -> bool:
    """Print a summary and prompt the user for approval.

    Returns True if approved, False if rejected.
    """
    print(f"\n{'='*60}")
    print(f"  HUMAN GATE: {gate_name}")
    print(f"{'='*60}")
    print(summary)
    print(f"{'='*60}")
    response = input("Approve and continue? [y/N]: ").strip().lower()
    approved = response in ("y", "yes")
    if not approved:
        logger.warning("Human gate '%s' rejected by user.", gate_name)
    return approved


# ── Single-paper pipeline ─────────────────────────────────────────────────────

def run_paper(paper_dir: str) -> None:
    """Run the full pipeline for a single paper directory.

    Parameters
    ----------
    paper_dir:
        Absolute path to a paper folder, e.g. ``papers/Paper_Meyer2024/``.
        Must contain ``paper_info.xlsx`` with the required fields.

    Raises
    ------
    PipelineHaltedByUser
        If a human gate is rejected.
    RuntimeError
        If required inputs are missing.
    Exception
        Any unhandled exception from an agent is caught, logged, and re-raised
        so ``run_all`` can continue with the next paper.
    """
    paper_name = os.path.basename(paper_dir.rstrip("/\\"))
    logger.info("Starting pipeline for %s", paper_name)

    try:
        # ── Step 1: Load or parse spec ────────────────────────────────────────
        spec_path = Path(paper_dir) / "spec.json"
        if spec_path.exists():
            import json
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        else:
            from pipeline.parser_agent import parse_paper
            papers_root = Path(paper_dir).parent
            spec = parse_paper(paper_name, str(papers_root))

        logger.info("[%s] Parser complete: estimator=%s", paper_name, spec.get("estimator"))
        if spec.get("manual_review_required") or spec.get("parse_confidence") != "high":
            raise RuntimeError(
                f"[{paper_name}] Parser confidence={spec.get('parse_confidence')!r}; "
                f"manual_review_required={spec.get('manual_review_required')}. "
                f"Flags: {spec.get('flags', [])}. "
                "Edit paper_info.xlsx and re-run before proceeding."
            )

        # ── Step 2: Prepare baseline dataset ─────────────────────────────────
        from pipeline.data_prep_agent import prepare_baseline
        baseline_path = Path(paper_dir) / "baseline.parquet"
        raw_data_path = spec.get("source_data_file")

        if not baseline_path.exists():
            if not raw_data_path or not Path(raw_data_path).exists():
                raise RuntimeError(
                    f"[{paper_name}] No source_data_file in spec and no baseline.parquet"
                )
            baseline_df = prepare_baseline(raw_data_path, spec, paper_dir)
        else:
            baseline_df = pd.read_parquet(str(baseline_path))

        logger.info("[%s] Data prep complete: N=%d", paper_name, len(baseline_df))

        # ── Step 3: Verify baseline + HUMAN GATE 1 ───────────────────────────
        from pipeline.baseline_verifier import verify_baseline
        published_coef: dict = {}
        verification = verify_baseline(baseline_df, spec, published_coef)
        gate1_summary = (
            f"Paper: {paper_name}\n"
            f"Baseline match: {verification['match']}\n"
            f"Discrepancies: {verification.get('discrepancies', [])}\n"
        )
        if not _human_gate("Baseline Verification", gate1_summary):
            raise PipelineHaltedByUser(gate=1)

        # ── Step 4: Select variables + HUMAN GATE 2 ──────────────────────────
        from pipeline.variable_selector import select_variables
        papers_root = Path(paper_dir).parent
        selection = select_variables(
            paper_name, str(papers_root), spec=spec, data=baseline_df
        )
        gate2_summary = (
            f"Paper: {paper_name}\n"
            f"Key vars:  {selection.get('key_vars')}\n"
            f"Aux var:   {selection.get('aux_var')!r}\n"
            f"Confidence: {selection.get('selection_confidence')}\n"
            f"Flags:     {selection.get('flags', [])}\n"
        )
        if not _human_gate("Variable Selection — final confirm", gate2_summary):
            raise PipelineHaltedByUser(gate=2)

        # ── Step 5: Generate MAR datasets ────────────────────────────────────
        from pipeline.missingness_generator import generate_missingness
        mar_paths = generate_missingness(
            str(baseline_path),
            selection["key_vars"],
            selection["aux_var"],
            paper_dir,
        )
        logger.info("[%s] MAR datasets generated: %d vars", paper_name, len(mar_paths))

        # ── Step 6: Apply listwise deletion ──────────────────────────────────
        from pipeline.listwise_agent import apply_listwise
        ld_map = apply_listwise(paper_dir)
        total_ld = sum(len(v) for v in ld_map.values())
        logger.info("[%s] Listwise deletion complete: %d files", paper_name, total_ld)

        # ── Step 7: Run regressions ───────────────────────────────────────────
        from pipeline.regression_runner import run_all_regressions
        results_xlsx = run_all_regressions(paper_dir, spec)
        logger.info("[%s] Regressions complete: %s", paper_name, results_xlsx)

        # ── Step 8: QC + HUMAN GATE 3 ────────────────────────────────────────
        from pipeline.qc_agent import run_qc
        qc_passed = run_qc(paper_dir)
        qc_report_text = (Path(paper_dir) / "qc_report.txt").read_text(encoding="utf-8")
        if not _human_gate("QC Review", qc_report_text):
            raise PipelineHaltedByUser(gate=3)

        logger.info("[%s] Pipeline complete.", paper_name)

    except PipelineHaltedByUser:
        raise
    except RuntimeError as exc:
        logger.error("[%s] Pipeline halted: %s", paper_name, exc)
        raise
    except Exception as exc:
        logger.exception("[%s] Unexpected error in pipeline: %s", paper_name, exc)
        raise


# ── Multi-paper runner ────────────────────────────────────────────────────────

def run_all(papers_dir: str, parallel: bool = False, skip_gates: bool = False) -> None:
    """Run the pipeline for all paper folders in papers_dir.

    Parameters
    ----------
    papers_dir:
        Absolute path to the ``papers/`` directory containing ``Paper_XXX/``
        subdirectories.
    parallel:
        If True, run papers concurrently. Requires ``skip_gates=True``.
    skip_gates:
        Must be True when ``parallel=True`` (gates require stdin).

    Raises
    ------
    ValueError
        If ``parallel=True`` and ``skip_gates=False``.
    """
    papers = sorted(Path(papers_dir).glob("Paper_*/"))
    if not papers:
        print("No paper directories found.")
        return

    if parallel and not skip_gates:
        raise ValueError(
            "parallel=True requires skip_gates=True (gates need stdin). "
            "Pass skip_gates=True explicitly to acknowledge gates will be skipped."
        )

    results: dict[str, str] = {}

    if parallel:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(run_paper, str(p)): p.name for p in papers}
            for f in concurrent.futures.as_completed(futures):
                name = futures[f]
                try:
                    f.result()
                    results[name] = "OK"
                except Exception as e:
                    results[name] = f"FAILED: {e}"
    else:
        for p in papers:
            try:
                run_paper(str(p))
                results[p.name] = "OK"
            except Exception as e:
                logger.error("[%s] %s", p.name, e)
                results[p.name] = f"FAILED: {e}"

    ok = [k for k, v in results.items() if v == "OK"]
    fail = [k for k, v in results.items() if v != "OK"]
    print(f"\nSummary: {len(ok)} succeeded, {len(fail)} failed")
    for name in fail:
        print(f"  FAILED: {name} — {results[name]}")
