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
5.  Missingness Generator → dict of parquet paths
    [no gate]
6.  Listwise Agent        → list of listwise parquet paths
    [no gate]
7.  Regression Runner     → results_df
    [no gate]
8.  QC Agent              → qc_report
    *** HUMAN GATE 3 ***  → user approves QC before paper is marked complete
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


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
    NotImplementedError
        Until the agent stubs are implemented.
    RuntimeError
        If a human gate is rejected (pipeline halted for this paper).
    Exception
        Any unhandled exception from an agent is caught, logged, and re-raised
        so ``run_all`` can continue with the next paper.
    """
    paper_name = os.path.basename(paper_dir.rstrip("/\\"))
    logger.info("Starting pipeline for %s", paper_name)

    try:
        # ── Step 1: Parse DO file ─────────────────────────────────────────────
        from pipeline.parser_agent import parse_do_file
        # TODO: Read do_path and pdf_path from paper_info.xlsx.
        do_path = None   # TODO: load from paper_info.xlsx
        pdf_path = None  # TODO: load from paper_info.xlsx (optional)
        spec = parse_do_file(do_path, pdf_path)
        logger.info("[%s] Parser complete: estimator=%s", paper_name, spec.get("estimator"))

        # ── Step 2: Prepare baseline dataset ─────────────────────────────────
        from pipeline.data_prep_agent import prepare_baseline
        # TODO: Read raw_data_path from paper_info.xlsx.
        raw_data_path = None  # TODO: load from paper_info.xlsx
        baseline_df = prepare_baseline(raw_data_path, spec, paper_dir)
        logger.info("[%s] Data prep complete: N=%d", paper_name, len(baseline_df))

        # ── Step 3: Verify baseline + HUMAN GATE 1 ───────────────────────────
        from pipeline.baseline_verifier import verify_baseline
        # TODO: Read published_coef from paper_info.xlsx.
        published_coef = {}  # TODO: load from paper_info.xlsx
        verification = verify_baseline(baseline_df, spec, published_coef)
        gate1_summary = (
            f"Paper: {paper_name}\n"
            f"Baseline match: {verification['match']}\n"
            f"Discrepancies: {verification.get('discrepancies', [])}\n"
        )
        if not _human_gate("Baseline Verification", gate1_summary):
            raise RuntimeError(f"[{paper_name}] Pipeline halted at Gate 1.")

        # ── Step 4: Select variables + HUMAN GATE 2 ──────────────────────────
        from pipeline.variable_selector import select_variables
        selection = select_variables(baseline_df, spec)
        gate2_summary = (
            f"Paper: {paper_name}\n"
            f"Key vars:  {selection['key_vars']}\n"
            f"Aux var:   {selection['aux_var']}\n"
            f"Rationale: {selection['rationale']}\n"
        )
        if not _human_gate("Variable Selection", gate2_summary):
            raise RuntimeError(f"[{paper_name}] Pipeline halted at Gate 2.")

        # ── Step 5: Generate MAR datasets ────────────────────────────────────
        from pipeline.missingness_generator import generate_mar_datasets
        missing_dir = os.path.join(paper_dir, "missing")
        mar_paths = generate_mar_datasets(
            baseline_df,
            selection["key_vars"],
            selection["aux_var"],
            missing_dir,
        )
        logger.info("[%s] MAR datasets generated: %d files", paper_name, len(mar_paths))

        # ── Step 6: Apply listwise deletion ──────────────────────────────────
        from pipeline.listwise_agent import apply_listwise_deletion
        listwise_dir = os.path.join(paper_dir, "listwise")
        ld_paths = apply_listwise_deletion(missing_dir, listwise_dir)
        logger.info("[%s] Listwise deletion complete: %d files", paper_name, len(ld_paths))

        # ── Step 7: Run regressions ───────────────────────────────────────────
        from pipeline.regression_runner import run_regressions
        # TODO: Build baseline_results from verification results.
        baseline_results = verification.get("results")
        results_df = run_regressions(listwise_dir, spec, baseline_results)
        logger.info("[%s] Regressions complete: %d rows", paper_name, len(results_df))

        # ── Step 8: QC + HUMAN GATE 3 ────────────────────────────────────────
        from pipeline.qc_agent import run_qc
        qc_report = run_qc(paper_dir, spec, results_df)
        gate3_summary = (
            f"Paper: {paper_name}\n"
            f"QC passed:  {qc_report['passed']}\n"
            f"Warnings:   {qc_report.get('warnings', [])}\n"
            f"Errors:     {qc_report.get('errors', [])}\n"
        )
        if not _human_gate("QC Review", gate3_summary):
            raise RuntimeError(f"[{paper_name}] Pipeline halted at Gate 3.")

        logger.info("[%s] Pipeline complete.", paper_name)

    except NotImplementedError:
        logger.error("[%s] Pipeline failed: agent stub not yet implemented.", paper_name)
        raise
    except RuntimeError as exc:
        logger.error("[%s] Pipeline halted: %s", paper_name, exc)
        raise
    except Exception as exc:
        logger.exception("[%s] Unexpected error in pipeline: %s", paper_name, exc)
        raise


# ── Multi-paper runner ────────────────────────────────────────────────────────

def run_all(papers_dir: str, parallel: bool = False) -> None:
    """Run the pipeline for all paper folders in papers_dir.

    Parameters
    ----------
    papers_dir:
        Absolute path to the ``papers/`` directory containing ``Paper_XXX/``
        subdirectories.
    parallel:
        If True, run papers concurrently using ``concurrent.futures.ThreadPoolExecutor``.
        Default False (sequential) — recommended until all agents are implemented
        and baseline verification is stable.

    Notes
    -----
    Parallel mode disables interactive human gates (they require stdin).
    Only use parallel=True after all three gates have been pre-approved for
    every paper (e.g. in a fully automated re-run).
    """
    # TODO: Glob all Paper_*/ subdirectories inside papers_dir.
    # TODO: Sort them for deterministic ordering.
    # TODO: If parallel=False:
    #         for each paper_dir, call run_paper(paper_dir) inside try/except;
    #         log success or failure; continue to next paper on failure.
    # TODO: If parallel=True:
    #         use concurrent.futures.ThreadPoolExecutor;
    #         warn user that human gates are disabled in parallel mode;
    #         collect futures and log results.
    # TODO: At the end, print a summary: N succeeded, N failed, list failures.
    raise NotImplementedError("run_all is not yet implemented.")
