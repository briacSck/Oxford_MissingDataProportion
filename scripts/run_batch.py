"""
scripts/run_batch.py
--------------------
Safe batch runner for the MAR simulation pipeline.

Reads outputs/pipeline_audit.csv, runs only READY papers via
pipeline.orchestrator.run_paper, captures every outcome (including gate halts
and module-specific bugs), clusters failures by traceback signature, and
persists results to:
  - outputs/batch_run_results.csv
  - outputs/batch_failure_clusters.json

Usage examples
--------------
# Dry run — list READY papers without executing
python3 scripts/run_batch.py --dry-run

# Run all READY papers
python3 scripts/run_batch.py

# Rerun only papers that previously raised UNKNOWN_EXCEPTION
python3 scripts/run_batch.py --filter-class UNKNOWN_EXCEPTION

# Rerun only papers that hit gate 1
python3 scripts/run_batch.py --filter-class HALT_GATE_1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

# ── Ensure repo root is importable ───────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.orchestrator import run_paper, PipelineHaltedByUser  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_batch")

# ── Constants ─────────────────────────────────────────────────────────────────
OUTCOME_CLASSES = {
    "SUCCESS",
    "HALT_GATE_1",
    "HALT_GATE_2",
    "HALT_GATE_3",
    "DATA_PREP_EXCEPTION",
    "BASELINE_EXCEPTION",
    "VARIABLE_SELECTION_EXCEPTION",
    "FE_EXCEPTION",
    "UNKNOWN_EXCEPTION",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deepest_pipeline_frame(exc: BaseException):
    """Return the deepest traceback frame whose filename contains 'pipeline',
    excluding run_batch.py itself.  Returns None if not found."""
    tb = exc.__traceback__
    if tb is None:
        return None
    frames = traceback.extract_tb(tb)
    for frame in reversed(frames):
        fn = frame.filename.replace("\\", "/")
        if "pipeline" in fn and "run_batch" not in fn:
            return frame
    return None


def classify_outcome(exc: BaseException | None, tb_str: str) -> str:
    """Map an exception (or None) to one of the OUTCOME_CLASSES strings."""
    if exc is None:
        return "SUCCESS"

    if isinstance(exc, PipelineHaltedByUser):
        gate = getattr(exc, "gate", None)
        if gate == 1:
            return "HALT_GATE_1"
        if gate == 2:
            return "HALT_GATE_2"
        if gate == 3:
            return "HALT_GATE_3"
        return "UNKNOWN_EXCEPTION"

    frame = _deepest_pipeline_frame(exc)
    if frame is not None:
        fn = frame.filename.replace("\\", "/")
        if "data_prep_agent" in fn:
            return "DATA_PREP_EXCEPTION"
        if "variable_selector" in fn:
            return "VARIABLE_SELECTION_EXCEPTION"
        if "baseline_verifier" in fn:
            return "BASELINE_EXCEPTION"
        if "regression_runner" in fn:
            return "FE_EXCEPTION"

    return "UNKNOWN_EXCEPTION"


def traceback_signature(exc: BaseException) -> str:
    """Return a short stable string for clustering similar failures."""
    frame = _deepest_pipeline_frame(exc)
    exc_name = type(exc).__name__
    if frame is not None:
        basename = Path(frame.filename).name
        return f"{exc_name}:{basename}:{frame.name}"
    return f"{exc_name}:unknown"


def run_one(paper_id: str, papers_dir: Path, dry_run: bool) -> dict:
    """Run the pipeline for a single paper and return a result dict."""
    paper_dir = str(papers_dir / paper_id)

    if dry_run:
        logger.info("[DRY RUN] Would run: %s", paper_id)
        return {
            "paper_id": paper_id,
            "outcome": "DRY_RUN",
            "signature": "",
            "tb_snippet": "",
            "duration_s": 0.0,
        }

    logger.info("Running pipeline for %s …", paper_id)
    exc_caught: BaseException | None = None
    tb_str = ""

    t0 = perf_counter()
    try:
        run_paper(paper_dir, use_llm_gates=True, force_proceed=True)
    except Exception as exc:
        exc_caught = exc
        tb_str = traceback.format_exc()
    duration_s = perf_counter() - t0

    outcome = classify_outcome(exc_caught, tb_str)
    sig = "ok" if exc_caught is None else traceback_signature(exc_caught)

    # Keep only the last 5 lines of the traceback for the CSV
    tb_lines = tb_str.strip().splitlines()
    tb_snippet = "\n".join(tb_lines[-5:]) if tb_lines else ""

    if exc_caught is not None:
        logger.warning("[%s] %s — %s", paper_id, outcome, sig)
    else:
        logger.info("[%s] SUCCESS (%.1fs)", paper_id, duration_s)

    return {
        "paper_id": paper_id,
        "outcome": outcome,
        "signature": sig,
        "tb_snippet": tb_snippet,
        "duration_s": round(duration_s, 3),
    }


# ── Core batch logic ──────────────────────────────────────────────────────────

def run_batch(
    audit_csv: Path,
    papers_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    filter_class: str | None = None,
    prev_results_csv: Path | None = None,
) -> None:
    """Load audit CSV, select READY papers, run them, and write outputs."""

    # 1. Load audit and filter to READY
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")
    audit_df = pd.read_csv(audit_csv)
    ready_ids: list[str] = list(
        audit_df.loc[audit_df["status_class"] == "READY", "paper_id"]
    )
    logger.info("Audit: %d READY papers found.", len(ready_ids))

    # 2. Optional filter by prior outcome
    if filter_class is not None:
        if filter_class not in OUTCOME_CLASSES:
            raise ValueError(
                f"--filter-class must be one of: {sorted(OUTCOME_CLASSES)}\n"
                f"Got: {filter_class!r}"
            )
        prev_path = prev_results_csv or (output_dir / "batch_run_results.csv")
        if not prev_path.exists():
            raise FileNotFoundError(
                f"Previous results CSV not found: {prev_path}\n"
                "Run without --filter-class first to generate it."
            )
        prev_df = pd.read_csv(prev_path)
        filter_ids = set(
            prev_df.loc[prev_df["outcome"] == filter_class, "paper_id"]
        )
        ready_ids = [pid for pid in ready_ids if pid in filter_ids]
        logger.info(
            "Filter applied (%s): %d papers selected.", filter_class, len(ready_ids)
        )

    if not ready_ids:
        logger.info("No papers to run. Exiting.")
        return

    if dry_run:
        print(f"\n[DRY RUN] Papers that would be processed ({len(ready_ids)}):")
        for pid in ready_ids:
            print(f"  {pid}")
        print()
        return

    # 3. Run papers
    results: list[dict] = []
    for paper_id in ready_ids:
        result = run_one(paper_id, papers_dir, dry_run=False)
        result["run_timestamp"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        results.append(result)

    # 4. Write batch_run_results.csv
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "batch_run_results.csv"
    results_df = pd.DataFrame(
        results,
        columns=["paper_id", "outcome", "signature", "tb_snippet", "duration_s", "run_timestamp"],
    )
    results_df.to_csv(results_path, index=False)
    logger.info("Results written to %s", results_path)

    # 5. Build failure clusters JSON
    clusters: dict[str, dict[str, dict]] = {}
    for row in results:
        outcome = row["outcome"]
        if outcome == "SUCCESS":
            continue
        sig = row["signature"]
        if outcome not in clusters:
            clusters[outcome] = {}
        if sig not in clusters[outcome]:
            clusters[outcome][sig] = {"papers": [], "tb_snippet": row["tb_snippet"]}
        clusters[outcome][sig]["papers"].append(row["paper_id"])

    clusters_path = output_dir / "batch_failure_clusters.json"
    with clusters_path.open("w", encoding="utf-8") as fh:
        json.dump(clusters, fh, indent=2)
    logger.info("Failure clusters written to %s", clusters_path)

    # 6. Print summary
    outcome_counts = results_df["outcome"].value_counts().to_dict()
    print("\n── Batch run summary ─────────────────────────────────────────")
    for outcome, count in sorted(outcome_counts.items()):
        print(f"  {outcome:<30} {count}")
    print(f"  {'TOTAL':<30} {len(results)}")
    print("──────────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for the MAR simulation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audit-csv",
        default="outputs/pipeline_audit.csv",
        help="Source audit CSV produced by pipeline_audit.py",
    )
    parser.add_argument(
        "--papers-dir",
        default="papers",
        help="Root directory containing Paper_XXX sub-folders",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for batch_run_results.csv and batch_failure_clusters.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print READY papers without executing the pipeline",
    )
    parser.add_argument(
        "--filter-class",
        default=None,
        metavar="OUTCOME",
        help=(
            "Re-run only papers whose prior outcome equals OUTCOME. "
            f"Valid values: {sorted(OUTCOME_CLASSES)}"
        ),
    )
    parser.add_argument(
        "--prev-results",
        default=None,
        help=(
            "Previous results CSV used with --filter-class. "
            "Defaults to <output-dir>/batch_run_results.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve paths relative to the repo root (parent of scripts/)
    repo_root = Path(__file__).resolve().parent.parent

    audit_csv = Path(args.audit_csv) if Path(args.audit_csv).is_absolute() else repo_root / args.audit_csv
    papers_dir = Path(args.papers_dir) if Path(args.papers_dir).is_absolute() else repo_root / args.papers_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else repo_root / args.output_dir
    prev_results = (
        Path(args.prev_results)
        if args.prev_results
        else None
    )
    if prev_results and not prev_results.is_absolute():
        prev_results = repo_root / prev_results

    run_batch(
        audit_csv=audit_csv,
        papers_dir=papers_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
        filter_class=args.filter_class,
        prev_results_csv=prev_results,
    )


if __name__ == "__main__":
    main()
