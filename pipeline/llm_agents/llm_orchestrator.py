"""
pipeline/llm_agents/llm_orchestrator.py
-----------------------------------------
Thin adapter layer: wraps gate judges, handles logging, and returns bool decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base_llm_agent import GateDecision
from .gate1_judge import judge_gate1
from .gate2_judge import judge_gate2
from .gate3_judge import judge_gate3

logger = logging.getLogger(__name__)

_LOG_FILE = "gate_decisions.log"


# ── Logging ───────────────────────────────────────────────────────────────────

def log_gate_decision(paper_dir: str, gate: int, decision: GateDecision) -> None:
    """Append gate decision to {paper_dir}/gate_decisions.log."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"[{ts}] Gate {gate}",
        f"  action   : {decision.action}",
        f"  approved : {decision.approved}",
        f"  confidence: {decision.confidence}",
        f"  reasoning: {decision.reasoning}",
    ]
    if decision.warning:
        lines.append(f"  WARNING  : {decision.warning}")
    if decision.issues:
        lines.append(f"  issues   : {decision.issues}")
    if decision.suspicious_patterns:
        lines.append(f"  suspicious_patterns: {decision.suspicious_patterns}")
    lines.append("")

    log_path = Path(paper_dir) / _LOG_FILE
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _route_decision(decision: GateDecision, gate: int, paper_id: str) -> bool:
    """Route decision action to a bool and log prominently if warning."""
    if decision.action == "HALT":
        logger.warning("[%s] Gate %d HALT: %s", paper_id, gate, decision.reasoning)
        return False
    if decision.action == "PROCEED_WITH_WARNING":
        logger.warning(
            "[%s] Gate %d PROCEED_WITH_WARNING: %s | warning=%s",
            paper_id, gate, decision.reasoning, decision.warning,
        )
        return True
    # PROCEED
    logger.info("[%s] Gate %d approved.", paper_id, gate)
    return True


# ── Gate wrappers ─────────────────────────────────────────────────────────────

def llm_gate1(
    paper_id: str,
    paper_dir: str,
    published_coef: Optional[float],
    published_sig: Optional[str],
    replicated_coef: float,
    replicated_pval: float,
    replicated_n: int,
    published_n: Optional[int] = None,
) -> bool:
    """Run Gate 1 judge, log decision, return approved bool."""
    decision = judge_gate1(
        paper_id=paper_id,
        published_coef=published_coef,
        published_significance=published_sig,
        replicated_coef=replicated_coef,
        replicated_pvalue=replicated_pval,
        replicated_n=replicated_n,
        published_n=published_n,
    )
    log_gate_decision(paper_dir, gate=1, decision=decision)
    return _route_decision(decision, gate=1, paper_id=paper_id)


def llm_gate2(
    paper_id: str,
    spec: dict,
    selected_key_vars: list[str],
    aux_var: str,
    df,
    paper_dir: str,
    paper_context: str = "",
) -> bool:
    """Run Gate 2 judge, log decision, return approved bool.

    Parameters
    ----------
    df:
        The baseline DataFrame (used for df.describe().to_dict()).
    """
    df_describe = df.describe().to_dict()
    decision = judge_gate2(
        paper_id=paper_id,
        spec=spec,
        selected_key_vars=selected_key_vars,
        aux_var=aux_var,
        df_describe=df_describe,
        paper_context=paper_context,
    )
    log_gate_decision(paper_dir, gate=2, decision=decision)
    return _route_decision(decision, gate=2, paper_id=paper_id)


def llm_gate3(
    paper_id: str,
    paper_dir: str,
) -> bool:
    """Run Gate 3 judge by reading qc_report.txt and regression_results.xlsx head."""
    import pandas as pd

    paper_path = Path(paper_dir)

    # Read QC report
    qc_path = paper_path / "qc_report.txt"
    qc_text = qc_path.read_text(encoding="utf-8") if qc_path.exists() else "(qc_report.txt not found)"

    # Read first rows of regression results as summary
    results_path = paper_path / "regression_results.xlsx"
    if results_path.exists():
        try:
            df_results = pd.read_excel(str(results_path))
            regression_summary = df_results.head(20).to_string(index=False)
        except Exception as exc:
            regression_summary = f"(Could not read regression_results.xlsx: {exc})"
    else:
        regression_summary = "(regression_results.xlsx not found)"

    decision = judge_gate3(
        paper_id=paper_id,
        qc_report_text=qc_text,
        regression_results_summary=regression_summary,
    )
    log_gate_decision(paper_dir, gate=3, decision=decision)
    return _route_decision(decision, gate=3, paper_id=paper_id)
