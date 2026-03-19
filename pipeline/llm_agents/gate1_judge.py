"""
pipeline/llm_agents/gate1_judge.py
------------------------------------
Gate 1: LLM judge for baseline coefficient verification.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base_llm_agent import BaseLLMAgent, GateDecision

logger = logging.getLogger(__name__)

_GATE1_SYSTEM_PROMPT = (
    "You are an expert econometrician evaluating whether a replicated regression "
    "result is sufficiently close to published values to proceed with a Missing At "
    "Random (MAR) simulation study. Consider: magnitude of coefficient difference, "
    "sign agreement, statistical significance tier agreement, and sample size "
    "differences. Approve if the replication is reasonable given normal variation "
    "across software/data versions. Flag serious discrepancies. "
    "Respond ONLY with valid JSON."
)

_GATE1_SCHEMA = {
    "approved": "bool",
    "confidence": "HIGH|MEDIUM|LOW",
    "reasoning": "string",
    "action": "PROCEED|PROCEED_WITH_WARNING|HALT",
    "warning": "string|null",
}


# ── Significance tier helpers ─────────────────────────────────────────────────

def _pval_to_tier(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def _same_sig_tier(pval: float, published_sig: Optional[str]) -> bool:
    if not published_sig:
        return True
    norm = "" if published_sig == "ns" else published_sig
    return _pval_to_tier(pval) == norm


# ── Main judge ────────────────────────────────────────────────────────────────

def judge_gate1(
    paper_id: str,
    published_coef: Optional[float],
    published_significance: Optional[str],
    replicated_coef: float,
    replicated_pvalue: float,
    replicated_n: int,
    published_n: Optional[int] = None,
) -> GateDecision:
    """Judge whether the baseline replication is acceptable.

    Rule-based pre-checks (no LLM):
    1. If published_coef is None → auto-approve with warning.
    2. If |relative_diff| < 10% AND same significance tier → auto-approve.
    3. Otherwise → call LLM.
    """
    # Pre-check 1: no published coefficient to compare against
    if published_coef is None:
        return GateDecision(
            approved=True,
            confidence="LOW",
            reasoning="No published coefficient available for comparison.",
            action="PROCEED_WITH_WARNING",
            warning="No published coefficient — cannot verify baseline match.",
        )

    # Pre-check 2: close enough without LLM
    rel_diff = abs((replicated_coef - published_coef) / published_coef)
    if rel_diff < 0.10 and _same_sig_tier(replicated_pvalue, published_significance):
        return GateDecision(
            approved=True,
            confidence="HIGH",
            reasoning=f"Replicated coefficient within 10% of published ({rel_diff:.1%}) with matching significance tier.",
            action="PROCEED",
        )

    # All other cases (including sign flip) → call LLM
    agent = BaseLLMAgent()
    user_msg = (
        f"Paper: {paper_id}\n"
        f"Published coefficient: {published_coef}\n"
        f"Published significance: {published_significance or 'not reported'}\n"
        f"Replicated coefficient: {replicated_coef}\n"
        f"Replicated p-value: {replicated_pvalue}\n"
        f"Replicated N: {replicated_n}\n"
        f"Published N: {published_n if published_n is not None else 'not reported'}\n"
        f"Relative difference: {rel_diff:.1%}\n"
        f"Sign agreement: {(replicated_coef > 0) == (published_coef > 0)}\n"
    )
    result = agent.call_structured(_GATE1_SYSTEM_PROMPT, user_msg, _GATE1_SCHEMA)

    return GateDecision(
        approved=bool(result.get("approved", False)),
        confidence=result.get("confidence", "MEDIUM"),
        reasoning=result.get("reasoning", ""),
        action=result.get("action", "HALT"),
        warning=result.get("warning"),
    )
