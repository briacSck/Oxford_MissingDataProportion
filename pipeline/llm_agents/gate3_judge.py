"""
pipeline/llm_agents/gate3_judge.py
------------------------------------
Gate 3: LLM judge for QC report review.
"""

from __future__ import annotations

import logging

from .base_llm_agent import BaseLLMAgent, GateDecision

logger = logging.getLogger(__name__)

_GATE3_SYSTEM_PROMPT = (
    "You are an expert econometrician reviewing a QC report for a Missing At Random "
    "(MAR) simulation study. The QC report contains PASS/WARN/ERROR entries for each "
    "check performed. Apply the following decision rules:\n"
    "- APPROVE (action=PROCEED) if all entries are PASS.\n"
    "- APPROVE WITH WARNING (action=PROCEED_WITH_WARNING) if there are only WARNs "
    "but no ERRORs.\n"
    "- HALT (action=HALT) if there are any ERRORs.\n"
    "Additionally, flag any dramatic coefficient shifts (>50%) at low missingness "
    "proportions (<10%) as suspicious patterns requiring review.\n"
    "List all suspicious patterns you detect. Respond ONLY with valid JSON."
)

_GATE3_SCHEMA = {
    "approved": "bool",
    "confidence": "HIGH|MEDIUM|LOW",
    "reasoning": "string",
    "action": "PROCEED|PROCEED_WITH_WARNING|HALT",
    "warning": "string|null",
    "suspicious_patterns": ["string"],
}


def judge_gate3(
    paper_id: str,
    qc_report_text: str,
    regression_results_summary: str,
) -> GateDecision:
    """Judge whether the QC report is acceptable to proceed.

    Always calls LLM — no rule-based shortcut.
    """
    agent = BaseLLMAgent()
    user_msg = (
        f"Paper: {paper_id}\n\n"
        f"QC REPORT:\n{qc_report_text}\n\n"
        f"REGRESSION RESULTS SUMMARY:\n{regression_results_summary}\n"
    )
    result = agent.call_structured(_GATE3_SYSTEM_PROMPT, user_msg, _GATE3_SCHEMA)

    return GateDecision(
        approved=bool(result.get("approved", False)),
        confidence=result.get("confidence", "MEDIUM"),
        reasoning=result.get("reasoning", ""),
        action=result.get("action", "HALT"),
        warning=result.get("warning"),
        suspicious_patterns=result.get("suspicious_patterns", []),
    )
