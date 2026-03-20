"""
pipeline/llm_agents/gate2_judge.py
------------------------------------
Gate 2: LLM judge for variable selection review.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .base_llm_agent import BaseLLMAgent, GateDecision

logger = logging.getLogger(__name__)

_GATE2_SYSTEM_PROMPT = (
    "You are an expert econometrician reviewing variable selection for a Missing At "
    "Random (MAR) simulation study. The selected key independent variables will have "
    "missingness induced in them. The aux_var is the auxiliary variable that drives "
    "the MAR mechanism. Review whether the selected variables are appropriate: they "
    "should be meaningful regressors, not binary/constant/near-constant (which would "
    "make MAR simulation problematic), and the aux_var must be a valid continuous "
    "predictor separate from the key variables. "
    "Respond ONLY with valid JSON."
)

_GATE2_SCHEMA = {
    "approved": "bool",
    "confidence": "HIGH|MEDIUM|LOW",
    "reasoning": "string",
    "action": "PROCEED|PROCEED_WITH_WARNING|HALT",
    "warning": "string|null",
    "issues": ["string"],
}


def judge_gate2(
    paper_id: str,
    spec: dict,
    selected_key_vars: list[str],
    aux_var: str,
    df_describe: dict,
    paper_context: str = "",
) -> GateDecision:
    """Judge whether the variable selection is valid for MAR simulation.

    Rule-based pre-check (no LLM):
    - aux_var must not be in selected_key_vars.

    All other cases call LLM.
    """
    # Pre-check A: no key_vars
    if not selected_key_vars:
        return GateDecision(
            approved=False, confidence="HIGH",
            reasoning="No key_vars selected — cannot run MAR simulation.",
            action="HALT", warning="key_vars is empty",
        )

    # Pre-check B: empty aux_var
    if not aux_var or not aux_var.strip():
        return GateDecision(
            approved=False, confidence="HIGH",
            reasoning="aux_var is empty — MAR mechanism needs an auxiliary variable.",
            action="HALT", warning="aux_var is empty",
        )

    # Pre-check C: aux_var in key_vars
    if aux_var in selected_key_vars:
        return GateDecision(
            approved=False, confidence="HIGH",
            reasoning="aux_var must not be in key_vars — corrupts MAR mechanism.",
            action="HALT", warning="aux_var is in selected_key_vars",
        )

    # All deterministic checks passed — call LLM
    agent = BaseLLMAgent()
    user_msg = (
        f"Paper: {paper_id}\n"
        f"Spec: {json.dumps(spec, indent=2)}\n"
        f"Selected key variables: {selected_key_vars}\n"
        f"Auxiliary variable (MAR driver): {aux_var!r}\n"
        f"Data summary (describe):\n{json.dumps(df_describe, indent=2, default=str)}\n"
    )
    if paper_context:
        user_msg = f"Paper context:\n{paper_context}\n\n" + user_msg

    result = agent.call_structured(_GATE2_SYSTEM_PROMPT, user_msg, _GATE2_SCHEMA)

    return GateDecision(
        approved=bool(result.get("approved", False)),
        confidence=result.get("confidence", "MEDIUM"),
        reasoning=result.get("reasoning", ""),
        action=result.get("action", "HALT"),
        warning=result.get("warning"),
        issues=result.get("issues", []),
    )
