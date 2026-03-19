"""
pipeline/llm_agents/spec_resolver.py
--------------------------------------
Gate 0: Resolves unresolved macros and fills in spec fields using an LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)

# ── Schema for LLM output ─────────────────────────────────────────────────────

SPEC_SCHEMA = {
    "dependent_var": "string",
    "estimator": "string",
    "key_independent_vars": ["string"],
    "control_vars": ["string"],
    "fixed_effects": ["string"],
    "cluster_var": "string | null",
    "sample_restrictions": ["string"],
    "instrumental_vars": ["string"],
    "resolved_macros": {"macro_name": "value"},
    "confidence": "HIGH | MEDIUM | LOW",
    "ambiguous_flags": ["string"],
    "reasoning": "string",
}

# Metadata-only keys — must NOT appear in the written spec.json
_METADATA_KEYS = {"confidence", "ambiguous_flags", "reasoning", "resolved_macros"}

SPEC_SYSTEM_PROMPT = (
    "You are an expert econometrician helping to replicate empirical economics papers. "
    "You will be given the content of a Stata do-file or R script along with any "
    "available context about the paper. Your task is to extract the regression "
    "specification: dependent variable, estimator, key independent variables, "
    "controls, fixed effects, clustering, sample restrictions, and instrumental "
    "variables. Resolve ALL macros (local/global in Stata or variables in R) to their "
    "actual variable names. Set confidence to HIGH if unambiguous, MEDIUM if some "
    "interpretation was required, LOW if critical information is missing or highly "
    "ambiguous. List any ambiguous decisions in ambiguous_flags. "
    "Output ONLY valid JSON."
)


# ── Exception ─────────────────────────────────────────────────────────────────

class SpecResolverLowConfidence(Exception):
    def __init__(self, paper_id: str, flags: list[str]):
        self.paper_id = paper_id
        self.flags = flags
        super().__init__(f"[{paper_id}] Low confidence. Flags: {flags}")


# ── Main entry point ──────────────────────────────────────────────────────────

def resolve_spec(
    do_file_path: str,
    paper_dir: str,
    paper_context: str = "",
    existing_spec: Optional[dict] = None,
) -> dict:
    """Resolve spec from do/R file using LLM.

    Parameters
    ----------
    do_file_path:
        Path to the Stata do-file or R script.
    paper_dir:
        Directory where spec.json and spec_resolver_log.txt will be written.
    paper_context:
        Optional free-text context (abstract, readme, etc.).
    existing_spec:
        Existing spec dict to merge — LLM None values fall back to existing values.

    Returns
    -------
    dict
        The resolved spec (without metadata keys).

    Raises
    ------
    SpecResolverLowConfidence
        If LLM confidence is LOW.
    """
    # 1. Read script file
    do_text = Path(do_file_path).read_text(encoding="utf-8", errors="replace")
    paper_id = Path(paper_dir).name

    user_message = f"SCRIPT FILE:\n{do_text}"
    if paper_context:
        user_message = f"PAPER CONTEXT:\n{paper_context}\n\n" + user_message

    # 2. Call LLM
    agent = BaseLLMAgent()
    llm_result = agent.call_structured(SPEC_SYSTEM_PROMPT, user_message, SPEC_SCHEMA)

    # 3. Merge with existing_spec: where LLM returned None, keep existing value
    if existing_spec:
        for key, existing_val in existing_spec.items():
            if key not in _METADATA_KEYS and existing_val is not None:
                if llm_result.get(key) is None:
                    llm_result[key] = existing_val

    confidence = llm_result.get("confidence", "LOW")
    ambiguous_flags = llm_result.get("ambiguous_flags", [])
    reasoning = llm_result.get("reasoning", "")
    resolved_macros = llm_result.get("resolved_macros", {})

    # 4. Build clean spec (no metadata keys)
    spec = {k: v for k, v in llm_result.items() if k not in _METADATA_KEYS}

    # Write spec.json
    spec_path = Path(paper_dir) / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # 5. Write log
    log_lines = []
    if confidence == "MEDIUM":
        log_lines.append("REVIEW RECOMMENDED")
        log_lines.append("")
    log_lines.append(f"Confidence: {confidence}")
    log_lines.append(f"Reasoning: {reasoning}")
    if ambiguous_flags:
        log_lines.append(f"Ambiguous flags: {ambiguous_flags}")
    if resolved_macros:
        log_lines.append(f"resolved_macros: {json.dumps(resolved_macros, indent=2)}")

    log_text = "\n".join(log_lines) + "\n"
    log_path = Path(paper_dir) / "spec_resolver_log.txt"
    log_path.write_text(log_text, encoding="utf-8")

    # 6. Raise on LOW confidence (after writing log)
    if confidence == "LOW":
        raise SpecResolverLowConfidence(paper_id, ambiguous_flags)

    logger.info("[%s] spec_resolver: confidence=%s", paper_id, confidence)
    return spec
