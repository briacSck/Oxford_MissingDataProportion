"""
pipeline/tests/test_gate_judges.py
------------------------------------
6 tests for gate1/gate2/gate3 judges (all mocked — no API key needed).
"""

from unittest.mock import patch

import pytest

from pipeline.llm_agents.base_llm_agent import BaseLLMAgent, GateDecision
from pipeline.llm_agents.gate1_judge import judge_gate1
from pipeline.llm_agents.gate2_judge import judge_gate2
from pipeline.llm_agents.gate3_judge import judge_gate3


# ── Gate 1 tests ──────────────────────────────────────────────────────────────

def test_gate1_identical_coef_no_llm_call():
    """Close coefficient + same significance tier → auto-approve, no LLM call."""
    with patch.object(BaseLLMAgent, "call_structured") as mock_cs:
        decision = judge_gate1(
            paper_id="Paper_Test",
            published_coef=1.0,
            published_significance="***",
            replicated_coef=1.05,
            replicated_pvalue=0.001,
            replicated_n=500,
            published_n=500,
        )

    mock_cs.assert_not_called()
    assert decision.approved is True
    assert decision.action == "PROCEED"


def test_gate1_sign_flip_calls_llm():
    """Sign flip (pub=+1.0, rep=-1.0) → LLM is called."""
    llm_response = {
        "approved": False,
        "confidence": "HIGH",
        "reasoning": "Sign flip detected.",
        "action": "HALT",
        "warning": "Replicated sign opposite to published.",
    }
    with patch.object(BaseLLMAgent, "call_structured", return_value=llm_response) as mock_cs:
        decision = judge_gate1(
            paper_id="Paper_Test",
            published_coef=1.0,
            published_significance="***",
            replicated_coef=-1.0,
            replicated_pvalue=0.001,
            replicated_n=500,
        )

    mock_cs.assert_called_once()
    assert decision.approved is False


def test_gate1_no_published_coef_auto_approves():
    """No published coefficient → auto-approve with warning, no LLM call."""
    with patch.object(BaseLLMAgent, "call_structured") as mock_cs:
        decision = judge_gate1(
            paper_id="Paper_Test",
            published_coef=None,
            published_significance=None,
            replicated_coef=0.5,
            replicated_pvalue=0.05,
            replicated_n=300,
        )

    mock_cs.assert_not_called()
    assert decision.approved is True
    assert decision.action == "PROCEED_WITH_WARNING"


# ── Gate 2 tests ──────────────────────────────────────────────────────────────

def test_gate2_binary_var_flagged():
    """Binary variable concern → LLM IS called; decision reflects LLM output."""
    llm_response = {
        "approved": False,
        "confidence": "HIGH",
        "reasoning": "x1 appears to be a binary indicator — MAR simulation problematic.",
        "action": "HALT",
        "warning": "binary variable detected",
        "issues": ["x1 is binary"],
    }
    spec = {"dependent_var": "y", "estimator": "OLS", "key_independent_vars": ["x1", "x2"]}

    with patch.object(BaseLLMAgent, "call_structured", return_value=llm_response) as mock_cs:
        decision = judge_gate2(
            paper_id="Paper_Test",
            spec=spec,
            selected_key_vars=["x1", "x2"],
            aux_var="z",
            df_describe={"x1": {"mean": 0.5, "std": 0.5, "min": 0.0, "max": 1.0}},
        )

    mock_cs.assert_called_once()
    assert decision.approved is False
    assert decision.issues is not None and len(decision.issues) > 0


def test_gate2_aux_in_key_vars_flagged():
    """aux_var in key_vars → rule-based HALT, no LLM call."""
    spec = {"dependent_var": "y", "estimator": "OLS"}

    with patch.object(BaseLLMAgent, "call_structured") as mock_cs:
        decision = judge_gate2(
            paper_id="Paper_Test",
            spec=spec,
            selected_key_vars=["x1", "x2"],
            aux_var="x1",
            df_describe={},
        )

    mock_cs.assert_not_called()
    assert decision.approved is False
    assert decision.action == "HALT"


# ── Gate 3 test ───────────────────────────────────────────────────────────────

def test_gate3_all_pass_approves():
    """All-PASS QC report → LLM approves."""
    llm_response = {
        "approved": True,
        "confidence": "HIGH",
        "reasoning": "All QC checks passed with no issues.",
        "action": "PROCEED",
        "warning": None,
        "suspicious_patterns": [],
    }

    with patch.object(BaseLLMAgent, "call_structured", return_value=llm_response):
        decision = judge_gate3(
            paper_id="Paper_Test",
            qc_report_text="CHECK coefficient_stability: PASS\nCHECK sample_size: PASS\n",
            regression_results_summary="coef  se  pval\n0.5   0.1  0.001\n",
        )

    assert decision.approved is True
    assert decision.action == "PROCEED"
