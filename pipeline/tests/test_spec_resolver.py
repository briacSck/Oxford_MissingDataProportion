"""
pipeline/tests/test_spec_resolver.py
--------------------------------------
5 tests for spec_resolver (all mocked — no API key needed).
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from pipeline.llm_agents.base_llm_agent import BaseLLMAgent
from pipeline.llm_agents.spec_resolver import (
    SpecResolverLowConfidence,
    resolve_spec,
)

# ── Shared base mock response ─────────────────────────────────────────────────

_BASE = {
    "dependent_var": "log_wage",
    "estimator": "OLS",
    "key_independent_vars": ["union"],
    "control_vars": ["educ", "exper"],
    "fixed_effects": [],
    "cluster_var": None,
    "sample_restrictions": [],
    "instrumental_vars": [],
    "resolved_macros": {"controls": "educ exper"},
    "confidence": "HIGH",
    "ambiguous_flags": [],
    "reasoning": "Clear OLS specification.",
}


def _make_do_file(tmp_path, content="reg log_wage union educ exper"):
    """Write a dummy do-file and return its path."""
    do_file = tmp_path / "analysis.do"
    do_file.write_text(content, encoding="utf-8")
    return str(do_file)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_high_confidence_writes_spec_json(tmp_path):
    """HIGH confidence: spec.json is written with correct dependent_var."""
    mock_response = dict(_BASE, confidence="HIGH")

    with patch.object(BaseLLMAgent, "call_structured", return_value=mock_response):
        spec = resolve_spec(
            do_file_path=_make_do_file(tmp_path),
            paper_dir=str(tmp_path),
        )

    spec_path = tmp_path / "spec.json"
    assert spec_path.exists(), "spec.json was not written"
    written = json.loads(spec_path.read_text())
    assert written["dependent_var"] == "log_wage"
    assert spec["dependent_var"] == "log_wage"


def test_medium_confidence_writes_with_warning(tmp_path):
    """MEDIUM confidence: no exception raised; log contains 'REVIEW RECOMMENDED'."""
    mock_response = dict(_BASE, confidence="MEDIUM",
                         ambiguous_flags=["cluster level unclear"],
                         reasoning="Some macro interpretation required.")

    with patch.object(BaseLLMAgent, "call_structured", return_value=mock_response):
        spec = resolve_spec(
            do_file_path=_make_do_file(tmp_path),
            paper_dir=str(tmp_path),
        )

    log_path = tmp_path / "spec_resolver_log.txt"
    assert log_path.exists(), "spec_resolver_log.txt was not written"
    log_text = log_path.read_text()
    assert "REVIEW RECOMMENDED" in log_text
    assert spec["dependent_var"] == "log_wage"


def test_low_confidence_raises_exception(tmp_path):
    """LOW confidence: SpecResolverLowConfidence is raised; flags are attached."""
    flags = ["dependent var ambiguous", "estimator unclear"]
    mock_response = dict(_BASE, confidence="LOW", ambiguous_flags=flags,
                         reasoning="Critical fields unclear.")

    with patch.object(BaseLLMAgent, "call_structured", return_value=mock_response):
        with pytest.raises(SpecResolverLowConfidence) as exc_info:
            resolve_spec(
                do_file_path=_make_do_file(tmp_path),
                paper_dir=str(tmp_path),
            )

    exc = exc_info.value
    assert exc.paper_id == tmp_path.name
    assert "dependent var ambiguous" in exc.flags
    assert "estimator unclear" in exc.flags

    # Log should still be written even on LOW
    log_path = tmp_path / "spec_resolver_log.txt"
    assert log_path.exists(), "log should be written even on LOW confidence"


def test_macro_resolution_in_output(tmp_path):
    """resolved_macros appear in log but NOT as a top-level key in spec.json."""
    mock_response = dict(_BASE, resolved_macros={"controls": "educ exper"})

    with patch.object(BaseLLMAgent, "call_structured", return_value=mock_response):
        resolve_spec(
            do_file_path=_make_do_file(tmp_path),
            paper_dir=str(tmp_path),
        )

    log_text = (tmp_path / "spec_resolver_log.txt").read_text()
    assert "resolved_macros" in log_text

    spec_data = json.loads((tmp_path / "spec.json").read_text())
    assert "resolved_macros" not in spec_data, (
        "resolved_macros is metadata — must not appear in spec.json"
    )


def test_existing_spec_merged(tmp_path):
    """When LLM returns None for dependent_var, existing_spec value is kept."""
    mock_response = dict(_BASE, dependent_var=None, confidence="HIGH")

    existing = {"dependent_var": "wages", "estimator": "IV"}

    with patch.object(BaseLLMAgent, "call_structured", return_value=mock_response):
        spec = resolve_spec(
            do_file_path=_make_do_file(tmp_path),
            paper_dir=str(tmp_path),
            existing_spec=existing,
        )

    assert spec["dependent_var"] == "wages", (
        "Should fall back to existing_spec value when LLM returns None"
    )
