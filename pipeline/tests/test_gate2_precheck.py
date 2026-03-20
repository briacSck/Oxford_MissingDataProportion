"""
pipeline/tests/test_gate2_precheck.py
---------------------------------------
Regression tests for gate2_judge deterministic pre-checks.
No LLM mocking needed — all tested paths are rule-based.
"""

import json

import pandas as pd
import pytest

from pipeline.llm_agents.gate2_judge import judge_gate2


# ---------------------------------------------------------------------------
# Test 1: empty aux_var → HALT (pre-check B)
# ---------------------------------------------------------------------------

def test_empty_aux_var_halts():
    dd = {"x": {"min": 0.0, "max": 5.0, "mean": 2.5, "std": 1.0}}
    d = judge_gate2("P", {}, ["x"], "", dd)
    assert d.action == "HALT"
    assert "empty" in (d.warning or "").lower()


# ---------------------------------------------------------------------------
# Test 2: aux_var in key_vars → HALT (pre-check C — regression on existing check)
# ---------------------------------------------------------------------------

def test_aux_in_key_vars_halts():
    dd = {
        "x": {"min": 0.0, "max": 5.0, "mean": 2.5, "std": 1.0},
        "y": {"min": 0.0, "max": 5.0, "mean": 2.5, "std": 1.0},
    }
    d = judge_gate2("P", {}, ["x", "y"], "x", dd)
    assert d.action == "HALT"


# ---------------------------------------------------------------------------
# Test 3: no key_vars → HALT (pre-check A)
# ---------------------------------------------------------------------------

def test_empty_key_vars_halts():
    dd = {"aux": {"min": 0.0, "max": 10.0, "mean": 5.0, "std": 2.0}}
    d = judge_gate2("P", {}, [], "aux", dd)
    assert d.action == "HALT"
    assert "key_vars" in (d.warning or "").lower()


# ---------------------------------------------------------------------------
# Test 4: whitespace-only aux_var → HALT (pre-check B variant)
# ---------------------------------------------------------------------------

def test_whitespace_aux_var_halts():
    dd = {"x": {"min": 0.0, "max": 5.0, "mean": 2.5, "std": 1.0}}
    d = judge_gate2("P", {}, ["x"], "   ", dd)
    assert d.action == "HALT"
    assert "empty" in (d.warning or "").lower()


# ---------------------------------------------------------------------------
# Test 5: Timestamp in df_describe → serialises without raising
# ---------------------------------------------------------------------------

def test_timestamp_serializes_safely():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "ts": pd.to_datetime(["2020-01", "2020-02", "2020-03"]),
    })
    df_describe = df.describe(include="all").to_dict()
    msg = json.dumps(df_describe, default=str)   # must not raise
    assert isinstance(msg, str)
