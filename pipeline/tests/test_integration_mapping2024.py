"""pipeline/tests/test_integration_mapping2024.py
--------------------------------------------------
Full-pipeline integration test for Paper_Mapping2024.

Requires either:
  - papers/Paper_Mapping2024/baseline.parquet (pre-existing), OR
  - the raw DTA file at its spec.json source_data_file path

LLM gate judges are mocked to return PROCEED, so no API key is needed.

Run with:
    pytest pipeline/tests/test_integration_mapping2024.py -v -m integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR  = REPO_ROOT / "papers" / "Paper_Mapping2024"
SPEC_PATH  = PAPER_DIR / "spec.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_spec() -> dict:
    if not SPEC_PATH.exists():
        pytest.skip("Paper_Mapping2024/spec.json not found")
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _get_data_path() -> Path | None:
    spec = _load_spec()
    src = spec.get("source_data_file")
    if src and Path(src).exists():
        return Path(src)
    return None


def _has_baseline() -> bool:
    return (PAPER_DIR / "baseline.parquet").exists()


# ── autouse skip fixture ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def skip_if_no_data() -> None:
    if not _has_baseline() and _get_data_path() is None:
        pytest.skip(
            "Paper_Mapping2024: no baseline.parquet and data file not on disk — "
            "run scripts/update_data_paths.py and ensure data is present"
        )


# ── Mocked GateDecision return value ─────────────────────────────────────────

_PROCEED_RESPONSE = {
    "approved": True,
    "confidence": "HIGH",
    "reasoning": "Mocked PROCEED for integration test.",
    "action": "PROCEED",
    "warning": None,
    "issues": None,
    "suspicious_patterns": None,
}


# ── Test 1: data prep ─────────────────────────────────────────────────────────

def test_data_prep_mapping2024(tmp_path: Path) -> None:
    """prepare_baseline loads data and writes baseline.parquet."""
    data_path = _get_data_path()
    if data_path is None:
        pytest.skip("Raw data file not available; cannot test data prep")

    spec = _load_spec()
    from pipeline.data_prep_agent import prepare_baseline

    df = prepare_baseline(str(data_path), spec, str(tmp_path))

    dep = spec.get("dependent_var", "")
    assert dep in df.columns, f"dependent_var {dep!r} not in baseline columns"
    assert len(df) > 0, "baseline DataFrame is empty"
    assert (tmp_path / "baseline.parquet").exists(), "baseline.parquet not written"


# ── Test 2: baseline verifier ────────────────────────────────────────────────

def test_baseline_verifier_mapping2024() -> None:
    """verify_baseline runs OLS and returns a valid result dict."""
    import pandas as pd
    from pipeline.baseline_verifier import verify_baseline

    spec = _load_spec()

    if _has_baseline():
        df = pd.read_parquet(str(PAPER_DIR / "baseline.parquet"))
    else:
        data_path = _get_data_path()
        if data_path is None:
            pytest.skip("No baseline.parquet and no raw data")
        from pipeline.data_prep_agent import prepare_baseline
        with tempfile.TemporaryDirectory() as tmp:
            df = prepare_baseline(str(data_path), spec, tmp)

    result = verify_baseline(df, spec, {})

    assert result.get("coef_estimate") is not None, "coef_estimate is None"
    assert result.get("n_obs", 0) > 0, "n_obs is 0 or missing"
    assert result.get("key_coef_name") == "log_pop_black_aa", (
        f"expected key_coef_name='log_pop_black_aa', got {result.get('key_coef_name')!r}"
    )


# ── Test 3: full pipeline (mocked LLM gates) ─────────────────────────────────

def test_full_pipeline_mapping2024() -> None:
    """Full run_paper with force_proceed=True, LLM gates mocked to PROCEED."""
    from pipeline.orchestrator import run_paper

    with patch(
        "pipeline.llm_agents.base_llm_agent.BaseLLMAgent.call_structured",
        return_value=_PROCEED_RESPONSE,
    ):
        run_paper(str(PAPER_DIR), use_llm_gates=True, force_proceed=True)

    # Verify expected artifacts exist
    assert (PAPER_DIR / "baseline.parquet").exists(), "baseline.parquet missing"
    assert (PAPER_DIR / "selection.json").exists(),   "selection.json missing"
    assert (PAPER_DIR / "regression_results.xlsx").exists(), "regression_results.xlsx missing"
    assert (PAPER_DIR / "qc_report.txt").exists(),    "qc_report.txt missing"

    missing_csvs = list((PAPER_DIR / "missing").glob("*.csv"))
    assert len(missing_csvs) > 0, "No missing/*.csv files generated"

    listwise_csvs = list((PAPER_DIR / "listwise").glob("*.csv"))
    assert len(listwise_csvs) > 0, "No listwise/*.csv files generated"
