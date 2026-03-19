"""pipeline/tests/test_data_loading.py
---------------------------------------
Real-data load tests for the 7 "Ready" papers.

Each test:
1. Reads spec.json → source_data_file
2. Skips if file not present on disk
3. Loads first 100 rows
4. Asserts dependent_var and at least one key_independent_var are present

These tests require real data files. If files are missing the test auto-skips,
keeping CI green.

Run with:
    pytest pipeline/tests/test_data_loading.py -v
    pytest pipeline/tests/test_data_loading.py -v -m "not slow"
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# Papers with populated, verified specs (skip Hu2025, P2P2018, StatusConsensus — empty specs)
READY_PAPERS = [
    "Paper_AntiDiscrimination",
    "Paper_Christensen2021",  # large — marked slow
    "Paper_Fang2022",
    "Paper_IPOFirms",
    "Paper_Mapping2024",
    "Paper_Meyer2024",
    "Paper_SMJ3560",
]

PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "papers"


def _load_spec(paper_id: str) -> dict:
    spec_path = PAPERS_DIR / paper_id / "spec.json"
    if not spec_path.exists():
        pytest.skip(f"spec.json not found for {paper_id}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _load_head(path: Path, n: int = 100) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(str(path), iterator=False).head(n)
    elif suffix == ".csv":
        return pd.read_csv(str(path), nrows=n)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(str(path), nrows=n)
    else:
        pytest.skip(f"Unsupported file format: {suffix}")


def _make_test_id(paper_id: str) -> str:
    return paper_id.replace("Paper_", "")


@pytest.mark.parametrize("paper_id", READY_PAPERS, ids=[_make_test_id(p) for p in READY_PAPERS])
def test_data_loads_and_has_spec_columns(paper_id: str) -> None:
    spec = _load_spec(paper_id)

    src = spec.get("source_data_file")
    if not src:
        pytest.skip(f"{paper_id}: source_data_file is None — run scripts/update_data_paths.py")

    data_path = Path(src)
    if not data_path.exists():
        pytest.skip(f"{paper_id}: data file not on disk: {data_path}")

    # Mark large file slow
    if paper_id == "Paper_Christensen2021":
        pytest.importorskip("pandas")  # ensure pandas is available
        if data_path.stat().st_size > 50 * 1024 * 1024:
            pytest.mark.slow  # pragma: no cover — marker applied at param level

    df = _load_head(data_path, n=100)

    dep = spec.get("dependent_var", "")
    keys = spec.get("key_independent_vars") or []

    assert dep, f"{paper_id}: dependent_var is empty in spec"
    assert dep in df.columns, (
        f"{paper_id}: dependent_var {dep!r} not in data columns {list(df.columns)[:20]}"
    )

    found_keys = [k for k in keys if k in df.columns]
    assert found_keys, (
        f"{paper_id}: none of key_independent_vars {keys} found in data columns "
        f"{list(df.columns)[:20]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "paper_id", ["Paper_Christensen2021"], ids=["Christensen2021"]
)
def test_christensen_slow_full_load(paper_id: str) -> None:
    """Full load of firmcyclepanel.dta (~16MB). Skipped unless -m slow."""
    spec = _load_spec(paper_id)
    src = spec.get("source_data_file")
    if not src or not Path(src).exists():
        pytest.skip(f"{paper_id}: data file not available")

    df = pd.read_stata(src)
    dep = spec.get("dependent_var", "")
    assert dep in df.columns, f"dependent_var {dep!r} missing after full load"
    assert len(df) > 0
