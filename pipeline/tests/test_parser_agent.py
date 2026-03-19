"""
pipeline/tests/test_parser_agent.py
------------------------------------
Tests for parser_agent.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import openpyxl
import pytest

# Add repo root to path so pipeline imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.parser_agent import (
    PaperSpec,
    _parse_do_file,
    _parse_r_file,
    _no_code_spec,
    parse_paper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_do(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.do"
    p.write_text(content, encoding='utf-8')
    return p


def _make_paper_dir(tmp_path: Path, paper_id: str = "Paper_Test") -> tuple[Path, Path]:
    """Create a minimal papers/<paper_id>/ with paper_info.xlsx and return (papers_root, paper_dir)."""
    papers_root = tmp_path / "papers"
    paper_dir = papers_root / paper_id
    paper_dir.mkdir(parents=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    headers = [
        "paper_short_name", "source_folder", "do_file", "data_file",
        "estimator", "depvar", "main_coef", "indepvars",
        "absorb", "cluster", "published_coef_main",
        "key_vars", "aux_var", "status", "notes",
    ]
    ws.append(headers)
    ws.append([paper_id, "src", "NONE", None, None, None, None, None,
               None, None, None, None, None, "test", None])
    wb.save(paper_dir / "paper_info.xlsx")
    return papers_root, paper_dir


# ---------------------------------------------------------------------------
# Test 1: basic OLS
# ---------------------------------------------------------------------------

def test_parse_do_basic(tmp_path):
    do = _write_do(tmp_path, "reg y x1 x2 x3, robust")
    result = _parse_do_file(do)
    assert result['dependent_var'] == 'y'
    assert result['key_independent_vars'] == ['x1']
    assert set(result['control_vars']) == {'x2', 'x3'}
    assert result['estimator'] == 'OLS'


# ---------------------------------------------------------------------------
# Test 2: FE regression with absorb and cluster
# ---------------------------------------------------------------------------

def test_parse_do_fe(tmp_path):
    do = _write_do(tmp_path, "reghdfe y x1 x2, absorb(firm year) cluster(firm)")
    result = _parse_do_file(do)
    assert result['dependent_var'] == 'y'
    assert result['estimator'] == 'FE'
    assert set(result['fixed_effects']) == {'firm', 'year'}
    assert result['cluster_var'] == 'firm'
    assert result['key_independent_vars'] == ['x1']
    assert result['control_vars'] == ['x2']


# ---------------------------------------------------------------------------
# Test 3: IV regression
# ---------------------------------------------------------------------------

def test_parse_do_iv(tmp_path):
    do = _write_do(tmp_path, "ivregress 2sls y (x1=z1) x2 x3")
    result = _parse_do_file(do)
    assert result['dependent_var'] == 'y'
    assert result['estimator'] == 'IV'
    assert 'z1' in result['instrumental_vars']
    assert 'x1' in result['key_independent_vars'] or 'x1' in result['control_vars']


# ---------------------------------------------------------------------------
# Test 4: global macro expansion
# ---------------------------------------------------------------------------

def test_parse_do_macro(tmp_path):
    do_content = (
        'global controls "x2 x3"\n'
        'reghdfe y x1 $controls, absorb(firm year) cl(firm)\n'
    )
    do = _write_do(tmp_path, do_content)
    result = _parse_do_file(do)
    assert result['dependent_var'] == 'y'
    all_vars = result['key_independent_vars'] + result['control_vars']
    assert 'x2' in all_vars, f"x2 not in {all_vars}"
    assert 'x3' in all_vars, f"x3 not in {all_vars}"


# ---------------------------------------------------------------------------
# Test 5: R felm parser
# ---------------------------------------------------------------------------

def test_parse_r_felm(tmp_path):
    r_content = (
        'library(lfe)\n'
        'fit <- felm(y ~ x1 + x2 | firm + year | 0 | firm, data=df)\n'
    )
    r_path = tmp_path / "test.R"
    r_path.write_text(r_content, encoding='utf-8')
    result = _parse_r_file(r_path)
    assert result['dependent_var'] == 'y'
    assert result['estimator'] == 'FE'
    assert 'firm' in result['fixed_effects']
    assert 'year' in result['fixed_effects']
    assert result['cluster_var'] == 'firm'
    assert 'R script — verify translation to Python' in result['flags']


# ---------------------------------------------------------------------------
# Test 6: no-code spec
# ---------------------------------------------------------------------------

def test_no_code(tmp_path):
    spec = _no_code_spec("Paper_Test", str(tmp_path))
    assert spec['manual_review_required'] is True
    assert spec['parse_confidence'] == 'low'
    assert spec['replication_code_type'] == 'none'
    assert any('manually' in f for f in spec['flags'])


# ---------------------------------------------------------------------------
# Test 7: spec.json written by parse_paper
# ---------------------------------------------------------------------------

def test_spec_json_written(tmp_path):
    papers_root, paper_dir = _make_paper_dir(tmp_path)
    # No actual DO file — will fall back to no-code spec
    spec = parse_paper("Paper_Test", papers_root, ra_task_root=tmp_path / "ra")
    spec_json = paper_dir / "spec.json"
    assert spec_json.exists(), "spec.json was not created"
    with open(spec_json) as f:
        loaded = json.load(f)
    assert loaded['paper_id'] == "Paper_Test"


# ---------------------------------------------------------------------------
# Test 8: all PaperSpec keys present in output
# ---------------------------------------------------------------------------

def test_all_keys_present(tmp_path):
    papers_root, paper_dir = _make_paper_dir(tmp_path)
    spec = parse_paper("Paper_Test", papers_root, ra_task_root=tmp_path / "ra")
    expected_keys = {
        'paper_id', 'paper_dir', 'title',
        'source_do_file', 'source_data_file', 'source_r_file',
        'replication_code_type',
        'estimator', 'dependent_var', 'key_independent_vars', 'control_vars',
        'fixed_effects', 'cluster_var', 'sample_restrictions',
        'interaction_terms', 'instrumental_vars',
        'published_coef', 'published_se', 'published_significance',
        'parse_confidence', 'flags', 'manual_review_required',
        'raw_regression_command',
    }
    missing = expected_keys - set(spec.keys())
    assert not missing, f"Missing keys in PaperSpec: {missing}"


# ---------------------------------------------------------------------------
# Test 9: /// continuation lines + local macro expansion
# ---------------------------------------------------------------------------

def test_parse_do_continuation_lines(tmp_path):
    """/// continuation + local macro expansion work together."""
    do_content = (
        'local controls "x2 x3 x4"\n'
        'reghdfe y x1 ///\n'
        '    `controls\', ///\n'
        '    absorb(firm year) cluster(firm)\n'
    )
    do = _write_do(tmp_path, do_content)
    result = _parse_do_file(do)
    assert result['dependent_var'] == 'y'
    assert result['key_independent_vars'] == ['x1']
    assert set(result['control_vars']) == {'x2', 'x3', 'x4'}
    assert set(result['fixed_effects']) == {'firm', 'year'}
    assert result['cluster_var'] == 'firm'
