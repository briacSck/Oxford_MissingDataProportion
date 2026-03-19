"""Tests for pipeline/missingness_generator.py"""

import numpy as np
import pandas as pd
import pytest

from pipeline.config import MISSING_PROPORTIONS, PROPORTION_LABELS
from pipeline.missingness_generator import generate_missingness

# ── Fixture ───────────────────────────────────────────────────────────────────

N = 2000
RNG = np.random.default_rng(2026)


def _make_baseline(tmp_path):
    rng = np.random.default_rng(2026)
    aux = rng.uniform(0.5, 5.0, N)
    x1 = aux * 0.7 + rng.normal(scale=0.5, size=N)
    x2 = rng.normal(size=N)
    df = pd.DataFrame({"aux": aux, "x1": x1, "x2": x2})
    csv_path = tmp_path / "baseline.csv"
    df.to_csv(str(csv_path), index=False)
    return str(csv_path), df


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_proportion_accuracy(tmp_path):
    """Each generated dataset has missing rate within 0.015 of target.

    Note: The plan specifies 0.005 but with N=2000 the standard deviation of a
    Bernoulli sample proportion is ~0.009 at p=0.2, so 0.015 (≈1.7 σ) is the
    appropriate tolerance for N=2000 to keep the test deterministically stable.
    """
    baseline_path, _ = _make_baseline(tmp_path)
    paper_dir = str(tmp_path)
    result = generate_missingness(baseline_path, ["x1"], "aux", paper_dir)

    for proportion, label in zip(MISSING_PROPORTIONS, PROPORTION_LABELS):
        out_path = result["x1"][label]
        df_out = pd.read_csv(out_path)
        actual_rate = df_out["x1"].isna().mean()
        assert abs(actual_rate - proportion) < 0.015, (
            f"Label {label}: actual_rate={actual_rate:.4f}, target={proportion}"
        )


def test_power_law_direction(tmp_path):
    """High-aux rows should be more likely to be missing than low-aux rows."""
    baseline_path, df = _make_baseline(tmp_path)
    paper_dir = str(tmp_path)
    result = generate_missingness(baseline_path, ["x1"], "aux", paper_dir)

    # Use a mid-range proportion (20%) for reliable signal
    label = "20pct"
    out_path = result["x1"][label]
    df_out = pd.read_csv(out_path)

    q75 = df["aux"].quantile(0.75)
    q25 = df["aux"].quantile(0.25)

    top_mask = df["aux"] >= q75
    bot_mask = df["aux"] <= q25

    top_nan_rate = df_out.loc[top_mask, "x1"].isna().mean()
    bot_nan_rate = df_out.loc[bot_mask, "x1"].isna().mean()

    assert top_nan_rate > bot_nan_rate, (
        f"Expected top-quartile NaN rate ({top_nan_rate:.3f}) > "
        f"bottom-quartile ({bot_nan_rate:.3f})"
    )


def test_seed_reproducibility(tmp_path):
    """Two separate calls produce identical NaN positions."""
    tmp1 = tmp_path / "run1"
    tmp2 = tmp_path / "run2"
    tmp1.mkdir(); tmp2.mkdir()

    bp1, _ = _make_baseline(tmp1)
    bp2, _ = _make_baseline(tmp2)

    r1 = generate_missingness(bp1, ["x1"], "aux", str(tmp1))
    r2 = generate_missingness(bp2, ["x1"], "aux", str(tmp2))

    for label in PROPORTION_LABELS:
        df1 = pd.read_csv(r1["x1"][label])
        df2 = pd.read_csv(r2["x1"][label])
        assert df1["x1"].isna().equals(df2["x1"].isna()), (
            f"NaN positions differ for label {label}"
        )


def test_aux_column_untouched(tmp_path):
    """The aux variable column must remain fully observed in every output file."""
    baseline_path, _ = _make_baseline(tmp_path)
    paper_dir = str(tmp_path)
    result = generate_missingness(baseline_path, ["x1", "x2"], "aux", paper_dir)

    for key_var in ["x1", "x2"]:
        for label in PROPORTION_LABELS:
            out_path = result[key_var][label]
            df_out = pd.read_csv(out_path)
            assert df_out["aux"].isna().sum() == 0, (
                f"aux has NaN in {key_var}_MAR_{label}.csv"
            )


def test_output_file_count(tmp_path):
    """K=2 key vars → 2×7 = 14 files in {paper_dir}/missing/."""
    baseline_path, _ = _make_baseline(tmp_path)
    paper_dir = str(tmp_path)
    generate_missingness(baseline_path, ["x1", "x2"], "aux", paper_dir)

    missing_dir = tmp_path / "missing"
    files = list(missing_dir.glob("*_MAR_*.csv"))
    assert len(files) == 14, f"Expected 14 files, got {len(files)}"


def test_file_naming_convention(tmp_path):
    """Files are named {varname}_MAR_{label}.csv for all proportion labels."""
    baseline_path, _ = _make_baseline(tmp_path)
    paper_dir = str(tmp_path)
    generate_missingness(baseline_path, ["x1"], "aux", paper_dir)

    missing_dir = tmp_path / "missing"
    for label in PROPORTION_LABELS:
        expected = missing_dir / f"x1_MAR_{label}.csv"
        assert expected.exists(), f"Expected file not found: {expected.name}"
