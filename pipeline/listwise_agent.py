"""
pipeline/listwise_agent.py
---------------------------
Listwise Agent: applies listwise deletion (complete-case analysis) to each
MAR-corrupted CSV, dropping rows where the *target variable* is NaN, and saves
the result as a new CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def apply_listwise(
    paper_dir: str,
) -> dict[str, dict[str, tuple[str, int, int]]]:
    """Apply listwise deletion to all MAR CSV files in ``{paper_dir}/missing/``.

    For each file, only rows where the *target variable* is NaN are dropped
    (not a global dropna — other columns may still contain NaN).

    Parameters
    ----------
    paper_dir:
        Root directory of the paper. Input files come from ``{paper_dir}/missing/``
        and output goes to ``{paper_dir}/listwise/``.

    Returns
    -------
    dict[str, dict[str, tuple[str, int, int]]]
        ``{varname: {label: (output_path, n_before, n_after)}}``
    """
    missing_dir = Path(paper_dir) / "missing"
    listwise_dir = Path(paper_dir) / "listwise"
    listwise_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, tuple[str, int, int]]] = {}

    for csv_path in sorted(missing_dir.glob("*_MAR_*.csv")):
        stem = csv_path.stem  # e.g. "x1_MAR_01pct"
        # Parse varname and label by splitting on "_MAR_"
        parts = stem.split("_MAR_", 1)
        if len(parts) != 2:
            logger.warning("Skipping unrecognised filename: %s", csv_path.name)
            continue
        varname, label = parts[0], parts[1]

        df = pd.read_csv(str(csv_path))
        n_before = len(df)

        if varname not in df.columns:
            logger.warning("Target column '%s' not in %s — skipping", varname, csv_path.name)
            continue

        df_clean = df.dropna(subset=[varname])
        n_after = len(df_clean)

        out_path = listwise_dir / f"{varname}_MAR_{label}_LD.csv"
        df_clean.to_csv(str(out_path), index=False)

        if varname not in results:
            results[varname] = {}
        results[varname][label] = (str(out_path), n_before, n_after)

        logger.info(
            "Listwise [%s_%s]: %d → %d rows (dropped %d, %.1f%%)",
            varname, label, n_before, n_after, n_before - n_after,
            100.0 * (n_before - n_after) / n_before if n_before else 0.0,
        )

    return results
