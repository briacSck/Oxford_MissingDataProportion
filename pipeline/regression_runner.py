"""
pipeline/regression_runner.py
------------------------------
Regression Runner: re-runs the original regression specification on each
listwise-deleted dataset and collects results into a tidy DataFrame.
"""

from __future__ import annotations

import pandas as pd


def run_regressions(
    ld_dir: str,
    spec: dict,
    baseline_results: pd.DataFrame,
) -> pd.DataFrame:
    """Run the regression on every listwise-deleted dataset and compile results.

    Parameters
    ----------
    ld_dir:
        Directory containing listwise-deleted parquet files
        (e.g. ``papers/Paper_XXX/listwise/``).
    spec:
        Regression specification dict from ``parse_do_file``.
    baseline_results:
        DataFrame row(s) with baseline (complete-data) regression results,
        used to compute percentage changes.

    Returns
    -------
    pd.DataFrame
        Results table with columns matching ``REGRESSION_RESULTS_COLUMNS``
        from ``config.py``.  One row per (proportion_label × variable).

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    """
    # TODO: Import REGRESSION_RESULTS_COLUMNS, PROPORTION_LABELS,
    #       SIGNIFICANCE_LEVELS from config.
    # TODO: Glob all *.parquet files in ld_dir; match filenames to PROPORTION_LABELS.
    # TODO: For each proportion dataset:
    #   a. Load parquet file.
    #   b. Run regression using spec["estimator"] (same as baseline_verifier logic).
    #   c. Extract coef, se, tstat, pvalue for spec["main_coef"].
    #   d. Compute significance stars from SIGNIFICANCE_LEVELS.
    #   e. Compute N (len of dataset after listwise deletion — already done).
    #   f. Compute pct_change_n and pct_change_coef vs. baseline.
    #   g. Append a result row to the accumulator list.
    # TODO: Concatenate all rows into a DataFrame with REGRESSION_RESULTS_COLUMNS.
    # TODO: Save results to papers/Paper_XXX/regression_results.xlsx (append sheet
    #       or overwrite, depending on run mode).
    # TODO: Return the results DataFrame.
    raise NotImplementedError("run_regressions is not yet implemented.")
