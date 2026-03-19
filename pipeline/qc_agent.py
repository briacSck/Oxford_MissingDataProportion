"""
pipeline/qc_agent.py
---------------------
QC Agent: performs quality control checks on the regression results before
the paper is marked as complete. Flags anomalies for human review.
"""

from __future__ import annotations

import pandas as pd


def run_qc(
    paper_dir: str,
    spec: dict,
    results_df: pd.DataFrame,
) -> dict:
    """Run quality control checks on regression results.

    Parameters
    ----------
    paper_dir:
        Absolute path to the paper's directory (e.g. ``papers/Paper_XXX/``).
    spec:
        Regression specification dict from ``parse_do_file``.
    results_df:
        Regression results DataFrame from ``run_regressions``.

    Returns
    -------
    dict
        QC report::

            {
                "passed":   bool,
                "warnings": list[str],
                "errors":   list[str],
            }

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    """
    # TODO: Check that results_df has exactly 7 rows (one per proportion).
    # TODO: Check that N is monotonically decreasing as proportion increases.
    # TODO: Check that no coefficient is NaN or Inf.
    # TODO: Flag if any coefficient changes sign relative to baseline (potential issue).
    # TODO: Flag if |pct_change_coef| > 50% at any proportion (large sensitivity).
    # TODO: Flag if N drops below 30 at 50% missingness (too few observations).
    # TODO: Check that all required files exist:
    #         - paper_dir/regression_results.xlsx
    #         - paper_dir/paper_info.xlsx
    #         - paper_dir/missing/*.parquet (7 files)
    #         - paper_dir/listwise/*.parquet (7 files)
    # TODO: Compile warnings and errors lists.
    # TODO: Set passed = (len(errors) == 0).
    # TODO: Return QC report dict.
    # NOTE: The orchestrator will pause here for HUMAN GATE 3 before writing
    #       the final combined output file.
    raise NotImplementedError("run_qc is not yet implemented.")
