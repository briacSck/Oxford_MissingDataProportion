"""
pipeline/baseline_verifier.py
------------------------------
Baseline Verifier: runs the original regression specification on the clean
baseline DataFrame and checks whether the estimated coefficient(s) match the
published values (within tolerance).
"""

from __future__ import annotations

import pandas as pd


def verify_baseline(
    baseline_df: pd.DataFrame,
    spec: dict,
    published_coef: dict[str, float],
) -> dict:
    """Run the baseline regression and compare to published coefficients.

    Parameters
    ----------
    baseline_df:
        Clean baseline DataFrame from ``prepare_baseline``.
    spec:
        Regression specification dict from ``parse_do_file``.
    published_coef:
        Mapping of variable name → published coefficient value, e.g.
        ``{"log_sales": 0.234}``.  Sourced from the paper's Table (entered
        manually in ``paper_info.xlsx``).

    Returns
    -------
    dict
        Verification report::

            {
                "match":        bool,           # True if all coefs within tolerance
                "tolerance":    float,          # absolute tolerance used
                "results":      pd.DataFrame,   # full regression output
                "discrepancies": list[dict],    # details of any mismatches
            }

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    """
    # TODO: Run regression using spec["estimator"]:
    #         - "reg" / "areg"  → statsmodels OLS (with FE as dummies if needed)
    #         - "reghdfe"       → linearmodels AbsorbingLS or within-transform
    #         - "xtreg fe"      → linearmodels PanelOLS with entity effects
    # TODO: Extract coefficient on spec["main_coef"].
    # TODO: Compare to published_coef[spec["main_coef"]] using absolute tolerance
    #       (default 0.005 — allows for rounding in published tables).
    # TODO: Build and return the verification report dict.
    # TODO: Log a warning if match is False; the orchestrator will trigger
    #       HUMAN GATE 1 before allowing the pipeline to continue.
    raise NotImplementedError("verify_baseline is not yet implemented.")
