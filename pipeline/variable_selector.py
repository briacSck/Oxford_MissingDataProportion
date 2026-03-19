"""
pipeline/variable_selector.py
------------------------------
Variable Selector: chooses which variables to introduce MAR missingness into,
and which auxiliary variable to use as the MAR predictor.
"""

from __future__ import annotations

import pandas as pd


def select_variables(
    baseline_df: pd.DataFrame,
    spec: dict,
) -> dict:
    """Select key variables for missingness injection and the auxiliary variable.

    Parameters
    ----------
    baseline_df:
        Clean baseline DataFrame.
    spec:
        Regression specification dict from ``parse_do_file``.

    Returns
    -------
    dict
        Selection result::

            {
                "key_vars":  list[str],   # 3–5 variables that will receive missingness
                "aux_var":   str,         # auxiliary variable used as MAR predictor
                "rationale": str,         # human-readable explanation of the choice
            }

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    """
    # TODO: Filter candidate variables:
    #         - Must be in spec["indepvars"].
    #         - Must be continuous or ordinal (skip binary dummies: nunique <= 2).
    #         - Must not be FE identifiers (skip spec["absorb"] vars).
    # TODO: Rank candidates by variance (higher variance → more informative to
    #       study the effect of missingness).
    # TODO: Select MIN_KEY_VARS to MAX_KEY_VARS top candidates (from config.py).
    # TODO: Choose aux_var: a variable NOT in key_vars, correlated with at least
    #       one key_var (use Pearson |r| > 0.1 as a minimum threshold).
    # TODO: Build rationale string for human review.
    # TODO: Return selection dict.
    # NOTE: The orchestrator will pause here for HUMAN GATE 2 before proceeding
    #       to the missingness generator.
    raise NotImplementedError("select_variables is not yet implemented.")
