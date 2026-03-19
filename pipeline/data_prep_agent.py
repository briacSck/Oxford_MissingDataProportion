"""
pipeline/data_prep_agent.py
---------------------------
Data Prep Agent: loads the raw dataset, applies any pre-processing described in
the DO file (merges, variable construction, sample restrictions), and saves a
clean baseline DataFrame ready for regression.
"""

from __future__ import annotations

import pandas as pd


def prepare_baseline(
    raw_data_path: str,
    spec: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Load and pre-process the raw dataset into a clean baseline DataFrame.

    Parameters
    ----------
    raw_data_path:
        Absolute path to the raw data file (.dta, .csv, or .xlsx).
    spec:
        Regression specification dict produced by ``parse_do_file``.
    output_dir:
        Directory where the cleaned baseline dataset will be saved
        (e.g. ``papers/Paper_XXX/``).

    Returns
    -------
    pd.DataFrame
        Clean baseline DataFrame with all variables required by ``spec``.

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    """
    # TODO: Detect file format from extension (.dta → pyreadstat/pandas,
    #       .csv → pd.read_csv, .xlsx → pd.read_excel).
    # TODO: Load the dataset; log number of rows and columns.
    # TODO: Apply sample restrictions from spec (e.g. year filters, non-missing
    #       requirements on the dependent variable).
    # TODO: Construct any derived variables mentioned in the DO file
    #       (log transforms, interaction terms, winsorisation).
    # TODO: Verify all variables in spec["indepvars"] + spec["absorb"] exist.
    # TODO: Save the clean baseline as a parquet file to output_dir/baseline.parquet.
    # TODO: Return the baseline DataFrame.
    raise NotImplementedError("prepare_baseline is not yet implemented.")
