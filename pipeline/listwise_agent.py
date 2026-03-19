"""
pipeline/listwise_agent.py
---------------------------
Listwise Agent: applies listwise deletion (complete-case analysis) to each
MAR-corrupted dataset, dropping any observation with a NaN in any of the key
variables, and saves the resulting clean datasets.
"""

from __future__ import annotations


def apply_listwise_deletion(
    missing_dir: str,
    output_dir: str,
) -> list[str]:
    """Apply listwise deletion to all parquet files in missing_dir.

    Parameters
    ----------
    missing_dir:
        Directory containing MAR-corrupted parquet files
        (one per proportion label, e.g. ``papers/Paper_XXX/missing/``).
    output_dir:
        Directory where listwise-deleted datasets will be saved
        (e.g. ``papers/Paper_XXX/listwise/``).

    Returns
    -------
    list[str]
        List of absolute paths of the saved listwise-deleted parquet files.

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    """
    # TODO: Glob all *.parquet files in missing_dir.
    # TODO: For each file:
    #   a. Load the parquet file into a DataFrame.
    #   b. Identify key_vars: columns that contain any NaN values
    #      (these were the variables corrupted by the missingness generator).
    #   c. Apply df.dropna(subset=key_vars) — listwise deletion.
    #   d. Log: proportion label, N before, N after, % dropped.
    #   e. Save the clean DataFrame to output_dir/<proportion_label>.parquet.
    # TODO: Return list of saved file paths.
    raise NotImplementedError("apply_listwise_deletion is not yet implemented.")
