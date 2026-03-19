"""
pipeline/missingness_generator.py
----------------------------------
Missingness Generator: injects MAR missingness into the key variables at each
of the 7 target proportions and saves one dataset per proportion.

MAR mechanism
~~~~~~~~~~~~~
For each target proportion p:
  1. Compute z = (aux_var - mean) / std  (standardise the auxiliary variable).
  2. Define logistic probability: P(missing | z) = sigmoid(intercept + MAR_STRENGTH * z).
  3. Find ``intercept`` by binary search so that mean(P(missing)) ≈ p.
  4. Draw Bernoulli(P(missing_i)) indicators for each observation.
  5. Set key_var to NaN where indicator == 1.
  6. Save dataset to output_dir/<proportion_label>.parquet.
"""

from __future__ import annotations

import pandas as pd


def generate_mar_datasets(
    baseline_df: pd.DataFrame,
    key_vars: list[str],
    aux_var: str,
    output_dir: str,
) -> dict[str, str]:
    """Generate one MAR-corrupted dataset per target proportion.

    Parameters
    ----------
    baseline_df:
        Clean baseline DataFrame (no missing values in key_vars).
    key_vars:
        List of variable names to receive MAR missingness (3–5 variables).
    aux_var:
        Name of the auxiliary variable used as the MAR predictor.
    output_dir:
        Directory where the corrupted datasets will be saved
        (e.g. ``papers/Paper_XXX/missing/``).

    Returns
    -------
    dict[str, str]
        Mapping of proportion label → absolute path of saved parquet file,
        e.g. ``{"01pct": "/…/missing/01pct.parquet", …}``.

    Raises
    ------
    NotImplementedError
        Until the agent is fully implemented.
    ValueError
        If aux_var is in key_vars (aux_var must remain fully observed).
    """
    # TODO: Import config values: MISSING_PROPORTIONS, PROPORTION_LABELS,
    #       MAR_STRENGTH, RANDOM_SEED.
    # TODO: Validate inputs:
    #         - All key_vars present in baseline_df.columns.
    #         - aux_var present and NOT in key_vars.
    #         - key_vars are free of NaNs in baseline_df (baseline must be clean).
    # TODO: Standardise aux_var: z = (aux_var - mean) / std.
    # TODO: For each (proportion, label) in zip(MISSING_PROPORTIONS, PROPORTION_LABELS):
    #   a. Binary-search for intercept b0 such that
    #      mean(sigmoid(b0 + MAR_STRENGTH * z)) ≈ proportion (tolerance 1e-4).
    #      Search bounds: b0 ∈ [-20, 20]; use ~50 bisection steps.
    #   b. Compute P_missing_i = sigmoid(b0 + MAR_STRENGTH * z_i) for each row.
    #   c. Set numpy random seed to RANDOM_SEED + index for reproducibility.
    #   d. Draw missing_indicator ~ Bernoulli(P_missing_i).
    #   e. Copy baseline_df; for each var in key_vars, set rows where
    #      missing_indicator == 1 to NaN.
    #   f. Assert actual missing rate ≈ proportion (warn if |actual - target| > 0.01).
    #   g. Save to output_dir/<label>.parquet.
    #   h. Log: paper, proportion label, target rate, actual rate, N missing.
    # TODO: Return mapping dict.
    raise NotImplementedError("generate_mar_datasets is not yet implemented.")
