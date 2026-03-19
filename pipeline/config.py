"""
pipeline/config.py
------------------
Fixed parameters shared across all pipeline agents.
"""

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 2026

# ── MAR mechanism ────────────────────────────────────────────────────────────
# Strength of the logistic link between the auxiliary variable and P(missing).
# Higher values → missingness is more strongly predicted by the aux variable.
MAR_STRENGTH = 1.5

# ── Target missingness proportions ───────────────────────────────────────────
MISSING_PROPORTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
PROPORTION_LABELS   = ["01pct", "05pct", "10pct", "20pct", "30pct", "40pct", "50pct"]

# ── Variable selection constraints ───────────────────────────────────────────
MIN_KEY_VARS = 3
MAX_KEY_VARS = 5

# ── Significance stars ───────────────────────────────────────────────────────
SIGNIFICANCE_LEVELS = {0.01: "***", 0.05: "**", 0.10: "*"}

# ── Regression results table columns ─────────────────────────────────────────
REGRESSION_RESULTS_COLUMNS = [
    "paper",
    "proportion_label",
    "proportion_value",
    "variable",
    "coef_baseline",
    "coef_listwise",
    "se_baseline",
    "se_listwise",
    "tstat_baseline",
    "tstat_listwise",
    "pvalue_baseline",
    "pvalue_listwise",
    "stars_baseline",
    "stars_listwise",
    "n_baseline",
    "n_listwise",
    "pct_change_n",
    "pct_change_coef",
]
