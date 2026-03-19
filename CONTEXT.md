# CONTEXT.md — MAR Missing-Data Simulation Project

## 1. Project Overview

This project replicates the main regressions from 10 empirical management/economics papers,
simulates Missing-At-Random (MAR) missingness at 7 different proportions, and measures how
listwise deletion changes key regression coefficients relative to the baseline (complete-data) result.

**Research Question**: How sensitive are published regression estimates to MAR missingness at
varying proportions (1%, 5%, 10%, 20%, 30%, 40%, 50%), when the missing data is handled by
listwise deletion?

---

## 2. Pipeline Architecture

The pipeline is a sequential multi-agent workflow. Each paper flows through 8 agents:

```
[Parser Agent]
      ↓
[Data Prep Agent]
      ↓
[Baseline Verifier]  ← HUMAN GATE 1 (approve baseline match before proceeding)
      ↓
[Variable Selector]  ← HUMAN GATE 2 (approve key vars before generating missingness)
      ↓
[Missingness Generator]
      ↓
[Listwise Agent]
      ↓
[Regression Runner]
      ↓
[QC Agent]           ← HUMAN GATE 3 (approve QC results before finalising)
```

Orchestration is handled by `pipeline/orchestrator.py`.

---

## 3. Missingness Mechanism (MAR)

**Type**: Missing At Random (MAR) — missingness in the key variable(s) depends on observed
auxiliary variables, not on the missing values themselves.

**Implementation**:
1. Select an auxiliary variable correlated with the key variable(s).
2. Compute a logistic missingness probability: `P(missing) = sigmoid(MAR_STRENGTH * z_score(aux_var))`.
3. Draw Bernoulli indicators and set values to NaN accordingly.
4. Repeat for each of the 7 target proportions by adjusting the intercept via binary search.

**Parameter**: `MAR_STRENGTH = 1.5` (controls how strongly the auxiliary variable predicts missingness).

---

## 4. Target Missingness Proportions

| Label  | Proportion |
|--------|-----------|
| 01pct  | 1%        |
| 05pct  | 5%        |
| 10pct  | 10%       |
| 20pct  | 20%       |
| 30pct  | 30%       |
| 40pct  | 40%       |
| 50pct  | 50%       |

---

## 5. Key Variables Strategy

For each paper, 3–5 key variables are selected from the main regression:
- **Priority**: main independent variable(s) of interest, then key controls.
- **Criteria**: continuous or ordinal variables (not binary dummies, not fixed-effect identifiers).
- **Source**: parsed from the DO file / replication code by the Parser Agent, confirmed by human review.

Parameters: `MIN_KEY_VARS = 3`, `MAX_KEY_VARS = 5`.

---

## 6. Listwise Deletion

After MAR missingness is introduced, listwise deletion is applied: any observation with a missing
value in any of the key variables is dropped. The remaining dataset is then used to re-run the
original regression specification.

---

## 7. Outcome Metrics

For each paper × proportion combination, the pipeline records:
- Coefficient estimate (β) on the main independent variable
- Standard error
- t-statistic / p-value
- Significance stars
- Sample size (N) after listwise deletion
- % change in N vs. baseline
- % change in β vs. baseline

---

## 8. Output Files

| File | Description |
|------|-------------|
| `papers/Paper_XXX/regression_results.xlsx` | Per-paper results across all 7 proportions |
| `papers/Paper_XXX/paper_info.xlsx` | Spec sheet: DO file path, key vars, aux var, estimator |
| `outputs/combined_results.xlsx` | Cross-paper summary (generated at end) |
| `logs/` | Per-run logs from each agent |

---

## 9. Paper Inventory (Audit Results)

### Summary

| # | Short Name | Folder | DO File | Estimator | Fixed Effects | Clustering | Status |
|---|-----------|--------|---------|-----------|--------------|-----------|--------|
| 1 | Mapping2024 | `0005 - Mapping entrepreneurial inclusion...` | `mapping_entrepreneurial_inclusion.do` | OLS | state + MSA | MSA | Ready |
| 2 | StatusConsensus | `0017` | **NONE** (R script: `movies.R`) | SUR/OLS in R | — | — | ⚠ No DO file |
| 3 | P2P2018 | `P2P2018 - Pathways to Profits` | **NONE** | Unknown | — | — | ⚠ No replication code |
| 4 | Meyer2024 | `SMJ 2024 - Competing for attention...` | `SMJ_Final_Do.do` | `reghdfe` | metaID + monthtime | metaID | Ready |
| 5 | SMJ3560 | `SMJ 3560` | `STATA CODE 3560.do` | `xtreg fe` | metaID (panel) | — | Ready |
| 6 | Christensen2021 | `christensen-et-al-2021-hedging-on-the-hill...` | `estimate.do` | `reghdfe` | gvkey + year | gvkey | Ready |
| 7 | Fang2022 | `fang-et-al-2022-anticorruption...` | `replicationcode.do` | `reghdfe` | firm + year | firm | Ready |
| 8 | Hu2025 | `hu-et-al-2025-reshaping-corporate-boards...` | **NONE** | Unknown | — | — | ⚠ No replication code |
| 9 | IPOFirms | `The Effect of Significant IPO Firms on Industry` | `code_tables.do` | OLS | fyear + ff48 | gvkey | Ready |
| 10 | AntiDiscrimination | `replication_pakacge` | `Accepted_paper_code_to_ms.do` | `reghdfe` | cm_id + fyear | cm_id | Ready |

### Gaps / Issues

- **Paper 2 (StatusConsensus / 0017)**: No Stata DO file. Replication code is an R script (`movies.R`).
  The pipeline will need an R-based parser and runner, or manual translation to Python/Stata.
- **Paper 3 (P2P2018)**: No replication code found at all. Manual specification required.
- **Paper 8 (Hu2025)**: No replication code found at all. Manual specification required.

### Data Sizes

| Paper | Data File(s) | Size |
|-------|-------------|------|
| Mapping2024 | `.dta` | 5.2 MB |
| StatusConsensus | 2 CSV files | 26 MB + 1.3 MB |
| P2P2018 | `.csv` + `.dta` | 1.3 MB |
| Meyer2024 | `.dta` | 1.4 MB |
| SMJ3560 | `.dta` | 84 KB |
| Christensen2021 | 2 `.dta` files | 16 MB + 153 MB |
| Fang2022 | `.csv` | 3.8 MB |
| Hu2025 | `.csv` + `.dta` | 11 MB + 9.5 MB |
| IPOFirms | 3 `.xlsx` files | — |
| AntiDiscrimination | 2 `.dta` files | 4 MB + 1.6 MB |

---

## 10. Reproducibility

- **Random seed**: `RANDOM_SEED = 2026` (set in `pipeline/config.py`, passed to all agents).
- All intermediate datasets saved to `papers/Paper_XXX/missing/` and `papers/Paper_XXX/listwise/`.
- All logs written to `logs/`.
- Environment: Python 3.10+, pandas, numpy, statsmodels, linearmodels, openpyxl.
  Papers 6, 7, 10 use `reghdfe` equivalents (absorbed FE via `absorb` in linearmodels or via `PanelOLS`).
