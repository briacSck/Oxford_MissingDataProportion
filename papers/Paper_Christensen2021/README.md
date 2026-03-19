# Paper_Christensen2021 — Hedging on the Hill (Christensen et al. 2021)

**Source folder**: `christensen-et-al-2021-hedging-on-the-hill...`

## Audit Notes

| Field | Value |
|-------|-------|
| DO file | `estimate.do` (18 KB) |
| Estimator | `reghdfe` |
| Fixed Effects | gvkey + year |
| Clustering | gvkey |
| Data | 2 `.dta` files (~16 MB + 153 MB) |
| Status | Ready |

> **Note**: Large dataset (153 MB). Data prep and regression steps may be slow.

## Pipeline Checklist

- [ ] DO file parsed (`parser_agent.py`)
- [ ] Baseline dataset prepared (`data_prep_agent.py`)
- [ ] Baseline regression verified — **HUMAN GATE 1**
- [ ] Key variables selected — **HUMAN GATE 2**
- [ ] MAR datasets generated (`missingness_generator.py`)
- [ ] Listwise deletion applied (`listwise_agent.py`)
- [ ] Regressions run (`regression_runner.py`)
- [ ] QC passed — **HUMAN GATE 3**

## Files

| File | Description |
|------|-------------|
| `paper_info.xlsx` | Spec sheet (fill in depvar, main_coef, data_file, published_coef_main) |
| `regression_results.xlsx` | Results across 7 missingness proportions (auto-populated) |
| `missing/` | MAR-corrupted datasets (7 parquet files) |
| `listwise/` | Listwise-deleted datasets (7 parquet files) |
