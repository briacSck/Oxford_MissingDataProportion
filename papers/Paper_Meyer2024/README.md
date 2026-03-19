# Paper_Meyer2024 — Competing for Attention (SMJ 2024)

**Source folder**: `SMJ 2024 - Competing for attention...`

## Audit Notes

| Field | Value |
|-------|-------|
| DO file | `SMJ_Final_Do.do` (15 KB) |
| Estimator | `reghdfe` |
| Fixed Effects | metaID + monthtime |
| Clustering | metaID |
| Data | `.dta` (~1.4 MB) |
| Status | Ready |

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
