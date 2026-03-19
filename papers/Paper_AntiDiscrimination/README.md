# Paper_AntiDiscrimination — Anti-Discrimination Laws (Replication Package)

**Source folder**: `replication_pakacge`

## Audit Notes

| Field | Value |
|-------|-------|
| DO file | `Accepted_paper_code_to_ms.do` (19 KB) |
| Estimator | `reghdfe` |
| Fixed Effects | cm_id + fyear |
| Clustering | cm_id |
| Data | 2 `.dta` files (~4 MB + 1.6 MB) |
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
