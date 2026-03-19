# Paper_Hu2025 — Reshaping Corporate Boards through Mandatory Gender Diversity Disclosures (Hu et al. 2025)

**Source folder**: `hu-et-al-2025-reshaping-corporate-boards...`

## Audit Notes

| Field | Value |
|-------|-------|
| DO file | **NONE** |
| Replication code | **NONE** |
| Estimator | Unknown |
| Fixed Effects | — |
| Clustering | — |
| Data | `.csv` + `.dta` (~11 MB + 9.5 MB) |
| Status | ⚠ No replication code |

> **WARNING**: No replication code of any kind found for this paper. The regression
> specification must be entered manually by reading the published paper and filling
> in `paper_info.xlsx` before the pipeline can proceed.

## Pipeline Checklist

- [ ] Regression spec entered manually in `paper_info.xlsx` (manual step)
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
| `paper_info.xlsx` | Spec sheet (fill in manually from paper) |
| `regression_results.xlsx` | Results across 7 missingness proportions (auto-populated) |
| `missing/` | MAR-corrupted datasets (7 parquet files) |
| `listwise/` | Listwise-deleted datasets (7 parquet files) |
