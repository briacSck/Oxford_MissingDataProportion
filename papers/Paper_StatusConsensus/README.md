# Paper_StatusConsensus — Status and Consensus

**Source folder**: `0017`

## Audit Notes

| Field | Value |
|-------|-------|
| DO file | **NONE** |
| Replication code | R script: `movies.R` |
| Estimator | SUR / OLS (in R) |
| Fixed Effects | — |
| Clustering | — |
| Status | ⚠ No Stata DO file |

> **WARNING**: No Stata DO file exists for this paper. The only replication code is an R
> script (`movies.R`). The pipeline will need either:
> (a) An R-based parser and regression runner, or
> (b) Manual translation of the R specification to Python/Stata before proceeding.
> This paper requires manual intervention before the pipeline can run.

## Pipeline Checklist

- [ ] R script translated or R runner implemented (manual step)
- [ ] DO file parsed / spec entered manually (`paper_info.xlsx`)
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
| `paper_info.xlsx` | Spec sheet (fill in manually from movies.R) |
| `regression_results.xlsx` | Results across 7 missingness proportions (auto-populated) |
| `missing/` | MAR-corrupted datasets (7 parquet files) |
| `listwise/` | Listwise-deleted datasets (7 parquet files) |
