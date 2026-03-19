# Paper_IPOFirms — The Effect of Significant IPO Firms on Industry

**Source folder**: `The Effect of Significant IPO Firms on Industry`

## Audit Notes

| Field | Value |
|-------|-------|
| DO file | `code_tables.do` (35 KB) |
| Estimator | OLS |
| Fixed Effects | fyear + ff48 |
| Clustering | gvkey |
| Data | 3 `.xlsx` files |
| Status | Ready |

> **Note**: Data is in Excel format (.xlsx). Data prep agent must use `pd.read_excel`.
> Variables may require winsorisation as described in the DO file.

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
