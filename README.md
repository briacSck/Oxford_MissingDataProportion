# Oxford_MissingDataProportion

Research pipeline measuring how MAR missingness affects published regression estimates when handled by listwise deletion.

---

## Research Question

> How sensitive are published regression estimates to Missing At Random (MAR) missingness at varying proportions (1%, 5%, 10%, 20%, 30%, 40%, 50%) when missing data is handled by listwise deletion?

The pipeline replicates baseline regressions from 10 empirical management/economics papers, injects MAR missingness at multiple target proportions, applies listwise deletion, and re-estimates the original specification to measure changes in coefficients, standard errors, significance, and sample size.

---

## Pipeline Architecture

The pipeline is a sequential multi-agent workflow. Orchestration is handled by `pipeline/orchestrator.py`.

```
[Parser Agent]
      ↓
[Data Prep Agent]
      ↓
[Baseline Verifier]  ← GATE 1: approve baseline match before proceeding
      ↓
[Variable Selector]  ← GATE 2: approve key variables before generating missingness
      ↓
[Missingness Generator]
      ↓
[Listwise Agent]
      ↓
[Regression Runner]
      ↓
[QC Agent]           ← GATE 3: approve QC results before finalising
```

| Stage | Module | Responsibility |
|-------|--------|---------------|
| Parser Agent | `pipeline/parser_agent.py` | Parse DO file / replication code; extract spec |
| Data Prep Agent | `pipeline/data_prep_agent.py` | Load `.dta`/`.csv`; build `baseline.parquet` |
| Baseline Verifier | `pipeline/baseline_verifier.py` | Re-run baseline; compare to published coefficients |
| Variable Selector | `pipeline/variable_selector_agent.py` | Select 3–5 key vars for MAR simulation |
| Missingness Generator | `pipeline/missingness_generator.py` | Apply power-law MAR at 7 proportions |
| Listwise Agent | `pipeline/listwise_agent.py` | Drop observations with any missing key var |
| Regression Runner | `pipeline/regression_runner.py` | Re-estimate spec on each listwise dataset |
| QC Agent | `pipeline/qc_agent.py` | Validate results; write `qc_report.txt` |

Human review gates are implemented as LLM-validated approval steps in `pipeline/orchestrator.py`. The pipeline halts with `HALT_GATE_N` status if a gate does not approve.

---

## MAR Simulation Design

**Mechanism**: Missing At Random — missingness in key variables depends on an observed auxiliary variable, not on the missing values themselves.

**Formula**: For each key variable and each target proportion `p`:

1. Select an auxiliary variable correlated with the key variable (preferably non-negative).
2. Compute raw probability: `prob_i ∝ aux_i ^ 1.5`
3. Rescale via binary search on a multiplicative scalar (tolerance ±0.001) so the realised missing fraction equals `p` exactly.
4. Draw Bernoulli indicators; set selected values to `NaN`.
5. Repeat for all 7 proportions.

**Key parameters** (defined in `pipeline/config.py`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MAR_STRENGTH` | `1.5` | Exponent in the power-law formula |
| `RANDOM_SEED` | `2026` | Set before every Bernoulli draw |
| `MIN_KEY_VARS` | `3` | Minimum key variables per paper |
| `MAX_KEY_VARS` | `5` | Maximum key variables per paper |

**Target proportions**: 1%, 5%, 10%, 20%, 30%, 40%, 50% (labels: `01pct` … `50pct`).

---

## Repository Structure

```
Oxford_MissingDataProportion/
├── pipeline/               # All pipeline agent modules
│   ├── config.py           # Shared constants (seed, proportions, column names)
│   ├── orchestrator.py     # Main pipeline driver; handles gates
│   ├── parser_agent.py
│   ├── data_prep_agent.py
│   ├── baseline_verifier.py
│   ├── variable_selector_agent.py
│   ├── missingness_generator.py
│   ├── listwise_agent.py
│   ├── regression_runner.py
│   └── qc_agent.py
├── scripts/
│   ├── run_batch.py        # Run the full 7-paper batch
│   └── update_data_paths.py
├── papers/
│   └── <Paper_Folder>/     # One folder per paper (10 total)
│       ├── spec.json           # Parsed specification (tracked)
│       ├── selection.json      # Approved key variables (tracked)
│       ├── qc_report.txt       # QC agent output (tracked)
│       ├── paper_info.xlsx     # Extracted spec metadata (gitignored)
│       ├── regression_results.xlsx  # Per-paper results (gitignored)
│       ├── missing/            # MAR-corrupted datasets (gitignored)
│       └── listwise/           # Post-deletion datasets (gitignored)
├── outputs/                # Cross-paper summaries (gitignored)
├── logs/                   # Per-run agent logs (gitignored)
├── CONTEXT.md              # Detailed project reference document
├── requirements.txt
└── .env                    # API keys — gitignored, never commit
```

---

## Paper Coverage

| # | Short Name | Estimator | Fixed Effects | Clustering | Pipeline Status |
|---|-----------|-----------|--------------|-----------|----------------|
| 1 | Mapping2024 | OLS | state + MSA | MSA | SUCCESS |
| 2 | Fang2022 | `reghdfe` | firm + year | firm | SUCCESS |
| 3 | IPOFirms | OLS | fyear + ff48 | gvkey | SUCCESS |
| 4 | AntiDiscrimination | `reghdfe` | cm_id + fyear | cm_id | SUCCESS |
| 5 | Meyer2024 | `reghdfe` | metaID + monthtime | metaID | HALT_GATE_2 |
| 6 | SMJ3560 | `xtreg fe` | metaID (panel) | — | HALT_GATE_2 |
| 7 | Christensen2021 | `reghdfe` | gvkey + year | gvkey | FE_EXCEPTION |
| 8 | StatusConsensus | SUR/OLS | — | — | No pipeline support (R-only) |
| 9 | P2P2018 | Unknown | — | — | No pipeline support (no replication code) |
| 10 | Hu2025 | Unknown | — | — | No pipeline support (no replication code) |

---

## Current Batch Status

Latest verified run on the 7-paper automated batch:

| Status | Count | Papers | Blocker |
|--------|-------|--------|---------|
| SUCCESS | 4 | AntiDiscrimination, Fang2022, IPOFirms, Mapping2024 | — |
| HALT_GATE_2 | 2 | SMJ3560, Meyer2024 | SMJ3560: no valid key vars found; Meyer2024: repair candidates conceptually invalid for MAR |
| FE_EXCEPTION | 1 | Christensen2021 | Dependent variable `intravol` / FE column `date` absent from baseline dataset |
| No support | 3 | StatusConsensus, P2P2018, Hu2025 | R-only or no replication code |

---

## Setup

**1. Clone and create a Python environment:**

```bash
git clone <repo-url>
cd Oxford_MissingDataProportion
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `statsmodels`, `linearmodels`, `openpyxl`, `python-dotenv`, `anthropic`.

**3. Configure API keys:**

Copy `.env.example` to `.env` and fill in your Anthropic API key:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-...
```

**4. Place source data:**

Source data files (`.dta`, `.csv`) must be present in each paper's folder under `RA Missing data task/` (this folder is gitignored). Update data paths with:

```bash
python scripts/update_data_paths.py
```

---

## How to Run

**Run the full batch (all 7 supported papers):**

```bash
python scripts/run_batch.py
```

**Run a single paper:**

```python
from pipeline.orchestrator import run_paper
run_paper("papers/0005 - Mapping entrepreneurial inclusion...")
```

**Workflow:**

1. Pipeline halts at Gate 1 for baseline approval — inspect the printed comparison and confirm.
2. Pipeline halts at Gate 2 for variable selection — review proposed key vars and confirm.
3. After Gate 2, missingness generation, listwise deletion, and regression re-estimation run automatically.
4. Pipeline halts at Gate 3 for QC approval — review `qc_report.txt` and confirm.
5. Results written to `papers/<folder>/regression_results.xlsx`.

---

## Output Files

**Per-paper** (in `papers/<folder>/`):

| File | Description |
|------|-------------|
| `spec.json` | Parsed specification (estimator, dep var, key vars, FE, cluster) |
| `selection.json` | Approved key variables and auxiliary variable |
| `regression_results.xlsx` | Results across all 7 proportions |
| `qc_report.txt` | QC agent validation report |

**`regression_results.xlsx` columns** (defined in `pipeline/config.py`):

| Column | Description |
|--------|-------------|
| `paper` | Paper short name |
| `proportion_label` | e.g. `05pct` |
| `proportion_value` | e.g. `0.05` |
| `variable` | Key variable name |
| `coef_baseline` / `coef_listwise` | Coefficient estimate (baseline vs. listwise) |
| `se_baseline` / `se_listwise` | Standard error |
| `tstat_baseline` / `tstat_listwise` | t-statistic |
| `pvalue_baseline` / `pvalue_listwise` | p-value |
| `stars_baseline` / `stars_listwise` | Significance stars (`*` / `**` / `***`) |
| `n_baseline` / `n_listwise` | Sample size |
| `pct_change_n` | % change in N vs. baseline |
| `pct_change_coef` | % change in coefficient vs. baseline |

**Cross-paper** (in `outputs/`): `combined_results.xlsx` — aggregated results across all successful papers.

---

## Reproducibility

- **Random seed**: `RANDOM_SEED = 2026`, set before every Bernoulli draw in the missingness generator.
- **MAR scalar tolerance**: binary search converges to ±0.001 of the target proportion.
- All intermediate datasets saved to `papers/<folder>/missing/` and `papers/<folder>/listwise/`.
- Fixed-effects estimation: `reghdfe`-equivalent papers (Fang2022, Christensen2021, AntiDiscrimination, Meyer2024) use `linearmodels.PanelOLS` with absorbed FE or equivalent.

---

## Known Limitations and Next Steps

**Current limitations:**

- Not yet a full 10-paper end-to-end automated replication.
- `Christensen2021`: baseline spec mismatch — `intravol` / `date` column resolution needed.
- `SMJ3560` and `Meyer2024`: halted at Gate 2; require revised variable selection or manual override.
- `StatusConsensus`: replication code is R-only (`movies.R`); no Python/Stata translation yet.
- `P2P2018`, `Hu2025`: no replication code available; require manual specification entry.

**Priority next steps:**

1. Resolve `Christensen2021` column-name mismatch in baseline verifier.
2. Investigate `SMJ3560` key variable candidates manually; consider relaxing selection criteria.
3. Translate `StatusConsensus` R regression to Python.
4. Source replication code for `P2P2018` and `Hu2025`.
5. Add smoke tests for the 4 successful papers to prevent regressions.
