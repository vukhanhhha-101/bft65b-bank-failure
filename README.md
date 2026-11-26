# BFT65B – A Big-Data Pipeline for Bank-Failure Early Warning (FDIC)

| | |
|---|---|
| **Student** | Vũ Khánh Hà |
| **Student ID** | 11230955 |
| **Course** | BFT65B – Big Data Analytics in Finance, National Economics University |
| **Repository** | https://github.com/vukhanhhha-101/bft65b-bank-failure |
| **Data** | FDIC BankFind public API (https://api.fdic.gov/banks), downloaded on <DOWNLOAD_DATE> |
| **Environment** | Google Colab, Python 3.13, Java 21 (PySpark); exact package versions in `requirements.txt`; random seed 42 |

## 1. What this project answers

The project builds a reproducible big-data pipeline on public FDIC data (US) as a prototype for a bank-failure early-warning
system. Most of the effort goes into the data pipeline, not into model accuracy. It answers the three questions of the exam:

* **Q1 – storage and processing.** How should four decades of bank reports be stored and processed on a laptop, and when would a Spark cluster be needed?
* **Q2 – data quality.** Which quality problems do the raw data contain, and how does the pipeline treat them?
* **Q3 – validation.** Can the pipeline output support an honestly validated early-warning model?

## 2. Data

| Table | Content | Size used |
|---|---|---|
| Financials | quarterly Call Reports, 1984Q1–2026Q2, 33 fields (3 keys + 30 Call Report fields) | 170 quarters, 1,682,738 rows (one Parquet file per quarter; rows equal the API total for every quarter; no duplicate CERT–REPDTE) |
| Failures | FDIC failure and assistance records | 4,117 records (RESTYPE = FAILURE or ASSISTANCE) |
| Institutions | bank reference data (ACTIVE, CHANGEC1, …) | 27,838 records |

Amounts are in thousands of USD and ratios in percent. Download method: the course starter script `src/fdic_download_starter.py`
(resumable, one file per quarter, manifest with API totals). `src/01b_verify_small_tables.py` extends the completeness check to the
Failures and Institutions tables. The data files are not committed (`.gitignore`).

## 3. Repository structure

```text
bft65b-bank-failure/
├── BFT65B_pipeline.ipynb        notebook that runs the whole project and displays every result
├── README.md   requirements.txt   run_all.sh   .gitignore
├── src/
│   ├── fdic_download_starter.py   course starter (unchanged)
│   ├── config.py                  paths, seed, dates, split, cost assumption
│   ├── 01b_verify_small_tables.py completeness check of Failures / Institutions
│   ├── 02_storage.py              schema + storage-layout benchmark              (Part 1)
│   ├── 03_quality.py              automated data-quality report                  (Part 2)
│   ├── 04_panel.py                clean bank-quarter panel, exit types, label    (Part 2)
│   ├── 04b_label_sensitivity.py   failure vs assistance label sensitivity        (Part 2)
│   ├── 05_benchmark.py            Polars vs pandas, execution plans              (Part 3)
│   ├── 06_spark.py                PySpark reproduction, plans, partitions        (Part 3)
│   ├── 06b_spark_scaleup.py       Polars vs Spark scale-up                       (Part 3)
│   ├── 07_features.py             point-in-time feature table + tests            (Part 4)
│   ├── 08_model.py                validation, thresholds, single test evaluation (Part 5)
│   ├── 08b_lag_sensitivity.py     publication-lag sensitivity on development data(Part 5)
│   ├── 09_mllib.py                Spark MLlib demo (development data only)       (Part 5)
│   ├── 10_dev_diagnostics.py      development-only diagnostics                   (Part 5)
│   └── 11_test_descriptives.py    descriptive tables and paired bootstrap        (Part 5)
├── reports/                      tables and figures produced by the scripts (all numbers in the report come from here)
└── tools/check_submission.py     pre-submission audit
```

## 4. Design

### 4.1 Storage (Part 1)
Fixed schema (CERT Int32, REPDTE Date, amounts Float64). Five layouts are compared on the full panel (size, write time, two typical queries; best of 3 runs):

| Layout | Size (MB) | Query A: 2 columns, all years (s) | Query B: one year, all columns (s) |
|---|---|---|---|
| CSV | 446.4 | 3.408 | 3.469 |
| Parquet, uncompressed | 395.4 | 0.097 | 0.386 |
| Parquet, snappy | 209.6 | 0.102 | 0.702 |
| Parquet, zstd-3 | 149.2 | 0.117 | 1.054 |
| **Parquet, zstd-3, partitioned by year (chosen)** | 156.3 | 0.296 | 0.086 |

The chosen layout is about 2.9x smaller than CSV and extracts one year about 40x faster; all-years scans are slower than a single zstd file, which is the price of partition pruning (see `reports/storage_benchmark.csv`).

### 4.2 Data quality and panel (Part 2)
`03_quality.py` checks completeness against API totals, missing values by decade, 12 validity rules, whether flow variables are year-to-date,
gaps in bank histories, extreme values, exit types and the Failures table. `04_panel.py` builds the clean panel: de-duplication, de-cumulation of
year-to-date flows (NETINC, INTINC, EINTEXP, NONII, NONIX, ELNATR), removal of reports dated on/after the failure date, exit classification
(failure / other exit / active; the exit type is never used as a feature) and the label.

### 4.3 Label
`y(i,q) = 1` if the earliest FAILURE date of bank *i* lies in (REPDTE of quarter *q*, REPDTE + 1 year]; otherwise 0. Only records with `RESTYPE = "FAILURE"` count as failures; assistance transactions are not failures (sensitivity in `reports/label_sensitivity.csv`);
mergers and other exits get `y = 0`. Reports dated on/after the failure date are removed. Rows with a report date up to 2025-06-30 are used, because later labels are incomplete.

### 4.4 Development / test split
Report quarters: development 1984Q1–2006Q4, gap 2007Q1–2008Q4 (8 quarters), test 2009Q1–2014Q4 (469 failed banks in the test period, 2,290 in development).
The test period follows the development period with a gap longer than the required four quarters and contains far more than 50 failures. Tuning, feature choices and the threshold use the development period only.

### 4.5 Features (Part 4)
31 point-in-time features in CAMELS groups: capital (equity ratio, regulatory capital ratios), asset quality (non-performing, past-due and non-accrual ratios, OREO, reserves, Texas ratio),
earnings (ROA, ROE, margin, quarterly ROA, provisions, cost-to-income), liquidity and funding, loan mix and growth, size, and 4-quarter changes. Lags are quarter-aware (a lag is `null` if the bank has a gap).
Flow variables are de-cumulated; a truncation test recomputes the features with data only up to three cut-off quarters and requires identical values (`reports/pit_test.txt`).
**Important limitation:** in the main results the report of quarter *q* is used to predict failures after the end of quarter *q*, although a Call Report is published about 30–45 days later. The publication lag is therefore **not** enforced in the main results.
`src/08b_lag_sensitivity.py` quantifies the effect on development data: excluding positive rows whose failure falls within 45 days of the report removes 494 of 4,214 positive rows (11.7%) and lowers the pooled out-of-fold PR-AUC of the gradient-boosting model from 0.342 to 0.309 (logit: 0.287 to 0.254) (`reports/dev_lag_sensitivity.csv`).
The performance figures should therefore be read as optimistic.

### 4.6 Modelling and validation (Part 5)
* **Models:** Texas-ratio rule, baseline logistic regression on 4 ratios, logistic regression on all features, HistGradientBoosting (max depth 4, 200 iterations). All preprocessing (winsorising, imputation, scaling) is inside a scikit-learn `Pipeline` fitted on training data only.
* **Validation:** expanding-window cross-validation on the development period, four folds (1989Q1–1991Q4, 1992Q1–1994Q4, 1995Q1–2000Q4, 2001Q1–2006Q4), purge of 5 quarters so that no training label window overlaps a validation period.
* **Metrics:** PR-AUC (primary), ROC-AUC, recall, precision, alarm rate and expected cost; accuracy is not used. Test confidence intervals use a bank-clustered bootstrap.
* **Alarm threshold:** minimises expected cost on development out-of-fold predictions. The cost of a missed failure comes from the data: the mean `COST` of failures in the development period (54,252 thousand USD per failed bank, 2,393 observations; 3.80 positive rows per failed bank), divided over the positive rows per failed bank. The cost of a false alarm is an **assumption** (50 thousand USD per review, `C_FALSE_ALARM_K` in `src/config.py`); `reports/threshold_sensitivity.csv` shows the thresholds for 10–500 thousand USD.
* **Spark MLlib:** `09_mllib.py` trains logistic regression and GBT on development data only.
* **Test protocol:** `python src/08_model.py --stage test` evaluates the test period once and writes `reports/TEST_USED.flag`, which blocks a second run on the same machine. The flag is not committed (`.gitignore`), so `run_all.sh` on a fresh clone runs the test stage once. The test period was evaluated once with this pipeline version.

## 5. Main measured results

| Topic | Result |
|---|---|
| Storage | zstd partitioned by year: 156 MB vs 446 MB CSV; one-year query 0.086 s vs 3.469 s |
| Polars vs pandas (1.7 M rows, median of 3 runs) | aggregation 0.32 s (lazy) vs 0.68 s; window features 0.86 s vs 2.67 s; label join 0.50 s vs 1.47 s; streaming lowers peak memory further (`reports/benchmark.csv`) |
| Scale-up | the same tasks were replicated x4 and x8 (13.5 M rows) on one machine (`reports/benchmark_scaleup.csv`) |
| PySpark vs Polars | aggregation, window and label-join results match (`reports/spark_match.txt`); execution plans in `reports/spark_plan_*.txt`; 16 shuffle partitions were faster than 200 for the aggregation (1.8 s vs 6.1 s) |
| Development CV (pooled out-of-fold PR-AUC) | Texas rule 0.297, baseline logit 0.202, logit (all features) 0.287, HistGradientBoosting 0.342 (`reports/cv_pooled.csv`) |
| Test period | see `reports/test_results.csv`, `reports/test_bank_level.csv`, `reports/test_paired_bootstrap.csv` |

## 6. How to run

**Google Colab (as developed).**
```python
from google.colab import drive; drive.mount("/content/drive")
!git clone https://github.com/vukhanhhha-101/bft65b-bank-failure.git
%cd bft65b-bank-failure
!pip install -q -r requirements.txt
!bash run_all.sh
```
**Local machine.** Python 3.11+ (developed with 3.13), Java 17+ for PySpark, Git Bash / WSL / Linux / macOS for `run_all.sh`:
```bash
git clone https://github.com/vukhanhhha-101/bft65b-bank-failure.git && cd bft65b-bank-failure
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```
A quick connectivity test: `python src/fdic_download_starter.py --start 2023-03-31 --end 2023-12-31 --out data_raw/smoke`.
The full download takes about one hour and is resumable (re-run the same command). `BFT65B_pipeline.ipynb` runs the same scripts step by step
and skips steps whose output already exists (`FORCE = True` recomputes). Before submitting, `python tools/check_submission.py BFT65B_pipeline.ipynb .` checks script references, saved errors, version pins and `.gitignore`.

Generated folders: `data_raw/` (downloads and manifest) and `data_proc/` (processed Parquet files) are created locally and are not committed.

## 7. Limitations
1. **Publication lag not enforced** (section 4.5): the main performance figures use information that would not yet have been published at the decision date. The sensitivity analysis shows a drop of about 3 PR-AUC points on development data.
2. The Texas-ratio rule is compared by PR-AUC/ROC-AUC; no cost-based alarm threshold is reported for it in the test table.
3. Data are the latest (restated) Call Report values, not the values as first filed.
4. Assistance transactions are not counted as failures; 485 FAILURE records have no CERT (all before 1984, outside the panel) and cannot be linked.
5. The false-alarm cost is an assumption (sensitivity table provided); one crisis (2009–2014) is used as the test period; rows of the same bank are correlated (bank-clustered bootstrap is used for the intervals).
6. The MLlib demo is untuned and runs on development data only; benchmarks come from one Colab machine, and Spark in local mode is only a proxy for a cluster.
7. For Vietnam: supervisory data are not public, filing lags and failure histories differ and failures are far fewer, so the cost-based threshold and the validation design would have to be re-estimated locally.

## 8. AI assistance and sources

Claude (Anthropic) was used to help draft and review parts of the pipeline code, the notebook and this documentation. <AI_SELF_DESCRIPTION: state exactly what you wrote, changed and verified yourself.>
The author ran the pipeline, inspected the outputs and is responsible for the content.

Other sources: the course starter script `fdic_download_starter.py` (BFT65B, NEU); FDIC BankFind API documentation (https://api.fdic.gov/banks/docs/); the documentation of Polars, pandas, scikit-learn and PySpark.
