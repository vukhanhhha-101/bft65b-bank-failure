# BFT65B – Big-Data Pipeline for Bank-Failure Early Warning

**Individual Final Project – National Economics University**

- Author: [Your full name]
- Student ID: [Your student ID]
- Data source: FDIC BankFind Suite API
- Research focus: Early warning signals of US bank failures.

## 1. Project Overview

This project develops a reproducible big-data pipeline for analyzing and predicting US bank failures using historical FDIC bank financial data.

The pipeline covers:
- Data acquisition and validation
- Data quality assessment
- Panel construction and failure labeling
- Storage and processing benchmarks
- Feature engineering
- Machine learning and model evaluation
- Point-in-time and publication-lag validation

## 2. Project Structure

```text
bft65b-bank-failure/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.sh
├── src/
├── reports/
├── docs/
├── notebooks/
├── data_raw/       (not committed)
└── data_proc/      (not committed)
```

## 3. Data Sources

The project uses:
- FDIC BankFind Suite API
- Historical bank Call Reports
- FDIC bank failure records

API: https://api.fdic.gov/banks

Raw and processed datasets are excluded from the public repository because of their size.

## 4. Data Processing Pipeline

| Stage | Script | Main purpose |
|---|---|---|
| Data verification | `01b_verify_small_tables.py` | Check missing and duplicate records |
| Storage optimization | `02_storage.py` | Compare storage layouts |
| Data quality | `03_quality.py` | Audit completeness and validity |
| Panel construction | `04_panel.py` | Build clean point-in-time panel |
| Label sensitivity | `04b_label_sensitivity.py` | Compare label definitions |
| Failure funnel | `04c_failure_funnel.py` | Reconcile failure records |
| Benchmarking | `05_benchmark.py` | Compare processing performance |
| Spark processing | `06_spark.py` | PySpark replication |
| Spark scale-up | `06b_spark_scaleup.py` | Evaluate scalability |
| Feature engineering | `07_features.py` | Construct financial predictors |
| Modelling | `08_model.py` | Train and evaluate models |
| Lag sensitivity | `08b_lag_sensitivity.py` | Examine publication lag |
| MLlib | `09_mllib.py` | Distributed ML demonstration |
| Diagnostics | `10_dev_diagnostics.py` | Development diagnostics |
| Test descriptives | `11_test_descriptives.py` | Descriptive test analysis |

## 5. Methodology

### 5.1 Failure Definition

The primary failure definition uses records with:

`RESTYPE == "FAILURE"`

ASSISTANCE-only records are not classified as failures under the primary definition.

The target variable identifies a failure occurring within the subsequent 12-month prediction horizon.

Mergers and other exits are not automatically classified as failures.

### 5.2 Point-in-Time Design

The pipeline applies a publication-lag restriction to prevent financial information from being used before it would have been available.

The configured reporting lag is three months, with a minimum publication lag requirement of 45 days.

### 5.3 Feature Engineering

Financial predictors cover:
- Capital adequacy and leverage
- Asset quality
- Loan composition
- Liquidity and deposits
- Profitability
- Four-quarter financial changes
- Historical ROA indicators

Year-to-date flow variables are de-cumulated before modelling.

## 6. Models and Development Results

Four models are evaluated:

1. Texas Ratio
2. Baseline Logistic Regression
3. Full Logistic Regression
4. HistGradientBoosting

The current development artifacts contain:
- 1,183,030 development observations
- 2,264 failed banks

### Pooled Cross-Validation Results

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Texas Ratio | 0.290535 | 0.842315 |
| Baseline Logistic Regression | 0.163185 | 0.952469 |
| Full Logistic Regression | 0.232595 | 0.943204 |
| HistGradientBoosting | 0.283979 | 0.972989 |

HistGradientBoosting achieves the highest ROC-AUC and the lowest estimated development cross-validation cost.

Texas Ratio achieves the highest pooled PR-AUC.

This illustrates that model performance depends on the evaluation metric.

## 7. Data Quality and Validation

The current pipeline has produced the following validation results:

- 1,682,738 raw financial panel observations
- 24,064 banks in the raw panel
- Zero duplicate `(CERT, REPDTE)` records
- Zero quarters with API row-count mismatches
- Point-in-time tests passed at three cut-offs
- Publication-lag test passed with a minimum observed gap of 90 days

The current feature dataset contains:
- 1,656,978 observations
- 31 explanatory features
- 38 columns including identifiers and labels

The four-quarter ROA mean is calculated only when four consecutive quarterly observations are available.

## 8. Test Evaluation Status

An earlier test evaluation was conducted on November 21, 2026, before subsequent publication-lag and panel updates.

The earlier results are archived separately under:

`reports/archive_pre_lag_fix/`

They are not presented as final test results for the current pipeline.

The current development artifacts were generated after the pipeline updates. An independent test evaluation for this updated version has not been established.

The test-use flag is retained to preserve the evaluation history and prevent accidental reuse.

## 9. Reproducibility

### Verified development environment

The project was developed and tested in the following environment:

- Python 3.13.18
- OpenJDK 21.0.12.1
- NumPy 2.4.6
- Pandas 2.2.3
- Polars 1.35.2
- Scikit-learn 1.6.1
- PySpark 4.0.5
- PyArrow 23.0.1
- Joblib 1.6.0
- Requests 2.32.4
- Matplotlib 3.10.0
- Psutil 5.9.5

Install dependencies using:

```bash
pip install -r requirements.txt
```

The versions in `requirements.txt` correspond to the verified Google Colab environment.

To reproduce the project:

1. Set up a compatible Python and Java environment.
2. Install dependencies using `requirements.txt`.
3. Download the required FDIC data.
4. Execute the processing stages in sequence.
5. Review the generated reports.
6. Run the development modelling stage.

Test evaluation is intentionally excluded from the standard development workflow.

See `run_all.sh` for the development pipeline. Review its commands before execution.

## 10. Reports

Important outputs are stored in `reports/`:

| Report | Description |
|---|---|
| `dq_report.md` | Data quality assessment |
| `dq_overview.png` | Data quality visualization |
| `panel_summary.json` | Panel construction summary |
| `failure_funnel.csv` | Failure-record reconciliation |
| `label_sensitivity.csv` | Label sensitivity analysis |
| `cv_scores.csv` | Cross-validation by fold |
| `cv_pooled.csv` | Pooled cross-validation metrics |
| `dev_summary.json` | Development summary |
| `threshold_sensitivity.csv` | Threshold sensitivity |
| `pit_test.txt` | Point-in-time validation |

## 11. Data and Version Control

Large raw datasets, processed Parquet files and model artifacts are not committed to the public repository.

The repository retains source code and selected reports for transparency and reproducibility.

## 12. References and AI Assistance Disclosure

**Data source:** Federal Deposit Insurance Corporation (FDIC), BankFind Suite API.

https://api.fdic.gov/banks

AI tools assisted with code review, debugging and documentation. The author reviewed, executed and adapted the project code and is responsible for the final implementation and conclusions.

The final disclosure should accurately identify the tools used and the nature of their assistance.
