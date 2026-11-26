# BFT65B – Big-Data Pipeline for Bank-Failure Early Warning

**Course:** BFT65B – Big Data Analytics in Finance  
**Institution:** National Economics University  
**Student:** Vũ Khánh Hà  
**Student ID:** 11230955  
**Data Source:** FDIC BankFind Suite API  
**Repository:** https://github.com/vukhanhhha-101/bft65b-bank-failure

---

## 1. Project Overview

This project develops a reproducible big-data pipeline for analysing U.S. bank failures and building an early-warning modelling workflow using publicly available data from the Federal Deposit Insurance Corporation (FDIC).

The main objective is to design and evaluate a reliable, scalable, and reproducible data-processing pipeline rather than simply maximise predictive performance.

The project covers:

- FDIC data acquisition and validation.
- Data storage optimisation.
- Automated data-quality assessment.
- Bank-quarter panel construction and failure labelling.
- Scalable data processing using Pandas, Polars, and PySpark.
- Point-in-time feature engineering.
- Machine learning and temporal validation.
- Out-of-sample model evaluation.

The implementation is documented through a Jupyter notebook, Python scripts, and generated reports.

## 2. Repository Structure

```text
bft65b-bank-failure/
│
├── BFT65B_pipeline.ipynb
├── README.md
├── requirements.txt
├── run_all.sh
├── .gitignore
│
├── src/
│   ├── fdic_download_starter.py
│   ├── config.py
│   ├── 01b_verify_small_tables.py
│   ├── 02_storage.py
│   ├── 03_quality.py
│   ├── 04_panel.py
│   ├── 04b_label_sensitivity.py
│   ├── 04c_failure_funnel.py
│   ├── 05_benchmark.py
│   ├── 06_spark.py
│   ├── 06b_spark_scaleup.py
│   ├── 07_features.py
│   ├── 08_model.py
│   ├── 08b_lag_sensitivity.py
│   ├── 09_mllib.py
│   └── 11_test_descriptives.py
│
├── reports/
│   └── Generated pipeline outputs and evaluation results
│
└── tools/
    └── check_submission.py
```

The `src/` directory contains the Python scripts implementing the individual stages of the pipeline.

The `reports/` directory stores generated benchmark results, data-quality reports, validation outputs, and model evaluation results.

The `tools/` directory contains a submission-checking utility.

Raw and processed datasets are generated locally and excluded from GitHub version control through `.gitignore`.

## 3. Data Source and Acquisition

The project uses publicly available data from the FDIC BankFind Suite API.

Three main datasets are collected:

| Dataset | Description | Key fields |
|---|---|---|
| Financials | Quarterly bank Call Report data | CERT, REPDTE, ASSET, DEP, EQ, NETINC |
| Failures | Bank failure and assistance records | CERT, FAILDATE, COST, RESTYPE |
| Institutions | Bank reference and institutional information | CERT, STNAME, ACTIVE |

The data are collected using the course-provided `fdic_download_starter.py` script through the FDIC BankFind Suite API.

The acquisition process uses API pagination and saves downloaded records in Parquet format. It also supports resumable downloads through a manifest and performs checks on downloaded record counts and duplicate bank-quarter observations.

The financial dataset covers:

- Start date: 1984-03-31.
- End date: 2026-06-30.
- Frequency: Quarterly.

The complete Failure and Institution datasets are also collected.

**Data source:** https://api.fdic.gov/banks

**API documentation:** https://api.fdic.gov/banks/docs/

## 4. Pipeline Workflow

The project is organised into six main parts following the course requirements.

| Part | Research component | Main scripts |
|---|---|---|
| Part 1 | Acquisition and Storage | `fdic_download_starter.py`, `01b_verify_small_tables.py`, `02_storage.py` |
| Part 2 | Data Quality and Panel Construction | `03_quality.py`, `04_panel.py`, `04b_label_sensitivity.py`, `04c_failure_funnel.py` |
| Part 3 | Scalable Processing | `05_benchmark.py`, `06_spark.py`, `06b_spark_scaleup.py` |
| Part 4 | Feature Pipeline | `07_features.py` |
| Part 5 | Modelling and Validation | `08_model.py`, `08b_lag_sensitivity.py`, `09_mllib.py`, `11_test_descriptives.py` |
| Part 6 | Conclusions and Implications | Research findings, limitations, and implications for Vietnam |

### Part 1 – Acquisition and Storage

Financial, failure, and institution records are collected through the FDIC API.

Different storage layouts are evaluated to identify an efficient format for large historical banking datasets.

The project uses Parquet with Zstandard compression for processed data.

### Part 2 – Data Quality and Panel Construction

The pipeline assesses missing values, invalid observations, reporting inconsistencies, and historical changes in financial variables.

It also constructs a bank-quarter panel and defines the failure target.

Additional scripts examine alternative failure-label definitions and provide a detailed failure-record reconciliation.

### Part 3 – Scalable Processing

Pandas, Polars, and PySpark are evaluated through processing benchmarks.

The experiments examine execution time, memory consumption, query optimisation, and scalability under increasing data volumes.

PySpark is also used to reproduce selected processing operations and compare the results with the Polars implementation.

### Part 4 – Feature Pipeline

Point-in-time financial features are constructed using historical bank observations.

The feature pipeline includes publication-lag considerations and a truncation test to check for potential look-ahead bias.

### Part 5 – Modelling and Validation

The modelling workflow includes:

- Baseline comparisons.
- Logistic regression.
- Gradient boosting.
- Expanding-window temporal cross-validation.
- Expected-cost-based threshold selection.
- A separate out-of-sample test evaluation.
- Supplementary publication-lag sensitivity analysis.

The Spark MLlib script provides an additional distributed modelling demonstration.

### Part 6 – Conclusions and Implications

The final part discusses the findings from the data pipeline, processing experiments, data-quality assessment, and model evaluation.

It also considers the limitations of applying the findings to an early-warning system in Vietnam.

## 5. Methodological Design

### 5.1 Failure Definition

The primary failure definition uses FDIC records classified as:

`RESTYPE == "FAILURE"`

Assistance transactions are treated separately and examined through label-sensitivity analysis.

For bank \(i\) at prediction quarter \(q\), the target is defined as:

\[
y_{i,q} =
\mathbb{1}\{q < FAILDATE_i \leq q+4\text{ quarters}\}
\]

A positive label indicates that the bank experiences a failure within the following four quarters.

Mergers and other non-failure exits are not classified as positive failure events.

### 5.2 Point-in-Time Design

The feature pipeline considers the availability of financial reports when constructing prediction variables.

A one-quarter publication lag is applied so that financial information is not treated as available before its reporting period.

A minimum publication-lag sensitivity check of 45 days is also examined.

Year-to-date flow variables are de-cumulated before being used in the modelling dataset.

A truncation test is performed to assess whether future observations affect historical feature values.

### 5.3 Temporal Development and Test Split

The configured prediction-quarter periods are:

| Period | Quarters | Purpose |
|---|---|---|
| Development | 1984Q1–2006Q4 | Model development and validation |
| Gap | 2007Q1–2008Q4 | Eight-quarter separation |
| Test | 2009Q1–2014Q4 | Out-of-sample evaluation |

The temporal gap separates development and test observations.

Model selection and threshold selection are performed using development data. The test period is reserved for the final evaluation.

### 5.4 Threshold Selection

The classification threshold is selected using expected cost based on out-of-fold development predictions.

The false-alarm cost is treated as an explicit modelling assumption configured in `src/config.py`, rather than an observed universal cost.

## 6. Environment and Dependencies

The project environment uses the package versions pinned in `requirements.txt`.

The recorded environment includes:

- Python 3.13.
- Pandas 2.2.3.
- Polars 1.35.2.
- NumPy 2.4.6.
- PyArrow 23.0.1.
- PySpark 4.0.5.
- Scikit-learn 1.6.1.
- Matplotlib 3.10.0.
- Joblib 1.6.0.

PySpark 4.x requires a compatible Java runtime (Java 17 or later).

The project was developed and executed using Google Colab.

## 7. Installation and Execution

### 7.1 Clone the Repository

```bash
git clone https://github.com/vukhanhhha-101/bft65b-bank-failure.git

cd bft65b-bank-failure
```

### 7.2 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

### 7.3 Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 7.4 Test FDIC API Connectivity

A short download can be used to verify API connectivity before starting the complete acquisition process.

```bash
python src/fdic_download_starter.py --start 2023-03-31 --end 2023-12-31
```

### 7.5 Run the Pipeline

The main execution script is:

```bash
bash run_all.sh
```

The script executes the pipeline stages in sequence, including data acquisition, storage, data-quality assessment, panel construction, processing benchmarks, Spark processing, feature engineering, model development, and evaluation.

The complete FDIC data acquisition may take approximately one hour. The downloader supports resumable execution.

**Important – Test-stage protection**

The test stage is designed to be executed only once after development and model selection are finalised.

The pipeline uses `reports/TEST_USED.flag` to record that the test evaluation has already been performed.

The submitted repository contains this flag because the final test evaluation has already been conducted. Do not rerun the complete pipeline on the submitted version, as the test stage is protected against repeated evaluation.

### 7.6 Run the Submission Checker

From the repository root:

```bash
python tools/check_submission.py
```

This utility checks notebook references to source scripts, saved notebook execution errors, dependency pins, required README references, and data exclusions in `.gitignore`.

## 8. Generated Files and Reproducibility

The pipeline generates two main local data directories:

| Directory | Description |
|---|---|
| `data_raw/` | Raw FDIC downloads and acquisition manifest |
| `data_proc/` | Processed datasets, bank-quarter
