# BFT65B -- A Big-Data Pipeline for Bank-Failure Early Warning (FDIC)

  --------------------------- -------------------------------------------------------
  **Student**                 Vũ Khánh Hà

  **Student ID**              11230955

  **Course**                  BFT65B -- Big Data Analytics in Finance, National
                              Economics University

  **Repository**              https://github.com/vukhanhhha-101/bft65b-bank-failure

  **Data**                    FDIC BankFind public API (https://api.fdic.gov/banks),
                              downloaded using the course data-acquisition workflow

  **Environment**             Google Colab, Python 3.13, Java 21 (PySpark); package
                              versions specified in `requirements.txt`; random seed
                              42
  --------------------------- -------------------------------------------------------

------------------------------------------------------------------------

## 1. Project Overview

This project develops a reproducible big-data pipeline using publicly
available US Federal Deposit Insurance Corporation (FDIC) data to
support a bank-failure early-warning system.

The main focus is the end-to-end data engineering process rather than
maximising predictive accuracy. The project covers data acquisition,
storage optimisation, data-quality assessment, scalable processing,
feature engineering, temporal validation and model evaluation.

It addresses three main research questions:

-   **Q1 -- Storage and processing:** How can several decades of bank
    reports be stored and processed efficiently on a standard machine,
    and when would distributed processing become necessary?
-   **Q2 -- Data quality and temporal integrity:** What quality problems
    exist in the raw data, and how can the pipeline address them while
    reducing look-ahead risk?
-   **Q3 -- Model validation:** How effectively can the resulting
    pipeline support an early-warning model, and what limitations affect
    the reliability of its evaluation?

## 2. Data Sources and Acquisition

The project uses three main FDIC datasets.

  ---------------------------------------------------------------------
  Dataset              Description                            Size used
  -------------------- -------------------- ---------------------------
  Financials           Quarterly Call           170 quarters, 1,682,738
                       Reports,                                    rows
                       1984Q1--2026Q2, 33   
                       fields (3 keys and   
                       30 financial fields) 

  Failures             FDIC failure and                   4,117 records
                       assistance records   

  Institutions         Bank reference and                27,838 records
                       institutional        
                       information          
  ---------------------------------------------------------------------

Financial amounts are expressed in thousands of USD and financial ratios
in percentage terms.

Data acquisition uses the course-provided starter script
`src/fdic_download_starter.py`. The download process supports
resumption, stores quarterly data separately and maintains a manifest
containing API totals.

The script `src/01b_verify_small_tables.py` extends completeness
verification to the Failures and Institutions datasets.

The downloaded raw data are excluded from GitHub through `.gitignore` to
avoid committing large data files. They can be regenerated using the
documented acquisition process.

## 3. Repository Structure

``` text
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
│   └── Experimental outputs, tables and diagnostics
│
└── tools/
    └── check_submission.py
```

The main notebook provides an integrated presentation of the project,
while the Python scripts implement the individual pipeline stages.

The `reports/` directory stores generated experimental results, and
`tools/check_submission.py` provides a pre-submission audit.

## 4. Pipeline Design and Methodology

### 4.1. Data Storage and Optimisation (Part 1)

The project compares five storage configurations using the full
bank-quarter panel. The comparison considers storage size, write
performance and representative analytical queries.

  ----------------------------------------------------------------------
  Storage format     Size (MB) Write time     Query A: Two  Query B: One
                               (s)            columns, all     year, all
                                                 years (s)   columns (s)
  -------------- ------------- ------------- ------------- -------------
  CSV                    446.4 9.09                  1.485         3.475

  Parquet,               395.4 2.29                  0.105         0.407
  uncompressed                                             

  Parquet,               209.6 2.77                  0.097         0.622
  Snappy                                                   

  Parquet,               149.2 5.30                  0.195         1.036
  Zstandard                                                
  level 3                                                  

  **Parquet,         **156.3** **7.71**          **0.230**     **0.098**
  Zstandard                                                
  level 3,                                                 
  partitioned by                                           
  year**                                                   
  ----------------------------------------------------------------------

Write times are based on three runs; query times are medians of five
runs, as recorded in `reports/storage_benchmark.csv`.

Year-partitioned Parquet with Zstandard compression is selected as the
primary storage layout.

It reduces storage requirements by approximately 65% compared with CSV.
In the recorded benchmark, the one-year query takes 0.098 seconds for
the partitioned layout versus 3.475 seconds for CSV (approximately 35
times faster).

Although a partitioned layout introduces some overhead for full-dataset
scans, it improves selective access to individual reporting years
through partition pruning.

Results are available in `reports/storage_benchmark.csv`.

### 4.2. Data Quality and Panel Construction (Part 2)

The raw FDIC datasets contain several challenges:

-   Missing observations and incomplete historical coverage.
-   Cumulative year-to-date financial flow variables.
-   Extreme financial values and potential data inconsistencies.
-   Gaps in bank reporting histories.
-   Multiple institutional exit types.
-   Differences between reporting dates and actual information
    availability.

The pipeline addresses these issues through automated data-quality
checks, schema validation, duplicate handling, bank-level reconciliation
and quarterly de-cumulation of year-to-date flow variables.

The main financial flows processed include NETINC, INTINC, EINTEXP,
NONII, NONIX and ELNATR.

The panel distinguishes failures, other institutional exits and active
institutions. Exit classifications are used for label construction and
are not included as predictive features.

The raw Failures table contains 4,117 records: 3,524 are classified as
`FAILURE` and 593 as `ASSISTANCE`. Of the 3,524 raw `FAILURE` records,
485 have no `CERT`; 3,039 records have a valid `CERT` and `FAILDATE`,
representing 3,036 distinct failed institutions. The 3,039 eligible
records are used for bank-level failure linkage. These counts refer to
different filtering and deduplication stages and are not
interchangeable.

The failure funnel and label sensitivity analyses are provided in:

-   `reports/dq_report.md`
-   `reports/failure_funnel.csv`
-   `reports/label_sensitivity.csv`

### 4.3. Failure Label Definition

The target variable is defined as:

`y(i,q) = 1`

if the earliest FDIC FAILURE date of bank i occurs within the following
one-year interval:

`(REPDTE, REPDTE + 1 year]`

Only records classified as `RESTYPE = "FAILURE"` are treated as
failures.

Assistance transactions, mergers and other institutional exits are not
classified as bank failures.

Reports dated on or after the failure date are removed to prevent the
use of financial information recorded after institutional failure.

Rows with reporting dates up to 2025-06-30 are retained because later
outcome labels are incomplete.

### 4.4. Development and Test Split

The modelling dataset uses a chronological split:

  -----------------------------------------------------------------------
  Period                  Reporting quarters      Purpose
  ----------------------- ----------------------- -----------------------
  Development             1984Q1--2006Q4          Feature development,
                                                  model selection and
                                                  threshold optimisation

  Gap                     2007Q1--2008Q4          Eight-quarter temporal
                                                  separation

  Test                    2009Q1--2014Q4          Historical
                                                  out-of-sample
                                                  evaluation
  -----------------------------------------------------------------------

The eight-quarter gap exceeds the required four-quarter separation.

The stored development data contain 2,290 failed banks, while the test
period contains 469 failed banks.

Model selection, feature choices and threshold optimisation are
performed using development data rather than the test results.

### 4.5. Feature Engineering (Part 4)

The feature pipeline constructs 31 point-in-time financial indicators
organised around CAMELS dimensions:

-   **Capital:** Equity and regulatory capital ratios.
-   **Asset quality:** Non-performing loans, past-due loans, non-accrual
    assets, OREO, reserves and the Texas ratio.
-   **Earnings:** ROA, ROE, net interest margin, quarterly
    profitability, provisions and cost-to-income.
-   **Liquidity and funding:** Liquidity and funding-related indicators.
-   **Sensitivity and business structure:** Loan composition, growth,
    size and four-quarter changes.

Financial flow variables are de-cumulated before feature calculation.

Quarter-aware lags are used, and a lag is set to null when the
corresponding bank has a reporting gap.

A truncation test checks whether feature values remain identical when
the underlying dataset is restricted to selected historical cut-off
dates. The results are recorded in `reports/pit_test.txt`.

**Important publication-lag limitation**

The primary modelling experiment uses quarter q financial reports to
predict failures after the end of quarter q. However, FDIC Call Reports
are generally published approximately 30--45 days after the reporting
date.

Therefore, the primary pipeline does not fully enforce the actual
publication delay. The truncation test checks historical feature
construction but does not independently establish when each report
became publicly available.

A supplementary development-only sensitivity analysis in
`src/08b_lag_sensitivity.py` examines the potential impact of a 45-day
delay.

It identifies 494 of 4,214 positive development out-of-fold observations
as failures occurring within 45 days after the reporting date. Excluding
these observations reduces HistGradientBoosting PR-AUC from 0.3419 to
0.3087 and all-feature Logistic Regression PR-AUC from 0.2869 to 0.2538,
as recorded in `reports/dev_lag_sensitivity.csv`.

This sensitivity analysis demonstrates that publication timing is a
material methodological issue. It does not establish that the primary
pipeline has been fully corrected.

Consequently, the primary performance figures should be interpreted as
potentially optimistic.

### 4.6. Modelling and Validation (Part 5)

The project evaluates four modelling approaches:

1.  Texas-ratio benchmark.
2.  Logistic Regression using four financial ratios.
3.  Logistic Regression using all engineered features.
4.  HistGradientBoosting with maximum depth 4 and 200 iterations.

All preprocessing operations, including winsorisation, imputation and
scaling, are fitted within the training pipeline to avoid information
leakage between training and validation observations.

**Development validation**

An expanding-window cross-validation strategy is applied using four
chronological folds:

-   1989Q1--1991Q4
-   1992Q1--1994Q4
-   1995Q1--2000Q4
-   2001Q1--2006Q4

A five-quarter purge is used to reduce overlap between training label
windows and validation periods.

**Evaluation metrics**

PR-AUC is the primary evaluation metric because bank failures represent
a relatively rare outcome.

Additional metrics include ROC-AUC, recall, precision, alarm rate and
expected cost.

Accuracy is not used as the principal performance measure.

**Threshold selection**

The operating threshold is selected by minimising expected cost on
development out-of-fold predictions.

The missed-failure cost is estimated using observed FDIC failure COST
data. The false-alarm review cost is an explicit modelling assumption of
USD 50,000 per review, controlled through `C_FALSE_ALARM_K` in
`src/config.py`.

Threshold sensitivity results are available in
`reports/threshold_sensitivity.csv`.

**Spark MLlib**

The script `src/09_mllib.py` demonstrates Logistic Regression and
Gradient-Boosted Trees using Spark MLlib on development data only.

**Test evaluation disclosure**

The repository's `reports/TEST_USED.flag` records `test evaluated once`.
The stored test artifacts are therefore documented as the single
recorded test evaluation in the submitted pipeline state.

The test evaluation does not fully adjust prediction dates for the
30--45-day publication delay. Accordingly, the stored test metrics are
historical out-of-sample results under the report-date design, not
verified performance from a fully publication-lag-adjusted pipeline. The
publication-lag sensitivity analysis is performed on development
out-of-fold predictions and does not constitute a second test
evaluation.

The saved results are available in:

-   `reports/test_results.csv`
-   `reports/test_bank_level.csv`
-   `reports/test_paired_bootstrap.csv`

## 5. Main Experimental Results

The following table summarises the principal recorded findings.
Benchmark values are taken from the saved project outputs.

  ---------------------------------------------------------------------
  Topic                              Main result
  ---------------------------------- ----------------------------------
  Storage optimisation               Year-partitioned Zstandard
                                     Parquet: 156.3 MB versus 446.4 MB
                                     for CSV

  Selective extraction               One-year query: 0.098 seconds
                                     versus 3.475 seconds for CSV

  Polars aggregation (1×)            Lazy Polars: 0.12 seconds; pandas:
                                     0.51 seconds; streaming Polars:
                                     0.08 seconds

  Polars window operation (1×)       Lazy Polars: 0.70 seconds; pandas:
                                     2.47 seconds; streaming Polars:
                                     0.76 seconds

  Polars label join (1×)             Lazy Polars: 0.35 seconds; pandas:
                                     0.86 seconds; streaming Polars:
                                     0.18 seconds

  Scale-up benchmark                 Polars lazy workloads scale from
                                     1.68 million to 13.45 million rows
                                     at 1×, 4× and 8×

  Spark local-mode scale-up          2.68 s at 1×, 8.87 s at 4× and
                                     15.22 s at 8×; corresponding
                                     Polars lazy times are 0.32 s, 0.66
                                     s and 1.36 s

  Spark consistency                  Aggregation, window and label-join
                                     checks match Polars; both engines
                                     return 10,904 positive labels in
                                     the recorded check

  Development PR-AUC                 Texas rule: 0.297; baseline
                                     Logistic Regression: 0.202;
                                     all-feature Logistic Regression:
                                     0.287; HistGradientBoosting: 0.342

  Publication-lag sensitivity        HGB PR-AUC decreases from 0.3419
                                     to 0.3087; all-feature Logistic
                                     Regression decreases from 0.2869
                                     to 0.2538

  Historical test results            Stored in `test_results.csv`,
                                     `test_bank_level.csv` and
                                     `test_paired_bootstrap.csv`; see
                                     the test-use disclosure in Section
                                     4.6
  ---------------------------------------------------------------------

The benchmark outputs show that Polars is efficient for the current
dataset scale. At 8× scale, the lazy Polars workload processes
approximately 13.45 million rows in 1.36 seconds, while the
corresponding Spark local-mode workload takes 15.22 seconds. This
comparison demonstrates the overhead of local Spark execution; it does
not imply that a properly provisioned multi-node Spark cluster would be
slower.

The development results favour HistGradientBoosting on pooled
out-of-fold PR-AUC. These results remain subject to the
publication-timing limitation described above. The saved test results
are reported separately and are not used to select models or thresholds.

## 6. Reproducibility and How to Run

### 6.1. Environment

The project was developed in Google Colab using Python 3.13 and Java 21
for PySpark.

Required Python package versions are specified in `requirements.txt`.

The random seed is set to 42 for reproducibility where supported by the
relevant libraries.

### 6.2. Google Colab

Clone the repository:

``` python
!git clone https://github.com/vukhanhhha-101/bft65b-bank-failure.git
%cd bft65b-bank-failure
```

Install the required packages:

``` python
!pip install -r requirements.txt
```

Run the pipeline:

``` python
!bash run_all.sh
```

The pipeline downloads the FDIC data, executes data verification,
storage and quality processing, benchmarks, feature engineering and
development-stage modelling.

**Test evaluation is disabled by default in the submitted
`run_all.sh`.** This is intentional to prevent accidental reuse of the
previously accessed test period.

The script preserves existing test artifacts and does not automatically
perform another test evaluation.

### 6.3. Local Execution

The repository can also be cloned and executed locally using Git Bash,
WSL, Linux or macOS.

``` bash
git clone https://github.com/vukhanhhha-101/bft65b-bank-failure.git

cd bft65b-bank-failure

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

bash run_all.sh
```

Java is required for PySpark.

### 6.4. Data Download

A short connectivity test can be performed using:

``` bash
python src/fdic_download_starter.py \
    --start 2023-03-31 \
    --end 2023-12-31 \
    --out data_raw/smoke
```

The full historical download may take approximately one hour, depending
on network and execution conditions. The download process supports
resumption.

### 6.5. Generated Outputs

The pipeline creates the following local directories:

-   `data_raw/`: Downloaded FDIC data and acquisition manifests.
-   `data_proc/`: Processed Parquet datasets and modelling panels.
-   `reports/`: Data-quality reports, benchmarks, model results and
    diagnostics.

Large raw and processed datasets are excluded from the repository
through `.gitignore`.

The notebook `BFT65B_pipeline.ipynb` provides a step-by-step
presentation of the project and its saved outputs.

The pre-submission audit can be run using:

``` bash
python tools/check_submission.py BFT65B_pipeline.ipynb .
```

This checks notebook references, saved execution errors, dependency pins
and `.gitignore` configuration.

## 7. Limitations and Future Improvements

Several limitations should be considered when interpreting the results.

1.  **Publication timing:** The primary experiment does not fully
    enforce the 30--45-day reporting publication delay. The development
    sensitivity analysis demonstrates a material reduction in predictive
    performance when observations affected by the assumed delay are
    excluded.

2.  **Historical financial data:** The FDIC dataset contains the latest
    available, potentially restated Call Report values rather than
    exclusively first-filed historical values.

3.  **Institutional exits:** Assistance transactions are not classified
    as failures. A total of 485 raw FAILURE records have no CERT
    identifier and therefore cannot be linked to a bank-level financial
    panel. This is a linkage limitation; the available audit does not
    establish that all such records predate 1984.

4.  **False-alarm costs:** The assumed review cost is not directly
    observed supervisory expenditure. Cost-based thresholds should
    therefore be interpreted as scenario-dependent.

5.  **Temporal and institutional dependence:** Observations from the
    same bank are correlated. Bank-clustered bootstrap analysis is used
    for test confidence intervals, but additional institutional and
    crisis-regime validation would strengthen the findings.

6.  **Historical generalisation:** The test period represents one
    historical crisis regime. Model performance may differ under other
    economic and regulatory conditions.

7.  **Computational scope:** Benchmarks were conducted in one Google
    Colab environment. The PySpark local-mode implementation
    demonstrates distributed-processing concepts but does not represent
    a production cluster. The MLlib demonstration is also not fully
    tuned.

8.  **Evaluation independence:** The repository marker records one test
    evaluation. The test period is nevertheless not a fully
    publication-lag-adjusted evaluation, so the saved metrics should be
    interpreted within that methodological limitation.

### Implications for Vietnam

The pipeline provides a methodological foundation for developing a
bank-failure early-warning system in Vietnam, but US FDIC data, failure
definitions and model thresholds cannot be transferred directly.

A Vietnamese implementation would require authorised supervisory data
from the State Bank of Vietnam (SBV), potentially including quarterly
financial statements, capital adequacy, liquidity, asset quality,
profitability and credit exposure.

Data processing and feature engineering would need to reflect Vietnamese
accounting standards, reporting formats, bank identifiers and regulatory
classifications.

Actual report publication timestamps should replace the assumed US
publication delay. Failure definitions, prediction horizons, cost
assumptions and decision thresholds would need to be recalibrated using
Vietnamese historical banking data.

The system should initially support supervisory decision-making through
risk ranking, early-warning indicators and explainable signals rather
than automatically determining whether an institution will fail.

The main contribution to Vietnam is therefore a reproducible and
scalable data-engineering architecture that can be adapted to local
supervisory data and regulatory requirements, rather than a directly
transferable US-trained prediction model.

## 8. AI Assistance, References and Academic Integrity

ChatGPT (OpenAI) and Claude (Anthropic) were used as AI-assisted tools
during the project for coding assistance, debugging, methodological
discussion, documentation drafting and language editing.

The student reviewed and modified the project materials, executed
project workflows, inspected available outputs and is responsible for
the submitted code, analysis, interpretations and methodological
disclosures.

AI-generated suggestions were not treated as independent experimental
verification. All reported results are based on the project's saved
experimental artifacts and are subject to the limitations explicitly
documented in this repository.

**References and technical documentation**

1.  Federal Deposit Insurance Corporation. (n.d.). *BankFind Suite and
    public API documentation*. https://api.fdic.gov/banks/docs/
2.  National Economics University. (2026). *BFT65B course materials and
    starter script* \[Course materials\].
3.  Polars. (n.d.). *Polars user guide*. https://docs.pola.rs/
4.  pandas development team. (n.d.). *pandas documentation*.
    https://pandas.pydata.org/docs/
5.  scikit-learn developers. (n.d.). *scikit-learn documentation*.
    https://scikit-learn.org/stable/
6.  Apache Software Foundation. (n.d.). *Apache Spark documentation*.
    https://spark.apache.org/docs/latest/
7.  OpenAI. (2026). *ChatGPT* \[Large language model\].
    https://chatgpt.com/
8.  Anthropic. (2026). *Claude* \[Large language model\].
    https://claude.ai/

------------------------------------------------------------------------

**Final note:** This repository documents the implemented pipeline,
recorded experimental results, methodological limitations and proposed
improvements. Predictive results should be interpreted cautiously,
particularly with respect to publication timing and repeated access to
the test period.
