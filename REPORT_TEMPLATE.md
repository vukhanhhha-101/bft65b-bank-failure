# A Big-Data Pipeline for Bank Failure Early Warning
**Name:** … **Student ID:** … **GitHub:** https://github.com/…/… (public)
**Download method & date:** FDIC API via `fdic_download_starter.py`, downloaded on …
**Row counts:** Financials … rows (… quarters, … banks) · Failures … · Institutions …
**Development period:** 1984Q1–2006Q4 (… failed banks) · **Gap:** 2007Q1–2008Q4 · **Test period:** 2009Q1–2014Q4 (… failed banks)
*(numbers → `reports/panel_summary.json`; test ≥ 50 failures is asserted in `04_panel.py`)*

## 1. Acquisition & Storage (15)
- Method: API, 1 request series per quarter, retry + back-off, manifest → resumable. Completeness: rows per quarter = API total, 0 duplicates (`dq_report.md` §1).
- Field choice (34 columns, table with code | meaning | why): capital (EQ, RBC1AAJ, RBCRWAJ), asset quality (NCLNLS, P3/P9ASSET, NAASSET, ORE, LNATRES), earnings (NETINC, ROA, NIMY…), liquidity (CHBAL, SC, BRO, DEP). **Check each against the field dictionary.**
- Layout: explicit schema (Int32 CERT, Date REPDTE, Float64 amounts), Parquet + zstd, partitioned by year, sorted by (CERT, REPDTE).
- **Table: CSV vs Parquet variants** ← paste `storage_benchmark.md` (size, write time, 2-column scan, one-year query). Explain: columnar → column pruning; row-group statistics + partitions → predicate pruning; zstd vs snappy trade-off.

## 2. Data Quality & Panel (20)
- Table of problems ← `dq_report.md`: missing fields by decade (older fields absent), rule violations, duplicates, gaps, tails.
- Key finding: flow variables are **year-to-date** (§4 of DQ report) → de-cumulated (`*_Q`).
- Handling table: problem | evidence | treatment | why (drop only ASSET ≤ 0; winsorise inside the pipeline, fitted on train; no imputation before the split).
- Exits: failure vs merger/other (`CHANGEC1`), `EXIT_TYPE` is diagnostic only.
- Point-in-time: features use quarter q and earlier; reports dated ≥ FAILDATE dropped; label window (q, q+4Q]; reporting lag (~45 days) and FDIC restatements → limitation (`fail_before_avail` count).

## 3. Scalable Processing (25)
- Polars: lazy scan, projection/predicate pushdown, window functions `over("CERT")` — paste `polars_plan_naive.txt` vs `polars_plan_optimized.txt` and explain.
- **Benchmark table** ← `benchmark.csv` (3 tasks × pandas / Polars eager / lazy / streaming; median of 3, fresh process, peak memory) + `benchmark_scaleup.csv` (×1, ×4, ×8).
- Spark: `spark_match.txt` (agg / window / label identical), `spark_plan_*.txt` (Exchange = shuffle, WindowExec, FileScan pushed filters), `spark_timing.csv` (200 vs 16 shuffle partitions).
- When is a cluster justified? Argue **from your numbers**: the panel is ~… MB in Parquet and fits in RAM; Spark is slower at this size because of JVM start-up and shuffle overhead; the cluster pays off when data × k exceeds RAM (memory wall at scale ×… in the scale-up test), when many users/jobs share data, or for daily micro-batch updates.

## 4. Feature Pipeline (15)
- Feature table (name | formula | CAMELS dimension | rationale | expected sign). 
- Point-in-time test (`pit_test.txt`): features recomputed with data truncated at 3 cut-offs are identical.
- sklearn `Pipeline`: Winsor(0.5%) → median imputer → scaler → model, fitted on training folds only.

## 5. Modelling & Validation (15)
- Split justification: test = 2008 crisis aftermath, ≥ 50 failures, 8-quarter gap > label horizon; regime shift = honest out-of-time test.
- CV: expanding window, 4 folds, purge 5 quarters. No random CV.
- Metrics: PR-AUC (primary, prevalence ≈ …%), ROC-AUC, recall/precision at threshold, expected cost; bank-clustered bootstrap CI. Accuracy is not reported (trivial 99%+).
- Models: rule (Texas ratio), logistic on 4 ratios, logistic all features, gradient boosting; table ← `cv_scores.csv`, `cv_pooled.csv`; MLlib demo ← `mllib_demo.csv`.
- Threshold: C_miss = mean COST of failures in dev ÷ positive rows per failure (data); C_FA = **assumption** (50k USD per review) + sensitivity ← `threshold_sensitivity.csv`.
- Single test evaluation ← `test_results.csv`, flag file.

## 6. Conclusions (5)
**Q1** Laptop + Parquet + Polars suffices for 1.7M rows; cluster when … (numbers).  
**Q2** Problems: YTD flows, missing fields in early decades, gaps, exits, outliers; point-in-time via lags & truncation test.  
**Q3** Yes/partly: test PR-AUC … vs baseline …, cost saving …%; honest caveats.  
**Limitations:** restated data, assistance transactions, crisis-regime shift, C_FA assumption, repeated rows per failing bank.  
**Vietnam:** supervisory data not public/short history; very few failures (rescue/transfer by SBV rather than closure) → low base rate, rely on distress events/rating migration/transfer learning; accounting (VAS vs US GAAP), different reporting lags, data-governance and access (on-prem Spark/Parquet lake) — verify facts with official sources before writing.

## References & AI disclosure
FDIC BankFind API; course starter script; polars/pandas/Spark docs; Claude (Anthropic) used for drafting code — state what you verified.
