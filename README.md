# BFT65B – Big-Data Pipeline for Bank-Failure Early Warning (FDIC data)

Individual final report, National Economics University.
**Name:** Vũ Khánh Hà · **Student ID:** 11230955 · **Repository:** https://github.com/vukhanhhha-101/bft65b-bank-failure

## Requirements
- Python 3.11–3.13 (results in the report were produced on Google Colab, Python 3.13).
- Java 17 or 21 (only for the PySpark steps; PySpark 4.0 does not run on Java 8/11). Check with `java -version`.
- About 6 GB of free disk space for the raw data and processed files, 8 GB RAM or more.
- All package versions are pinned in `requirements.txt`; all seeds are fixed (`SEED = 42` in `src/config.py`).

## Quick start
```bash
git clone https://github.com/vukhanhhha-101/bft65b-bank-failure.git && cd bft65b-bank-failure
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/fdic_download_starter.py --start 2023-03-31 --end 2023-12-31   # smoke test (~3 min)
bash run_all.sh                                          # full run: download ~1 h, remaining steps ~30-60 min
```
Data are never committed (`data_raw/` and `data_proc/` are git-ignored).
On Google Colab, open `BFT65B_pipeline.ipynb`: it clones this repository, installs `requirements.txt` and runs the steps in order.
A step is skipped when its output already exists in `reports/` (set `FORCE = True` in the notebook to recompute).

## The test period is evaluated once
`python src/08_model.py --stage test` writes `reports/TEST_USED.flag` and refuses to run again.
The committed `reports/test_results.csv` is the result of that single run. `run_all.sh` therefore skips the test stage when the flag exists.
To reproduce the test numbers from scratch on a fresh clone, run `FORCE_TEST=1 bash run_all.sh` (this deletes the flag first and re-runs the same frozen pipeline; it does not change any development decision).

## Pipeline
| Step | Script | Main outputs (in `reports/`) |
|---|---|---|
| Download (resumable, verified against API totals) | `src/fdic_download_starter.py` | `manifest_financials.json` |
| P1 verify small tables | `src/01b_verify_small_tables.py` | `small_tables_check.json` |
| P1 storage layouts | `src/02_storage.py` | `storage_benchmark.csv` |
| P2 data-quality audit | `src/03_quality.py` | `dq_report.md`, `dq_overview.png` |
| P2 clean point-in-time panel + label | `src/04_panel.py` | `panel_summary.json` |
| P2 label sensitivity | `src/04b_label_sensitivity.py` | `label_sensitivity.csv`, `assistance_continuity.csv` |
| P3 Polars vs pandas benchmark, query plans | `src/05_benchmark.py` | `benchmark*.csv`, `polars_plan_*.txt` |
| P3 PySpark replication + match check | `src/06_spark.py` | `spark_match.txt`, `spark_plan_*.txt`, `spark_timing.csv` |
| P3 Polars vs Spark scale-up | `src/06b_spark_scaleup.py` | `spark_scaleup.csv` |
| P4 features + look-ahead test | `src/07_features.py` | `pit_test.txt` |
| P5 CV, baselines, models, threshold; **single** test | `src/08_model.py --stage dev` / `--stage test` | `cv_*.csv`, `dev_summary.json`, `test_results.csv` |
| P5 Spark MLlib demo (development data only) | `src/09_mllib.py` | `mllib_demo.csv` |
| P5 test descriptives (funnel, bank-level recall, paired bootstrap) | `src/11_test_descriptives.py` | `failure_funnel.csv`, `test_bank_level.csv`, `test_paired_bootstrap.csv` |

`tools/check_submission.py` is a self-check of the repository (scripts referenced, pinned requirements, saved errors).

## Design decisions (details in the report)
- Label `y = 1{failure in (q, q + 4 quarters]}`; mergers and other exits = 0; only `RESTYPE == "FAILURE"` counts (assistance transactions are not failures; `FAILURE_ONLY` switch in `config.py`). Modelling uses quarters up to 2025Q2.
- Development 1984Q1–2006Q4, gap 2007Q1–2008Q4 (8 quarters), test 2009Q1–2014Q4. Tuning, feature choice and the threshold use development data only.
- Year-to-date flow variables are de-cumulated; reports dated after the failure date are removed; features use only quarter q and earlier (truncation test in `pit_test.txt`).
- Validation: expanding window, 4 folds, 5-quarter purge. Threshold from expected cost on out-of-fold development predictions; the false-alarm cost is an assumption (50 thousand USD per alarm) with a sensitivity table.

## Sources and AI-tool disclosure
- Data: FDIC BankFind Suite API, https://api.fdic.gov/banks (downloaded on **20-21/11/2026**).
- `src/fdic_download_starter.py`: starter script provided by the course (used unchanged).
- Libraries: pandas, NumPy, scikit-learn, Polars, PyArrow, PySpark, matplotlib, joblib, psutil, requests (versions in `requirements.txt`).
- AI assistance: Claude (Anthropic) was used to draft and review code, to debug and to review results; I ran every step, checked the outputs and made the design decisions myself..
