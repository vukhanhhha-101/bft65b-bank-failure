# BFT65B – Big-Data Pipeline for Bank-Failure Early Warning (FDIC data)

Individual final report, National Economics University. Name / Student ID: **<fill in>**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # Python 3.11 recommended (PySpark 3.5 + Java 11/17/21)
python src/fdic_download_starter.py --start 2023-03-31 --end 2023-12-31   # smoke test (~3 min)
bash run_all.sh                                          # full run: download ~1 h, rest ~30-60 min on a laptop
```
Data are never committed (`data_raw/`, `data_proc/` are git-ignored). All seeds fixed (`SEED = 42` in `src/config.py`).

## Pipeline
| Step | Script | Output (in `reports/`) |
|---|---|---|
| Download (resumable, verified vs API totals) | `src/fdic_download_starter.py` | `manifest_financials.json` |
| P1 storage layouts + optimised panel | `02_storage.py` | `storage_benchmark.*` |
| P2 data-quality audit | `03_quality.py` | `dq_report.md`, `dq_overview.png` |
| P2 clean point-in-time panel + label | `04_panel.py` | `panel_summary.json` |
| P3 Polars vs pandas benchmark, query plans | `05_benchmark.py` | `benchmark*.csv`, `polars_plan_*.txt` |
| P3 PySpark replication + match check | `06_spark.py` | `spark_match.txt`, `spark_plan_*.txt`, `spark_timing.csv` |
| P4 features + look-ahead test | `07_features.py` | `pit_test.txt` |
| P5 CV, baseline, models, threshold, **single** test | `08_model.py --stage dev/test` | `cv_*.csv`, `test_results.csv` |
| P5 Spark MLlib demo | `09_mllib.py` | `mllib_demo.csv` |

## Design decisions (see report)
Label `y = 1{FAILDATE in (q, q+1y]}`, mergers/other exits = 0, only `RESTYPE == "FAILURE"` (switch `FAILURE_ONLY`).
Development 1984Q1–2006Q4, gap 8 quarters, test 2009Q1–2014Q4. YTD flow variables are de-cumulated.
Threshold from expected cost on out-of-fold development predictions. `TEST_USED.flag` blocks a second test run.

## Sources & AI-tool disclosure
- Data: FDIC BankFind Suite API, https://api.fdic.gov/banks (accessed <date>).
- `src/fdic_download_starter.py`: course-provided starter script (unchanged).
- Remaining code drafted with the assistance of Claude (Anthropic AI assistant), then reviewed, run and adapted by the author. <describe what you changed/verified>
