#!/usr/bin/env bash
# End-to-end rerun from a fresh clone.
#   bash run_all.sh               full pipeline; the test stage is skipped if reports/TEST_USED.flag exists
#   FORCE_TEST=1 bash run_all.sh  also re-runs the frozen test stage (deletes the flag first)
# Smoke test first:  python src/fdic_download_starter.py --start 2023-03-31 --end 2023-12-31 --out data_raw/fdic
set -euo pipefail

python src/fdic_download_starter.py --out data_raw/fdic     # ~1 hour, resumable
python src/01b_verify_small_tables.py
python src/02_storage.py
python src/03_quality.py
python src/04_panel.py
python src/04b_label_sensitivity.py
python src/05_benchmark.py
python src/05_benchmark.py --scales 1 4 8 --reps 3
python src/06_spark.py                                      # needs Java 17 or 21
python src/06b_spark_scaleup.py --scales 1 4 8 --reps 3
python src/07_features.py
python src/08_model.py --stage dev
python src/09_mllib.py

# Test period: evaluated ONCE, after every development decision is final.
if [ "${FORCE_TEST:-0}" = "1" ]; then
  echo "FORCE_TEST=1: re-running the frozen test stage (no development decision may change)."
  rm -f reports/TEST_USED.flag
fi
if [ -f reports/TEST_USED.flag ]; then
  echo "reports/TEST_USED.flag exists: test stage skipped; using the committed test_results.csv."
else
  python src/08_model.py --stage test
fi
python src/11_test_descriptives.py
