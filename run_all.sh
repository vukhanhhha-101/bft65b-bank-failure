#!/usr/bin/env bash
# End-to-end rerun. Tip: test first with  python src/fdic_download_starter.py --start 2023-03-31 --end 2023-12-31 --out data_raw/fdic
set -euo pipefail
python src/fdic_download_starter.py --out data_raw/fdic     # ~1 hour, resumable
python src/02_storage.py
python src/03_quality.py
python src/04_panel.py
python src/05_benchmark.py
python src/05_benchmark.py --scales 1 4 8 --reps 1
python src/06_spark.py
python src/07_features.py
python src/08_model.py --stage dev
python src/09_mllib.py

echo "Development pipeline completed."
echo "Test evaluation is intentionally excluded."
echo "Do not run the test stage without an approved evaluation plan."
