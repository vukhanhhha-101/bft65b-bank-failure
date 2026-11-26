
#!/usr/bin/env bash
# BFT65B - Reproducible FDIC Bank-Failure Early-Warning Pipeline
# Python 3.11+; Java 17+ required for PySpark.
# Test evaluation is disabled by default to prevent accidental test reuse.

set -euo pipefail

export PYTHONHASHSEED=42

echo "=========================================="
echo "BFT65B - BANK FAILURE EARLY WARNING PIPELINE"
echo "=========================================="

# 1. Data acquisition and verification
echo "[1/6] Data acquisition and verification"

python src/fdic_download_starter.py --out data_raw/fdic
python src/01b_verify_small_tables.py

# 2. Storage and data quality
echo "[2/6] Storage and data quality"

python src/02_storage.py
python src/03_quality.py
python src/04_panel.py
python src/04b_label_sensitivity.py
python src/04c_failure_funnel.py

# 3. Scalable processing benchmarks
echo "[3/6] Processing benchmarks"

python src/05_benchmark.py
python src/05_benchmark.py --scales 1 4 8 --reps 1

python src/06_spark.py
python src/06b_spark_scaleup.py --scales 1 4 8 --reps 1

# 4. Feature engineering
echo "[4/6] Feature engineering"

python src/07_features.py

# 5. Model development and diagnostics
echo "[5/6] Model development and diagnostics"

python src/08_model.py --stage dev
python src/08b_lag_sensitivity.py
python src/09_mllib.py
python src/10_dev_diagnostics.py --fast

# 6. Test evaluation
echo "[6/6] Test evaluation"

if [[ "${RUN_TEST:-0}" == "1" ]]; then
    if [[ -f reports/TEST_USED.flag ]]; then
        echo "Test evaluation skipped: TEST_USED.flag already exists."
        echo "Existing test results are preserved."
    else
        echo "WARNING: Explicit test evaluation requested."
        python src/08_model.py --stage test
    fi
else
    echo "Test evaluation skipped by default."
    echo "Existing test results will be preserved."
fi

# Generate descriptive outputs only when stored test results exist.
if [[ -f reports/test_results.csv ]]; then
    python src/11_test_descriptives.py
else
    echo "No stored test_results.csv found; test descriptives skipped."
fi

echo "=========================================="
echo "PIPELINE FINISHED"
echo "Check generated outputs in reports/."
echo "=========================================="
