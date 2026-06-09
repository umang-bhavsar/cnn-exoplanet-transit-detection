#!/usr/bin/env bash
# Full end-to-end pipeline: download → preprocess → train → evaluate
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/4 Downloading KOI catalog + Kepler light curves ==="
python -m src.download

echo "=== 2/4 Preprocessing (BLS, phase-fold, resample) ==="
python -m src.preprocess

echo "=== 3/4 Training 1D CNN with focal loss ==="
python -m src.train

echo "=== 4/5 Evaluating on test set ==="
python -m src.evaluate

echo "=== 5/5 SMOTE tabular baseline ==="
python -m src.baseline

echo ""
echo "Done! Launch demo: streamlit run app/streamlit_app.py"
