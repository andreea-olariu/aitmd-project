#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end smoke test on synthetic data (no download, no course data):
# split -> 1 federated round -> metrics.json exists -> explainability scripts
# -> centralised baseline, for an image dataset and a HAR-like CSV.
#
#   bash tests/smoke_test.sh        (about 2 minutes; run by the CI as well)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
python tests/synthetic_data.py "$WORK"

check() {  # check RUN_DIR
    if [ ! -f "$1/metrics.json" ] || [ ! -f "$1/final_model.pt" ]; then
        echo "SMOKE TEST FAILED: no results in $1" >&2
        exit 1
    fi
}

echo ">>> images: split, 1 round, explanations, centralised"
python data_splitting/split_data.py --input "$WORK/images.npz" --num-clients 3 \
    --strategy dirichlet --alpha 1.0 --min-samples 5 --out "$WORK/img" >/dev/null
SAVE_DIR="$WORK/run_img" bash scripts/run_simulation.sh "$WORK/img" "num-server-rounds=1" >"$WORK/img.log" 2>&1 \
    || { tail -40 "$WORK/img.log"; exit 1; }
check "$WORK/run_img"
python explainability/error_analysis.py --run "$WORK/run_img" --out-dir "$WORK/ea" >/dev/null
python explainability/gradcam.py --run "$WORK/run_img" --num-images 4 --out "$WORK/g.png" >/dev/null
python experiments/centralised.py --partitions "$WORK/img" --epochs 1 --seeds 1 --name smoke-central \
    >/dev/null
rm -rf outputs/experiments/smoke-central

echo ">>> HAR-like CSV: split, 1 round, feature importance"
python data_splitting/split_data.py --input "$WORK/skeletons.csv" --label-col Action \
    --sample-col IDSample --feature-cols '^J' --strategy group --group-col Camera \
    --num-clients 3 --out "$WORK/har" >/dev/null
SAVE_DIR="$WORK/run_har" bash scripts/run_simulation.sh "$WORK/har" "num-server-rounds=1" >"$WORK/har.log" 2>&1 \
    || { tail -40 "$WORK/har.log"; exit 1; }
check "$WORK/run_har"
python explainability/feature_importance.py --run "$WORK/run_har" --repeats 1 \
    --out "$WORK/fi.png" >/dev/null

echo ">>> SMOKE TEST PASSED"
