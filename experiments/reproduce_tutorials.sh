#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-creates EVERY number in the tutorials: all splits, all experiments, seeds
# 1, 2, 3. About 2 hours on a laptop CPU. Finished seeds are skipped, so the
# script can be stopped and restarted at any time.
#
#   bash experiments/reproduce_tutorials.sh            # everything
#   bash experiments/reproduce_tutorials.sh t4         # only tutorial 4 (t3, t4, t5, t6)
#
# Results: outputs/experiments/<name>/summary.json, and the tables printed at
# the end (also saved to outputs/experiments/tables.md).
# Data splits always use split_data.py's default --seed 42; the seeds 1, 2, 3
# are the training seeds (run config 'seed').
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

ONLY="${1:-all}"
P=data/partitions
EXP=outputs/experiments

split() {  # split NAME ARGS... : create data/partitions/NAME once
    local name="$1"
    shift
    [ -f "$P/$name/meta.json" ] || python data_splitting/split_data.py "$@" --out "$P/$name" >/dev/null
}
fed() {  # fed NAME PARTITIONS "OVERRIDES"
    python experiments/run_seeds.py --name "$1" --partitions "$P/$2" --config "$3" | tail -5
}
central() {  # central NAME PARTITIONS ARGS...
    local name="$1" partitions="$2"
    shift 2
    python experiments/centralised.py --name "$name" --partitions "$P/$partitions" "$@" | tail -5
}
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

# --------------------------- Tutorial 3: does the bias matter? -------------
if want t3; then
    D="--dataset dermamnist --num-clients 5"
    split derma_iid5 $D --strategy iid
    split derma_dir1.0_5 $D --strategy dirichlet --alpha 1.0
    split derma_quantity0.5_5 $D --strategy quantity --alpha 0.5
    split derma_dir0.1_5 $D --strategy dirichlet --alpha 0.1
    split derma_patho2_5 $D --strategy pathological --classes-per-client 2
    R="num-server-rounds=5"
    fed t3-iid-fedavg derma_iid5 "$R"
    fed t3-dir1.0-fedavg derma_dir1.0_5 "$R"
    fed t3-quantity0.5-fedavg derma_quantity0.5_5 "$R"
    fed t3-dir0.1-fedavg derma_dir0.1_5 "$R"
    fed t3-dir0.1-fedprox derma_dir0.1_5 "$R strategy='fedprox'"
    fed t3-patho2-fedavg derma_patho2_5 "$R"
    fed t3-patho2-fedprox derma_patho2_5 "$R strategy='fedprox'"
fi

# --------------------------- Tutorial 4: robustness and privacy -------------
if want t4; then
    split derma_iid5 --dataset dermamnist --num-clients 5 --strategy iid
    R="num-server-rounds=5"
    A="malicious-clients='4'"
    fed t4-clean-fedavg derma_iid5 "$R"
    fed t4-signflip-fedavg derma_iid5 "$R $A attack='sign-flip'"
    fed t4-signflip-fedmedian derma_iid5 "$R $A attack='sign-flip' strategy='fedmedian'"
    fed t4-signflip-fedtrimmedavg derma_iid5 "$R $A attack='sign-flip' strategy='fedtrimmedavg'"
    fed t4-signflip-krum derma_iid5 "$R $A attack='sign-flip' strategy='krum'"
    fed t4-noise-fedavg derma_iid5 "$R $A attack='noise'"
    fed t4-noise-fedmedian derma_iid5 "$R $A attack='noise' strategy='fedmedian'"
    fed t4-noise-krum derma_iid5 "$R $A attack='noise' strategy='krum'"
    fed t4-clean-krum derma_iid5 "$R strategy='krum'"
    fed t4-noise-inflate100-fedavg derma_iid5 "$R $A attack='noise' attack-inflate-examples=100"
    fed t4-noise-inflate100-fedmedian derma_iid5 "$R $A attack='noise' attack-inflate-examples=100 strategy='fedmedian'"
    # Data poisoning: client 4 has only wrong labels, sends honest weights, lies about its size
    split derma_iid5_poison4 --dataset dermamnist --num-clients 5 --strategy iid --label-noise 0,0,0,0,1.0
    fed t4-poison-fedavg derma_iid5_poison4 "$R"
    fed t4-poison-inflate100-fedavg derma_iid5_poison4 "$R $A attack='none' attack-inflate-examples=100"
    fed t4-poison-inflate100-fedmedian derma_iid5_poison4 "$R $A attack='none' attack-inflate-examples=100 strategy='fedmedian'"
    for sigma in 0.1 0.5 1.0 2.0; do
        fed "t4-dp$sigma" derma_iid5 "$R dp-noise-multiplier=$sigma"
    done
fi

# --------------------------- Tutorial 5: the iterative process --------------
if want t5; then
    split derma_dir0.3_3 --dataset dermamnist --num-clients 3 --strategy dirichlet --alpha 0.3
    split derma64_dir0.3_3 --dataset dermamnist --size 64 --num-clients 3 --strategy dirichlet --alpha 0.3
    R="num-server-rounds=10"
    B="loss-weighting='balanced'"
    fed t5-it0-fedavg derma_dir0.3_3 "$R"
    fed t5-it1-balanced derma_dir0.3_3 "$R $B"
    fed t5-it2-balanced-qfedavg derma_dir0.3_3 "$R $B strategy='qfedavg' qfedavg-q=1.0"
    fed t5-alt-balanced-64px derma64_dir0.3_3 "$R $B"
    fed t5-alt-balanced-20rounds derma_dir0.3_3 "num-server-rounds=20 $B"
    fed t5-alt-balanced-3epochs derma_dir0.3_3 "$R $B local-epochs=3"
    fed t5-alt-balanced-fedprox derma_dir0.3_3 "$R $B strategy='fedprox'"
    fed t5-alt-balanced-fedadam derma_dir0.3_3 "$R $B strategy='fedadam'"
    fed t5-alt-balanced-fedavgm derma_dir0.3_3 "$R $B strategy='fedavgm'"
    central t5-central-plain derma_dir0.3_3 --epochs 10
    central t5-central-balanced derma_dir0.3_3 --epochs 10 --loss-weighting balanced
fi

# --------------------------- Tutorial 6: HAR -------------------------------
if want t6; then
    H="--input data/raw/har/train_client1.csv data/raw/har/train_client2.csv data/raw/har/train_client3.csv --label-col Action --sample-col IDSample --feature-cols ^J"
    if [ ! -f data/raw/har/train_client1.csv ]; then
        echo "Skipping tutorial 6: install the HAR data first (data_splitting/get_data.py --har <zip>)"
    else
        split har_camera $H --strategy group --group-col Camera --num-clients 3
        split har_files_skew $H --strategy group --group-col source_file --num-clients 3 --feature-skew
        split har_dirichlet $H --strategy dirichlet --alpha 0.3 --num-clients 3
        split har_camera_test3 $H --strategy group --group-col Camera --test-groups 3 --num-clients 2
        R="num-server-rounds=10 local-epochs=3"
        fed t6-camera-fedavg har_camera "$R"
        fed t6-files-skew-fedavg har_files_skew "$R"
        fed t6-dirichlet-fedavg har_dirichlet "$R"
        fed t6-camera-test3-fedavg har_camera_test3 "$R"
        fed t6-camera-signflip-fedavg har_camera "$R malicious-clients='2'"
        fed t6-camera-signflip-fedmedian har_camera "$R malicious-clients='2' strategy='fedmedian'"
        fed t6-camera-signflip-krum har_camera "$R malicious-clients='2' strategy='krum'"
        fed t6-camera-fedprox har_camera "$R strategy='fedprox'"
        fed t6-camera-dp0.5 har_camera "$R dp-noise-multiplier=0.5"
        central t6-central-camera har_camera --epochs 30
        central t6-central-camera-test3 har_camera_test3 --epochs 30
    fi
fi

echo
echo "=========================== tables (mean ± std over seeds) ==========================="
{
    for t in t3 t4 t5 t6; do
        dirs=$(ls -d "$EXP/$t"-* 2>/dev/null || true)
        [ -n "$dirs" ] || continue
        echo
        echo "### $t"
        # shellcheck disable=SC2086
        python experiments/summarize.py $dirs --with-test
    done
} | tee "$EXP/tables.md"
