#!/bin/bash
# Submit the independent Musci-ASR-2B evaluator to HF Jobs.
# Usage: HF_TOKEN=hf_... bash musci-asr-2b/submit_jobs.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/submit_utils.sh"

SPACE="${SPACE:-Musci-research/open-asr-leaderboard-musci-asr-2b}"
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/asr_leaderboard_h200}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}"
FLAVOR="${FLAVOR:-h200}"
ORG_NAME="${ORG_NAME:-}"
MODEL_ID="${MODEL_ID:-Musci-research/Musci-ASR-2B}"
# Set this to a Hub commit for reproducible runs. Empty means the model's main revision.
MODEL_REVISION="${MODEL_REVISION:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Set USE_LOCAL_SCRIPT=1 or USE_LOCAL_NORMALIZER=1 while iterating locally.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SCRIPT_INJECT=""
[[ "${USE_LOCAL_SCRIPT:-0}" == "1" ]] && LOCAL_SCRIPT_INJECT="$(local_script_inject "${SCRIPT_DIR}" run_eval.py)"
LOCAL_NORMALIZER_INJECT=""
[[ "${USE_LOCAL_NORMALIZER:-0}" == "1" ]] && LOCAL_NORMALIZER_INJECT="$(local_normalizer_inject)"

# name split batch_size [dataset repo]. The ten entries are the complete current
# English leaderboard set plus Monsoon en-IN and both URGENT2024 variants.
DATASET_CONFIGS=(
    "ami_cleaned test ${BATCH_SIZE}"
    "earnings22_cleaned_aa_chunked test ${BATCH_SIZE} ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "gigaspeech_cleaned test ${BATCH_SIZE}"
    "librispeech test.clean ${BATCH_SIZE}"
    "librispeech test.other ${BATCH_SIZE}"
    "spgispeech test ${BATCH_SIZE}"
    "voxpopuli_cleaned_aa test ${BATCH_SIZE}"
    "monsoon_en_in test ${BATCH_SIZE} VoiceArena/Monsoon_en_IN_test"
    "urgent2024 test ${BATCH_SIZE}"
    "urgent2024_clean test ${BATCH_SIZE}"
)
filter_only_datasets || exit 1

MODEL_FOLDER="${MODEL_ID//\//-}"
MODEL_REVISION_ARG=""
[[ -n "${MODEL_REVISION}" ]] && MODEL_REVISION_ARG="--model_revision=${MODEL_REVISION}"
NAMESPACE_ARG=""
[[ -n "${ORG_NAME}" ]] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

for cfg in "${DATASET_CONFIGS[@]}"; do
    read -r DATASET SPLIT EFFECTIVE_BATCH_SIZE DATASET_PATH <<< "${cfg}"
    if [[ -n "${DATASET_PATH:-}" ]]; then
        DATASET_CONFIG=""
    else
        DATASET_PATH="${DEFAULT_DATASET_PATH}"
        DATASET_CONFIG="${DATASET}"
    fi
    echo "Submitting ${MODEL_ID} on ${DATASET} (${DATASET_PATH}, ${SPLIT})"
    hf jobs run \
        --flavor "${FLAVOR}" \
        --timeout 12h \
        --secrets HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting jobs}" \
        --env HF_AUDIO_DECODER_BACKEND=soundfile \
        ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" \
        "hf.co/spaces/${SPACE}" \
        bash -c "
            ${LOCAL_NORMALIZER_INJECT}
            ${LOCAL_SCRIPT_INJECT}
            PYTHONPATH=/app python run_eval.py \\
                --model_id=${MODEL_ID} \\
                ${MODEL_REVISION_ARG} \\
                --dataset_path=${DATASET_PATH} \\
                --dataset=${DATASET_CONFIG} \\
                --split=${SPLIT} \\
                --device=0 \\
                --batch_size=${EFFECTIVE_BATCH_SIZE} \\
                --max_new_tokens=${MAX_NEW_TOKENS} \\
                --max_eval_samples=-1 &&
            mkdir -p /results/${MODEL_FOLDER} &&
            cp results/*.jsonl /results/${MODEL_FOLDER}/
        " &
done

wait
sleep 10
mkdir -p "./results/${MODEL_FOLDER}"
RUN_RESULTS="$(python "${FETCH_RUN_RESULTS}" \
    --bucket "${RESULTS_BUCKET}" \
    --model-folder "${MODEL_FOLDER}" \
    --local-dir "./results/${MODEL_FOLDER}" \
    --since "${RUN_START}" \
    --expected "${#DATASET_CONFIGS[@]}")"
PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('${RUN_RESULTS}', '${MODEL_ID}')
"
