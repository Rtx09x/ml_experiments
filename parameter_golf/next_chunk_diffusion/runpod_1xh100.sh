#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-${WORKSPACE_ROOT}/parameter-golf}"
EXPERIMENT_REPO_DIR="${EXPERIMENT_REPO_DIR:-${WORKSPACE_ROOT}/ml_experiments}"
EXPERIMENT_DIR="${EXPERIMENT_REPO_DIR}/parameter_golf/next_chunk_diffusion"
RUN_ID="${RUN_ID:-next_chunk_diffusion_1xh100_90m}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/next_chunk_diffusion_runs}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_PATH="${RUN_DIR}/train.log"
ZIP_PATH="${RUN_ROOT}/${RUN_ID}.zip"
TRAIN_RC=0

cd "${WORKSPACE_ROOT}"

if [ -d "${PARAMETER_GOLF_DIR}" ] && [ -f "${PARAMETER_GOLF_DIR}/data/cached_challenge_fineweb.py" ]; then
  echo "using existing Parameter Golf workspace: ${PARAMETER_GOLF_DIR}"
elif [ ! -d "${PARAMETER_GOLF_DIR}" ]; then
  git clone https://github.com/openai/parameter-golf.git "${PARAMETER_GOLF_DIR}"
else
  BAD_DIR="${PARAMETER_GOLF_DIR}.not_parameter_golf.$(date +%s)"
  echo "moving non-Parameter-Golf path aside: ${PARAMETER_GOLF_DIR} -> ${BAD_DIR}" >&2
  mv "${PARAMETER_GOLF_DIR}" "${BAD_DIR}"
  git clone https://github.com/openai/parameter-golf.git "${PARAMETER_GOLF_DIR}"
fi

if [ ! -d "${EXPERIMENT_REPO_DIR}/.git" ]; then
  git clone https://github.com/Rtx09x/ml_experiments.git "${EXPERIMENT_REPO_DIR}"
fi

rm -rf "${RUN_DIR}"
mkdir -p "${PARAMETER_GOLF_DIR}/working/next_chunk_diffusion"
cp "${EXPERIMENT_DIR}/train_gpt.py" "${PARAMETER_GOLF_DIR}/working/next_chunk_diffusion/train_gpt.py"
cp "${EXPERIMENT_DIR}/README.md" "${PARAMETER_GOLF_DIR}/working/next_chunk_diffusion/README.md"

mkdir -p "${RUN_DIR}"
cp "${EXPERIMENT_DIR}/train_gpt.py" "${RUN_DIR}/train_gpt.py"
cp "${EXPERIMENT_DIR}/README.md" "${RUN_DIR}/README.md"
cp "${EXPERIMENT_DIR}/runpod_1xh100.sh" "${RUN_DIR}/runpod_1xh100.sh"
cp "${EXPERIMENT_DIR}/verify_packaging.py" "${RUN_DIR}/verify_packaging.py"

cd "${PARAMETER_GOLF_DIR}"

TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
DATASET_DIR="${PARAMETER_GOLF_DIR}/data/datasets/fineweb10B_sp1024"
TOKENIZER_FILE="${PARAMETER_GOLF_DIR}/data/tokenizers/fineweb_1024_bpe.model"
EXISTING_TRAIN_SHARDS=0
if [ -d "${DATASET_DIR}" ]; then
  EXISTING_TRAIN_SHARDS="$(find "${DATASET_DIR}" -maxdepth 1 -name 'fineweb_train_*.bin' -type f 2>/dev/null | wc -l | tr -d ' ')"
fi

if [ "${EXISTING_TRAIN_SHARDS}" -ge "${TRAIN_SHARDS}" ] && [ -f "${TOKENIZER_FILE}" ]; then
  echo "dataset already present: ${EXISTING_TRAIN_SHARDS} train shards, tokenizer present"
else
  if [ -d "${DATASET_DIR}" ]; then
    echo "removing partial/stale dataset dir before download: ${DATASET_DIR}"
    rm -rf "${DATASET_DIR}"
  fi
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
fi

export RUN_ID
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Aggressive probe: bigger model, 90-minute training cap, 50% sparsity, int6 export.
export MODEL_DIM="${MODEL_DIM:-576}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-4.0}"

# Cap training only. Export, final eval, artifact copy, and zipping may finish after this.
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-5400}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

export CHUNKDIFF_ENABLED="${CHUNKDIFF_ENABLED:-1}"
export CHUNKDIFF_WEIGHT="${CHUNKDIFF_WEIGHT:-0.05}"
export CHUNKDIFF_CHUNK_SIZE="${CHUNKDIFF_CHUNK_SIZE:-64}"
export CHUNKDIFF_MASK_MIN="${CHUNKDIFF_MASK_MIN:-0.50}"
export CHUNKDIFF_MASK_MAX="${CHUNKDIFF_MASK_MAX:-0.95}"
export CHUNKDIFF_EVERY="${CHUNKDIFF_EVERY:-1}"
export CHUNKDIFF_MIN_STEP="${CHUNKDIFF_MIN_STEP:-0}"

export SPARSE_ENABLED="${SPARSE_ENABLED:-1}"
export SPARSE_TARGET="${SPARSE_TARGET:-0.50}"
export SPARSE_START_STEP="${SPARSE_START_STEP:-800}"
export SPARSE_END_STEP="${SPARSE_END_STEP:-5000}"
export SPARSE_EVERY="${SPARSE_EVERY:-100}"
export SPARSE_INCLUDE="${SPARSE_INCLUDE:-mlp,attn}"

export GPTQ_EXPORT="${GPTQ_EXPORT:-1}"
export GPTQ_CALIB_BATCHES="${GPTQ_CALIB_BATCHES:-32}"
export GPTQ_BLOCK_SIZE="${GPTQ_BLOCK_SIZE:-128}"
export FAST_EXPORT="${FAST_EXPORT:-1}"
export SELECTIVE_PRUNE_MODE="${SELECTIVE_PRUNE_MODE:-full}"
export LZMA_PRESET="${LZMA_PRESET:-6}"
export QUANT_ROUNDTRIP_EVAL="${QUANT_ROUNDTRIP_EVAL:-1}"
export TARGET_MB="${TARGET_MB:-15.9}"

export GATED_ATTENTION="${GATED_ATTENTION:-0}"
export VALUE_RESIDUAL="${VALUE_RESIDUAL:-0}"
export TRIGRAM="${TRIGRAM:-0}"
export DTG_ENABLED="${DTG_ENABLED:-0}"
export LAWA_ENABLED="${LAWA_ENABLED:-0}"
export EXPORT_ONLY_CHECKPOINT="${EXPORT_ONLY_CHECKPOINT:-}"

python3 - <<'PY' > "${RUN_DIR}/env.json"
import json
import os

keys = [
    "RUN_ID", "DATA_PATH", "TOKENIZER_PATH", "VOCAB_SIZE",
    "MODEL_DIM", "NUM_LAYERS", "NUM_HEADS", "NUM_KV_HEADS", "MLP_MULT",
    "MAX_WALLCLOCK_SECONDS", "TRAIN_BATCH_TOKENS", "VAL_BATCH_SIZE",
    "VAL_LOSS_EVERY", "TRAIN_LOG_EVERY",
    "CHUNKDIFF_ENABLED", "CHUNKDIFF_WEIGHT", "CHUNKDIFF_CHUNK_SIZE",
    "CHUNKDIFF_MASK_MIN", "CHUNKDIFF_MASK_MAX", "CHUNKDIFF_EVERY",
    "CHUNKDIFF_MIN_STEP",
    "SPARSE_ENABLED", "SPARSE_TARGET", "SPARSE_START_STEP",
    "SPARSE_END_STEP", "SPARSE_EVERY", "SPARSE_INCLUDE",
    "GPTQ_EXPORT", "GPTQ_CALIB_BATCHES", "GPTQ_BLOCK_SIZE",
    "FAST_EXPORT", "SELECTIVE_PRUNE_MODE", "LZMA_PRESET",
    "QUANT_ROUNDTRIP_EVAL", "TARGET_MB",
    "GATED_ATTENTION", "VALUE_RESIDUAL", "TRIGRAM",
    "DTG_ENABLED", "LAWA_ENABLED", "EXPORT_ONLY_CHECKPOINT",
]
print(json.dumps({k: os.environ.get(k) for k in keys}, indent=2, sort_keys=True))
PY

rm -f final_model.pt final_model_pre_quant.pt final_model.int6.ptz

set +e
torchrun --standalone --nproc_per_node=1 working/next_chunk_diffusion/train_gpt.py \
  2>&1 | tee "${LOG_PATH}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

for artifact in final_model.pt final_model_pre_quant.pt final_model.int6.ptz; do
  if [ -f "${PARAMETER_GOLF_DIR}/${artifact}" ]; then
    cp "${PARAMETER_GOLF_DIR}/${artifact}" "${RUN_DIR}/${artifact}"
  fi
done

if [ -d "${PARAMETER_GOLF_DIR}/data/tokenizers" ]; then
  mkdir -p "${RUN_DIR}/tokenizers"
  cp "${PARAMETER_GOLF_DIR}/data/tokenizers/fineweb_1024_bpe.model" "${RUN_DIR}/tokenizers/" 2>/dev/null || true
fi

python3 "${EXPERIMENT_DIR}/verify_packaging.py" "${RUN_DIR}" "${ZIP_PATH}" "${TRAIN_RC}"

echo "run_dir=${RUN_DIR}"
echo "zip_path=${ZIP_PATH}"
echo "train_return_code=${TRAIN_RC}"

exit "${TRAIN_RC}"
