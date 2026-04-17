#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-${WORKSPACE_ROOT}/parameter-golf}"
EXPERIMENT_REPO_DIR="${EXPERIMENT_REPO_DIR:-${WORKSPACE_ROOT}/ml_experiments}"
RUN_ID="${RUN_ID:-blockdiff_sparse_ticket_1xh100}"

cd "${WORKSPACE_ROOT}"

if [ ! -d "${PARAMETER_GOLF_DIR}/.git" ]; then
  git clone https://github.com/openai/parameter-golf.git "${PARAMETER_GOLF_DIR}"
fi

if [ ! -d "${EXPERIMENT_REPO_DIR}/.git" ]; then
  git clone https://github.com/Rtx09x/ml_experiments.git "${EXPERIMENT_REPO_DIR}"
fi

mkdir -p "${PARAMETER_GOLF_DIR}/working/blockdiff_sparse_ticket"
cp "${EXPERIMENT_REPO_DIR}/parameter_golf/blockdiff_sparse_ticket/train_gpt.py" \
  "${PARAMETER_GOLF_DIR}/working/blockdiff_sparse_ticket/train_gpt.py"

cd "${PARAMETER_GOLF_DIR}"

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS:-80}"

export RUN_ID
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Keep training under 50 minutes so export/eval has room before the hard 1h timeout.
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

export BLOCKDIFF_ENABLED="${BLOCKDIFF_ENABLED:-1}"
export BLOCKDIFF_WEIGHT="${BLOCKDIFF_WEIGHT:-0.03}"
export BLOCKDIFF_BLOCK_SIZE="${BLOCKDIFF_BLOCK_SIZE:-16}"
export BLOCKDIFF_MASK_MIN="${BLOCKDIFF_MASK_MIN:-0.25}"
export BLOCKDIFF_MASK_MAX="${BLOCKDIFF_MASK_MAX:-0.75}"
export BLOCKDIFF_EVERY="${BLOCKDIFF_EVERY:-1}"
export BLOCKDIFF_MIN_STEP="${BLOCKDIFF_MIN_STEP:-0}"

export SPARSE_ENABLED="${SPARSE_ENABLED:-1}"
export SPARSE_TARGET="${SPARSE_TARGET:-0.30}"
export SPARSE_START_STEP="${SPARSE_START_STEP:-500}"
export SPARSE_END_STEP="${SPARSE_END_STEP:-4000}"
export SPARSE_EVERY="${SPARSE_EVERY:-100}"
export SPARSE_INCLUDE="${SPARSE_INCLUDE:-mlp,attn}"

export GPTQ_CALIB_BATCHES="${GPTQ_CALIB_BATCHES:-32}"
export GATED_ATTENTION="${GATED_ATTENTION:-0}"
export VALUE_RESIDUAL="${VALUE_RESIDUAL:-0}"
export TRIGRAM="${TRIGRAM:-0}"
export DTG_ENABLED="${DTG_ENABLED:-0}"
export LAWA_ENABLED="${LAWA_ENABLED:-0}"

timeout "${HARD_TIMEOUT_SECONDS:-3600}" \
  torchrun --standalone --nproc_per_node=1 working/blockdiff_sparse_ticket/train_gpt.py
