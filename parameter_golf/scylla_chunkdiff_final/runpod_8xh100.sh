#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${RUN_MODE:-full}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${PGOLF_WORKSPACE:-/workspace/parameter-golf}"
DATA_ROOT="${PGOLF_DATA_ROOT:-${WORKSPACE}/data/datasets}"
DATA_PATH="${PGOLF_SCYLLA_DATA_PATH:-${DATA_ROOT}/fineweb_scylla}"
HF_DATASET="${PGOLF_HF_DATASET:-LightSpeedUp/parameter-golf-data}"
RUN_ID="${RUN_ID:-scylla_chunkdiff_final_8xh100}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/scylla_chunkdiff_final_runs}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

detect_gpus() {
  python - <<'PY'
import torch
print(max(1, torch.cuda.device_count() if torch.cuda.is_available() else 0))
PY
}

install_deps() {
  python -m pip install -q --upgrade pip
  python -m pip install -q packaging ninja wheel setuptools einops huggingface-hub hf_transfer sentencepiece tokenmonster datasets tqdm brotli numpy
  python - <<'PY'
import subprocess, sys
try:
    import torch
    ok = torch.__version__.startswith("2.9.1")
except Exception:
    ok = False
if not ok:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu128",
    ])
try:
    import flash_attn_interface  # noqa: F401
except Exception:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl",
    ])
PY
}

download_data() {
  mkdir -p "${DATA_ROOT}"
  if compgen -G "${DATA_PATH}/fineweb_val_*.bin" > /dev/null && compgen -G "${DATA_PATH}/fineweb_train_*.bin" > /dev/null; then
    echo "Scylla data already present at ${DATA_PATH}"
    return
  fi
  echo "Downloading Scylla tokenized FineWeb into ${DATA_PATH}"
  HF_DATASET="${HF_DATASET}" DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["HF_DATASET"],
    repo_type="dataset",
    local_dir=os.environ["DATA_ROOT"],
    allow_patterns=["fineweb_scylla/*"],
)
PY
}

prepare_run_dir() {
  mkdir -p "${RUN_DIR}"
  cp "${HERE}/train_gpt.py" "${RUN_DIR}/train_gpt.py"
  cp "${HERE}/candidate.vocab" "${RUN_DIR}/candidate.vocab"
  cp "${HERE}/candidate.meta.npz" "${RUN_DIR}/candidate.meta.npz"
}

run_train() {
  local nproc iterations max_wall val_every log_every train_tokens val_batch chunkdiff_every chunkdiff_weight chunkdiff_seq_len ppm_subset
  if [[ "${MODE}" == "smoke" ]]; then
    nproc="${NPROC_PER_NODE:-1}"
    iterations="${ITERATIONS:-250}"
    max_wall="${MAX_WALLCLOCK_SECONDS:-0}"
    val_every="${VAL_LOSS_EVERY:-125}"
    log_every="${TRAIN_LOG_EVERY:-25}"
    train_tokens="${TRAIN_BATCH_TOKENS:-131072}"
    val_batch="${VAL_BATCH_SIZE:-131072}"
    chunkdiff_every="${CHUNKDIFF_EVERY:-4}"
    chunkdiff_weight="${CHUNKDIFF_WEIGHT:-0.006}"
    chunkdiff_seq_len="${CHUNKDIFF_SEQ_LEN:-192}"
    ppm_subset="${PPM_SUBSET_TOKENS:-262144}"
  else
    nproc="${NPROC_PER_NODE:-$(detect_gpus)}"
    iterations="${ITERATIONS:-20000}"
    max_wall="${MAX_WALLCLOCK_SECONDS:-0}"
    val_every="${VAL_LOSS_EVERY:-1000}"
    log_every="${TRAIN_LOG_EVERY:-100}"
    train_tokens="${TRAIN_BATCH_TOKENS:-786432}"
    val_batch="${VAL_BATCH_SIZE:-524288}"
    chunkdiff_every="${CHUNKDIFF_EVERY:-4}"
    chunkdiff_weight="${CHUNKDIFF_WEIGHT:-0.006}"
    chunkdiff_seq_len="${CHUNKDIFF_SEQ_LEN:-256}"
    ppm_subset="${PPM_SUBSET_TOKENS:-2000000}"
  fi

  cd "${RUN_DIR}"
  RUN_ID="${RUN_ID}" \
  SEED="${SEED:-42}" \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${RUN_DIR}/candidate.vocab" \
  TOKENIZER_META_PATH="${RUN_DIR}/candidate.meta.npz" \
  TOKENIZER_META_VALIDATE="${TOKENIZER_META_VALIDATE:-1}" \
  VOCAB_SIZE=998 \
  XSA_LAST_N=11 \
  USE_GPTQ=1 \
  GPTQ_RESERVE_MS="${GPTQ_RESERVE_MS:-9000}" \
  GPTQ_CALIB_SAMPLES="${GPTQ_CALIB_SAMPLES:-64}" \
  TTT_ENABLED="${TTT_ENABLED:-0}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2816}" \
  BIGRAM_DIM="${BIGRAM_DIM:-112}" \
  ITERATIONS="${iterations}" \
  MAX_WALLCLOCK_SECONDS="${max_wall}" \
  TRAIN_BATCH_TOKENS="${train_tokens}" \
  VAL_BATCH_SIZE="${val_batch}" \
  VAL_LOSS_EVERY="${val_every}" \
  TRAIN_LOG_EVERY="${log_every}" \
  WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}" \
  CHUNKDIFF_ENABLED="${CHUNKDIFF_ENABLED:-1}" \
  CHUNKDIFF_WEIGHT="${chunkdiff_weight}" \
  CHUNKDIFF_EVERY="${chunkdiff_every}" \
  CHUNKDIFF_SEQ_LEN="${chunkdiff_seq_len}" \
  CHUNKDIFF_MASK_MIN="${CHUNKDIFF_MASK_MIN:-0.15}" \
  CHUNKDIFF_MASK_MAX="${CHUNKDIFF_MASK_MAX:-0.55}" \
  CHUNKDIFF_STEPS="${CHUNKDIFF_STEPS:-8}" \
  PPM_ENABLED="${PPM_ENABLED:-1}" \
  PPM_ORDER="${PPM_ORDER:-5}" \
  PPM_SUBSET_TOKENS="${ppm_subset}" \
  PPM_LAMBDA_HI="${PPM_LAMBDA_HI:-0.35}" \
  PPM_LAMBDA_LO="${PPM_LAMBDA_LO:-0.02}" \
  PPM_CONF_THRESHOLD="${PPM_CONF_THRESHOLD:-0.08}" \
  torchrun --standalone --nproc_per_node="${nproc}" train_gpt.py
}

case "${MODE}" in
  install)
    install_deps
    ;;
  data)
    install_deps
    download_data
    ;;
  smoke|full)
    install_deps
    download_data
    prepare_run_dir
    run_train
    ;;
  *)
    echo "Usage: ./runpod_8xh100.sh [install|data|smoke|full]" >&2
    exit 2
    ;;
esac
