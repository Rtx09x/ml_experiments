#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-${WORKSPACE_ROOT}/parameter-golf}"
EXPERIMENT_REPO_DIR="${EXPERIMENT_REPO_DIR:-${WORKSPACE_ROOT}/ml_experiments}"
EXPERIMENT_DIR="${EXPERIMENT_REPO_DIR}/parameter_golf/leaderboard_final_sp8192"
RUN_ID="${RUN_ID:-leaderboard_final_sp8192_seed${SEED:-42}}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/leaderboard_final_sp8192_runs}"
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

mkdir -p "${PARAMETER_GOLF_DIR}/working/leaderboard_final_sp8192"
cp "${EXPERIMENT_DIR}/train_gpt.py" "${PARAMETER_GOLF_DIR}/working/leaderboard_final_sp8192/train_gpt.py"
cp "${EXPERIMENT_DIR}/SOURCE_RECORD_README.md" "${PARAMETER_GOLF_DIR}/working/leaderboard_final_sp8192/SOURCE_RECORD_README.md"
cp "${EXPERIMENT_DIR}/runpod_8xh100.sh" "${PARAMETER_GOLF_DIR}/working/leaderboard_final_sp8192/runpod_8xh100.sh"

mkdir -p "${RUN_DIR}"
cp "${EXPERIMENT_DIR}/train_gpt.py" "${RUN_DIR}/train_gpt.py"
cp "${EXPERIMENT_DIR}/SOURCE_RECORD_README.md" "${RUN_DIR}/SOURCE_RECORD_README.md"
cp "${EXPERIMENT_DIR}/source_submission.json" "${RUN_DIR}/source_submission.json"
cp "${EXPERIMENT_DIR}/runpod_8xh100.sh" "${RUN_DIR}/runpod_8xh100.sh"

cd "${PARAMETER_GOLF_DIR}"

python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = []
for package, module in [("brotli", "brotli"), ("sentencepiece", "sentencepiece")]:
    if importlib.util.find_spec(module) is None:
        missing.append(package)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
PY

TRAIN_SHARDS="${TRAIN_SHARDS:-128}"
DATASET_DIR="${PARAMETER_GOLF_DIR}/data/datasets/fineweb10B_sp8192"
TOKENIZER_FILE="${PARAMETER_GOLF_DIR}/data/tokenizers/fineweb_8192_bpe.model"
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
  MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}" \
    python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards "${TRAIN_SHARDS}"
fi

rm -f final_model.pt final_model_pre_quant.pt final_model.int6.ptz
mkdir -p logs

export RUN_ID
export DATA_DIR="${DATA_DIR:-./data/}"
export SEED="${SEED:-42}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export MODEL_DIM="${MODEL_DIM:-512}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export XSA_LAST_N="${XSA_LAST_N:-11}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-4.0}"
export NUM_LOOPS="${NUM_LOOPS:-2}"
export LOOP_START="${LOOP_START:-3}"
export LOOP_END="${LOOP_END:-5}"
export ENABLE_LOOPING_AT="${ENABLE_LOOPING_AT:-0.35}"
export PARALLEL_RESIDUAL_START="${PARALLEL_RESIDUAL_START:-7}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}"

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export MATRIX_LR="${MATRIX_LR:-0.022}"
export MUON_WD="${MUON_WD:-0.095}"
export EMA_DECAY="${EMA_DECAY:-0.9965}"
export WARMDOWN_FRAC="${WARMDOWN_FRAC:-0.72}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"

export COMPRESSOR="${COMPRESSOR:-brotli}"
export MATRIX_BITS="${MATRIX_BITS:-6}"
export EMBED_BITS="${EMBED_BITS:-8}"
export MATRIX_CLIP_SIGMAS="${MATRIX_CLIP_SIGMAS:-12.85}"
export EMBED_CLIP_SIGMAS="${EMBED_CLIP_SIGMAS:-20.0}"
export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-64}"
export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-12}"

export SLIDING_WINDOW_ENABLED="${SLIDING_WINDOW_ENABLED:-1}"
export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_LR="${TTT_LR:-0.005}"
export TTT_EPOCHS="${TTT_EPOCHS:-3}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"

python3 - <<'PY' > "${RUN_DIR}/env.json"
import json, os
keys = [
    "RUN_ID", "SEED", "VOCAB_SIZE", "MODEL_DIM", "QK_GAIN_INIT",
    "MAX_WALLCLOCK_SECONDS", "ITERATIONS", "TRAIN_BATCH_TOKENS",
    "VAL_LOSS_EVERY", "MATRIX_LR", "MUON_WD", "EMA_DECAY",
    "MATRIX_BITS", "EMBED_BITS", "MATRIX_CLIP_SIGMAS", "EMBED_CLIP_SIGMAS",
    "SLIDING_WINDOW_ENABLED", "TTT_ENABLED", "TTT_LR", "TTT_EPOCHS",
    "TTT_MOMENTUM", "TTT_CHUNK_TOKENS",
]
print(json.dumps({k: os.environ.get(k) for k in keys}, indent=2, sort_keys=True))
PY

set +e
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" working/leaderboard_final_sp8192/train_gpt.py 2>&1 | tee "${LOG_PATH}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

python3 - "${LOG_PATH}" "${RUN_DIR}/metrics_summary.json" "${TRAIN_RC}" <<'PY'
import json, re, sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
return_code = int(sys.argv[3])
summary = {"train_return_code": return_code}

patterns = {
    "last_val": re.compile(r"(?P<step>\d+)/(?P<iters>\d+) val_loss: (?P<val_loss>[0-9.]+) val_bpb: (?P<val_bpb>[0-9.]+)"),
    "pre_quant": re.compile(r"pre-quantization post-ema val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"),
    "quantized": re.compile(r"quantized val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"),
    "sliding": re.compile(r"quantized_sliding_window val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"),
    "ttt": re.compile(r"quantized_ttt val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"),
    "size": re.compile(r"Total submission size quantized\+[^:]+: (?P<bytes>\d+) bytes"),
}

for line in log_path.read_text(errors="replace").splitlines():
    for key, pattern in patterns.items():
        if match := pattern.search(line):
            summary[key] = match.groupdict()

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

for artifact in final_model.pt final_model_pre_quant.pt final_model.int6.ptz logs/"${RUN_ID}".txt; do
  if [ -f "${PARAMETER_GOLF_DIR}/${artifact}" ]; then
    cp "${PARAMETER_GOLF_DIR}/${artifact}" "${RUN_DIR}/$(basename "${artifact}")"
  fi
done

python3 - "${RUN_DIR}" "${ZIP_PATH}" <<'PY'
from pathlib import Path
import sys, zipfile
run_dir = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
zip_path.parent.mkdir(parents=True, exist_ok=True)
files = [p for p in sorted(run_dir.rglob("*")) if p.is_file()]
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in files:
        zf.write(p, p.relative_to(run_dir.parent))
print(f"zip_path={zip_path}")
print(f"zip_bytes={zip_path.stat().st_size}")
PY

echo "run_dir=${RUN_DIR}"
echo "zip_path=${ZIP_PATH}"
echo "train_return_code=${TRAIN_RC}"
exit "${TRAIN_RC}"
