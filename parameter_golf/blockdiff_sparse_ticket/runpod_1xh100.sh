#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-${WORKSPACE_ROOT}/parameter-golf}"
EXPERIMENT_REPO_DIR="${EXPERIMENT_REPO_DIR:-${WORKSPACE_ROOT}/ml_experiments}"
RUN_ID="${RUN_ID:-blockdiff_sparse_ticket_1xh100}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/blockdiff_sparse_ticket_runs}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_PATH="${RUN_DIR}/train.log"
SUMMARY_PATH="${RUN_DIR}/metrics_summary.json"
METRICS_CSV_PATH="${RUN_DIR}/metrics.csv"
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

mkdir -p "${PARAMETER_GOLF_DIR}/working/blockdiff_sparse_ticket"
cp "${EXPERIMENT_REPO_DIR}/parameter_golf/blockdiff_sparse_ticket/train_gpt.py" \
  "${PARAMETER_GOLF_DIR}/working/blockdiff_sparse_ticket/train_gpt.py"
cp "${EXPERIMENT_REPO_DIR}/parameter_golf/blockdiff_sparse_ticket/README.md" \
  "${PARAMETER_GOLF_DIR}/working/blockdiff_sparse_ticket/README.md"

mkdir -p "${RUN_DIR}"
cp "${EXPERIMENT_REPO_DIR}/parameter_golf/blockdiff_sparse_ticket/train_gpt.py" "${RUN_DIR}/train_gpt.py"
cp "${EXPERIMENT_REPO_DIR}/parameter_golf/blockdiff_sparse_ticket/README.md" "${RUN_DIR}/README.md"
cp "${EXPERIMENT_REPO_DIR}/parameter_golf/blockdiff_sparse_ticket/runpod_1xh100.sh" "${RUN_DIR}/runpod_1xh100.sh"

cd "${PARAMETER_GOLF_DIR}"

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS:-80}"

export RUN_ID
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Cap training only. Export, eval, artifact copy, and zipping are allowed to finish.
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}"
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

python3 - <<'PY' > "${RUN_DIR}/env.json"
import json
import os

keys = [
    "RUN_ID", "DATA_PATH", "TOKENIZER_PATH", "VOCAB_SIZE", "MAX_WALLCLOCK_SECONDS",
    "TRAIN_BATCH_TOKENS", "VAL_BATCH_SIZE", "VAL_LOSS_EVERY", "TRAIN_LOG_EVERY",
    "BLOCKDIFF_ENABLED", "BLOCKDIFF_WEIGHT", "BLOCKDIFF_BLOCK_SIZE",
    "BLOCKDIFF_MASK_MIN", "BLOCKDIFF_MASK_MAX", "BLOCKDIFF_EVERY",
    "BLOCKDIFF_MIN_STEP", "SPARSE_ENABLED", "SPARSE_TARGET",
    "SPARSE_START_STEP", "SPARSE_END_STEP", "SPARSE_EVERY", "SPARSE_INCLUDE",
    "GPTQ_CALIB_BATCHES", "GATED_ATTENTION", "VALUE_RESIDUAL", "TRIGRAM",
    "DTG_ENABLED", "LAWA_ENABLED",
]
print(json.dumps({k: os.environ.get(k) for k in keys}, indent=2, sort_keys=True))
PY

set +e
torchrun --standalone --nproc_per_node=1 working/blockdiff_sparse_ticket/train_gpt.py \
  2>&1 | tee "${LOG_PATH}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

python3 - <<'PY' "${LOG_PATH}" "${SUMMARY_PATH}" "${METRICS_CSV_PATH}" "${TRAIN_RC}"
import csv
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
return_code = int(sys.argv[4])

patterns = {
    "train": re.compile(
        r"step:(?P<step>\d+)/(?P<iters>\d+)\s+train_loss:(?P<train_loss>[0-9.]+)\s+"
        r"blockdiff_loss:(?P<blockdiff_loss>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms\s+"
        r"step_avg:(?P<step_avg_ms>[0-9.]+)ms\s+tok/s:(?P<tok_per_sec>[0-9.]+)"
    ),
    "val": re.compile(
        r"step:(?P<step>\d+)/(?P<iters>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+"
        r"val_bpb:(?P<val_bpb>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms"
    ),
    "final": re.compile(
        r"(?P<name>final_[^ ]+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
    ),
    "post_ema": re.compile(r"DIAGNOSTIC post_ema val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"),
    "size": re.compile(r"Total submission size [^:]+:\s*(?P<bytes>\d+)\s+bytes"),
    "model_bytes": re.compile(r"Serialized model(?: int6\+lzma)?:\s*(?P<bytes>\d+)\s+bytes"),
    "sparsity": re.compile(r"sparsity:(?P<kind>step:\d+|final)\s+target:(?P<target>[0-9.]+)\s+actual:(?P<actual>[0-9.]+)"),
}

rows = []
summary = {"return_code": return_code, "log_path": str(log_path)}
if log_path.exists():
    for line in log_path.read_text(errors="replace").splitlines():
        if m := patterns["train"].search(line):
            row = {"kind": "train", **m.groupdict()}
            rows.append(row)
            summary["last_train"] = row
        elif m := patterns["val"].search(line):
            row = {"kind": "val", **m.groupdict()}
            rows.append(row)
            summary["last_val"] = row
        elif m := patterns["final"].search(line):
            row = {"kind": "final", **m.groupdict()}
            rows.append(row)
            summary[m.group("name")] = row
        elif m := patterns["post_ema"].search(line):
            summary["post_ema"] = m.groupdict()
        elif m := patterns["size"].search(line):
            summary["total_submission_bytes"] = int(m.group("bytes"))
        elif m := patterns["model_bytes"].search(line):
            summary.setdefault("serialized_model_bytes", []).append(int(m.group("bytes")))
        elif m := patterns["sparsity"].search(line):
            summary["last_sparsity"] = m.groupdict()

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
keys = []
for row in rows:
    for key in row:
        if key not in keys:
            keys.append(key)
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
PY

for artifact in final_model.pt final_model_pre_quant.pt final_model.int6.ptz; do
  if [ -f "${PARAMETER_GOLF_DIR}/${artifact}" ]; then
    cp "${PARAMETER_GOLF_DIR}/${artifact}" "${RUN_DIR}/${artifact}"
  fi
done

if [ -d "${PARAMETER_GOLF_DIR}/data/tokenizers" ]; then
  mkdir -p "${RUN_DIR}/tokenizers"
  cp "${PARAMETER_GOLF_DIR}/data/tokenizers/fineweb_1024_bpe.model" "${RUN_DIR}/tokenizers/" 2>/dev/null || true
fi

python3 - <<'PY' "${RUN_DIR}" "${ZIP_PATH}"
import sys
import zipfile
from pathlib import Path

run_dir = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
zip_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(run_dir.parent))
print(zip_path)
PY

echo "run_dir=${RUN_DIR}"
echo "zip_path=${ZIP_PATH}"
echo "train_return_code=${TRAIN_RC}"

exit "${TRAIN_RC}"
