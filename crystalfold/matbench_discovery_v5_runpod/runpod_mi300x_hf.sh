#!/usr/bin/env bash
set -euo pipefail

# Paste your HF token here if you want the least interactive workflow.
# Yes, this is less secure. Keep the pod private and rotate the token later.
HF_TOKEN="${HF_TOKEN:-PASTE_YOUR_HF_TOKEN_HERE}"

HF_REPO="${HF_REPO:-Rtx09/crystalfold-v5-artifacts}"
ROOT="${ROOT:-/workspace}"
REPO_DIR="$ROOT/ml_experiments"
PKG_DIR="$REPO_DIR/crystalfold/matbench_discovery_v5_runpod"
VENV_DIR="$ROOT/crystalfold_venv"
ART_DIR="$ROOT/crystalfold_v5_artifacts"
RUNS_DIR="$ART_DIR/runs"
BRANCH="${BRANCH:-boobie/sp8192-chunkdiff-v2}"
MODE="${1:-smoke}"

if [[ "$HF_TOKEN" == "PASTE_YOUR_HF_TOKEN_HERE" ]]; then
  echo "Set HF_TOKEN in this script or export HF_TOKEN before running."
  exit 1
fi

echo "==> setup"
apt-get update -y
apt-get install -y python3.10 python3.10-venv python3.10-dev build-essential git curl

if [[ ! -d "$REPO_DIR" ]]; then
  git clone -b "$BRANCH" https://github.com/Rtx09x/ml_experiments.git "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi

rm -rf "$VENV_DIR"
python3.10 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$PKG_DIR/requirements.txt" huggingface_hub hf_transfer

echo "==> download dataset"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN
"$VENV_DIR/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO"] if "HF_REPO" in os.environ else "Rtx09/crystalfold-v5-artifacts",
    repo_type="dataset",
    local_dir="/workspace/crystalfold_v5_artifacts",
    token=os.environ["HF_TOKEN"],
)
print("download done")
PY

echo "==> rocm sanity"
"$VENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

mkdir -p "$RUNS_DIR"

if [[ "$MODE" == "smoke" ]]; then
  echo "==> 1 epoch smoke train"
  cd "$PKG_DIR"
  "$VENV_DIR/bin/python" -m crystalfold_v5.cli train \
    --dataset "$ART_DIR/mpt_v5.h5" \
    --run-dir "$RUNS_DIR/mi300x_smoke" \
    --epochs 1 \
    --batch-size 256 \
    --workers 8 \
    --amp bf16 \
    --no-pin-memory
elif [[ "$MODE" == "full" ]]; then
  echo "==> full train"
  cd "$PKG_DIR"
  "$VENV_DIR/bin/python" -m crystalfold_v5.cli train \
    --dataset "$ART_DIR/mpt_v5.h5" \
    --run-dir "$RUNS_DIR/mi300x_v5" \
    --epochs 80 \
    --batch-size 256 \
    --workers 8 \
    --amp bf16 \
    --no-pin-memory
else
  echo "Unknown mode: $MODE"
  echo "Use: smoke | full"
  exit 1
fi

