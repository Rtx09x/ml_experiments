#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
REPO_DIR="$ROOT/ml_experiments"
PKG_DIR="$REPO_DIR/crystalfold/matbench_discovery_v5_runpod"
VENV_DIR="$ROOT/crystalfold_venv"
ART_DIR="$ROOT/crystalfold_v5_artifacts"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/Rtx09x/ml_experiments.git "$REPO_DIR"
fi

python3.10 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$PKG_DIR/requirements.txt" huggingface_hub hf_transfer

cat <<'EOF'
Environment ready.

Next step:
1. Export HF_TOKEN
2. Download dataset
3. Run smoke train
4. Run full MI300X train
EOF

