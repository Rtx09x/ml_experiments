# CrystalFold Matbench Discovery V5 RunPod Pipeline

This folder is a clean, runnable package for the CrystalFold Matbench Discovery
experiment. It is designed for two-pod execution:

1. CPU pod: download official public datasets, build HDF5 graph artifacts, write
   manifests and normalization stats to a shared network volume.
2. GPU pod: copy the processed HDF5 to local disk if requested, then train.

The original local `v5` research scripts had useful ideas and several expensive
bugs. This package fixes the important ones:

- training loader reads the same flat HDF5 schema written by preprocessing
- atom feature dimension is `37`, matching the magmom-enabled dataset builder
- autograd forces differentiate total crystal energy, not naked energy/atom
- periodic image offsets are stored and used in differentiable edge vectors
- graph attention normalizes per destination atom, not globally across edges
- dataset download is anchored on official Figshare article metadata

## Sources

- MPtrj Figshare article: `23713842`
- Matbench Discovery Figshare article: `22715158`
- MPtrj target: `energy_per_atom`, the MP2020-corrected total energy per atom
- WBM structures: official Matbench Discovery initial structures

## Install

```bash
cd /workspace/ml_experiments/crystalfold/matbench_discovery_v5_runpod
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Command 1: CPU Preprocessing

Use this on a CPU pod with the network volume mounted at `/workspace/vol`.
For the final build, prefer 32 vCPU / 64 GB RAM and `--workers 24`.

```bash
python -m crystalfold_v5.cli preprocess \
  --root /workspace/vol/crystalfold_v5 \
  --workers 24 \
  --snapshots 4 \
  --include-wbm
```

For a fast smoke run before burning time:

```bash
python -m crystalfold_v5.cli preprocess \
  --root /workspace/vol/crystalfold_v5_smoke \
  --workers 4 \
  --max-materials 200 \
  --include-wbm \
  --max-wbm 200
```

## Command 2: GPU Training

Use this on the H100 pod with the same network volume mounted. `--local-copy`
copies the processed HDF5 to local pod storage first, because making an H100
wait on network storage is just disrespectful.

```bash
python -m crystalfold_v5.cli train \
  --dataset /workspace/vol/crystalfold_v5/processed/mpt_v5.h5 \
  --run-dir /workspace/vol/crystalfold_v5/runs/h100_v5 \
  --epochs 80 \
  --batch-size 192 \
  --workers 8 \
  --amp bf16 \
  --compile \
  --local-copy /workspace
```

## MI300X

There is a separate ROCm-oriented note in [MI300X.md](./MI300X.md) and a helper
bootstrap script in [runpod_1xmi300x.sh](./runpod_1xmi300x.sh).

The main practical differences:

- do not use `--compile` on the first MI300X run
- keep `--amp bf16`
- start with `--batch-size 256`
- use `--no-pin-memory`

If you want the shortest possible setup, use
[MI300X_QUICKSTART.md](./MI300X_QUICKSTART.md) and
[runpod_mi300x_hf.sh](./runpod_mi300x_hf.sh).

## Notes

- The CPU build stores both raw official downloads and processed files under
  `--root`, so the GPU pod does not need internet.
- The processed dataset includes `manifest.json`, `scaler.json`, and source
  Figshare metadata for reproducibility.
- The WBM preprocessing artifact is written to
  `processed/wbm_initial_v5.h5`; training does not need it, evaluation does.
