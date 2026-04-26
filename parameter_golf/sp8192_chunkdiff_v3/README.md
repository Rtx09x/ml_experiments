# SP8192 Causal ChunkDiff V3

## What this is

V3 keeps the top SP8192 causal leaderboard stack for legal BPB scoring, and makes the diffusion part more explicit than V1.

Evaluation still uses the causal `forward_logits()` path only. Diffusion is training-time only.

## Diffusion-specific pieces

V3 adds these over V1:

- learned mask embedding for corrupted chunk tokens
- learned diffusion timestep embedding
- timestep-dependent mask schedule over `CHUNKDIFF_STEPS`
- dedicated denoising adapter before the tied token projection
- ChunkDiff runs every step by default
- cropped ChunkDiff window to keep masked SDPA memory safe

Default diffusion knobs:

```text
CHUNKDIFF_ENABLED=1
CHUNKDIFF_WEIGHT=0.03
CHUNKDIFF_CHUNK_SIZE=64
CHUNKDIFF_MASK_MIN=0.20
CHUNKDIFF_MASK_MAX=0.90
CHUNKDIFF_EVERY=1
CHUNKDIFF_SEQ_LEN=128
CHUNKDIFF_STEPS=8
CHUNKDIFF_TIME_COND=1
CHUNKDIFF_DENOISE_ADAPTER=1
```

## Legal status

Legal BPB scoring is unchanged:

- `forward_logits(input_ids)` remains causal
- validation scoring sees only prefix tokens
- no future validation tokens are used for the scored target
- ChunkDiff denoising is only an auxiliary training objective

Good description for non-record submission:

```text
A causal SP8192 TopStack language model trained with a time-conditioned chunk diffusion denoising branch. BPB evaluation uses the legal causal forward path only.
```

## Model size

Default V3 is based on the 100M top-stack lane:

```text
MODEL_DIM=896
EMBEDDING_DIM=512
NUM_LAYERS=11
MLP_MULT=4
VOCAB_SIZE=8192
SPARSE_TARGET=0.50
MATRIX_BITS=4
EXPORT_EMA=0
```

The denoising adapter adds about 262k parameters plus tiny timestep/mask embeddings.

## Run command

```bash
%%bash
set -euo pipefail

cd /workspace
rm -rf ml_experiments

git clone -b boobie/sp8192-chunkdiff-v3 https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_chunkdiff_v3/runpod_8xh100.sh

HF_HUB_VERBOSITY=info \
TRAIN_SHARDS=128 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=999999 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=100 \
CHUNKDIFF_SEQ_LEN=128 \
SPARSE_TARGET=0.50 \
MATRIX_BITS=4 \
EXPORT_EMA=0 \
./ml_experiments/parameter_golf/sp8192_chunkdiff_v3/runpod_8xh100.sh
```

If it still hits memory pressure, lower only the ChunkDiff aux window:

```bash
CHUNKDIFF_SEQ_LEN=128
```

If the run is stable but too slow, reduce diffusion pressure:

```bash
CHUNKDIFF_WEIGHT=0.02 CHUNKDIFF_EVERY=2
```

## What to watch

Expected startup lines:

```text
model_params:...
chunkdiff:enabled:1 weight:0.03 chunk_size:64 mask:[0.2,0.9] every:1 min_step:0 seq_len:128 steps:8 time_cond:1 denoise_adapter:1
```

Then watch:

```text
train_loss: ... chunkdiff_loss: ...
val_bpb: ...
Total submission size quantized+brotli: ... bytes
```
