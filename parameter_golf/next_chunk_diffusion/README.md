# Next Chunk Diffusion

problem

Run an aggressive non-record Parameter Golf probe:

```text
~45-50M params
next-chunk masked denoising
50% training-time sparsity
int6 export
1x H100 SXM
90-minute training cap
```

cause

Pure bidirectional diffusion can cheat BPB by seeing future text. This experiment
keeps normal causal autoregressive BPB as the score path, then adds a
future-chunk denoising objective that does not reveal later chunks to the target
chunk.

fix

The model trains with two losses:

```text
AR loss:
  standard next-token cross entropy
  used for val_bpb

chunkdiff loss:
  choose a future chunk
  mask 50-95% of that chunk
  let the chunk attend bidirectionally inside itself
  do not let it attend to later chunks
  predict the original chunk tokens
```

This is not pure diffusion scoring. It is a causal LM with chunk-diffusion
training pressure. That is the honest label.

## Default RunPod Config

```bash
RUN_ID=next_chunk_diffusion_1xh100_90m
MODEL_DIM=576
NUM_LAYERS=11
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=4.0

MAX_WALLCLOCK_SECONDS=5400
TRAIN_BATCH_TOKENS=65536
VAL_BATCH_SIZE=65536
VAL_LOSS_EVERY=10000
TRAIN_LOG_EVERY=50

CHUNKDIFF_ENABLED=1
CHUNKDIFF_WEIGHT=0.05
CHUNKDIFF_CHUNK_SIZE=64
CHUNKDIFF_MASK_MIN=0.50
CHUNKDIFF_MASK_MAX=0.95
CHUNKDIFF_EVERY=1

SPARSE_ENABLED=1
SPARSE_TARGET=0.50
SPARSE_START_STEP=800
SPARSE_END_STEP=5000
SPARSE_EVERY=100
SPARSE_INCLUDE=mlp,attn

GPTQ_EXPORT=1
GPTQ_CALIB_BATCHES=32
FAST_EXPORT=1
SELECTIVE_PRUNE_MODE=full
LZMA_PRESET=6
QUANT_ROUNDTRIP_EVAL=1
TARGET_MB=15.9
```

## RunPod Command

Use this in a notebook `%%bash` cell:

```bash
cd /workspace
rm -rf ml_experiments
git clone https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/next_chunk_diffusion/runpod_1xh100.sh
./ml_experiments/parameter_golf/next_chunk_diffusion/runpod_1xh100.sh
```

The launcher writes:

```text
/workspace/next_chunk_diffusion_runs/<RUN_ID>/
  README.md
  env.json
  metrics.csv
  metrics_summary.json
  runpod_1xh100.sh
  train.log
  train_gpt.py
  verify_packaging.py
  final_model.pt
  final_model_pre_quant.pt
  final_model.int6.ptz
  tokenizers/fineweb_1024_bpe.model

/workspace/next_chunk_diffusion_runs/<RUN_ID>.zip
```

Only training is capped. Export, final eval, artifact copy, and zipping are
allowed to finish.

## Recovery Export

If training saved `final_model_pre_quant.pt` but export failed, rerun export
without retraining:

```bash
cd /workspace
rm -rf ml_experiments
git clone https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/next_chunk_diffusion/runpod_1xh100.sh
RUN_ID=next_chunk_diffusion_1xh100_90m_salvage \
EXPORT_ONLY_CHECKPOINT=/workspace/parameter-golf/final_model_pre_quant.pt \
./ml_experiments/parameter_golf/next_chunk_diffusion/runpod_1xh100.sh
```

## Export Fixes Compared To blockdiff_sparse_ticket

The old export path could spend minutes in a repeated compression binary search.
This run defaults to:

```text
SELECTIVE_PRUNE_MODE=full
LZMA_PRESET=6
```

Meaning:

```text
zero all +/-1 int6 values once
compress once for size estimate
write final int6 package
```

If exact size tuning is needed later:

```text
SELECTIVE_PRUNE_MODE=binary
LZMA_PRESET=9
```

Do that only when the BPB is worth it. Paying for slow packaging on a bad score
is just expensive denial.

## What To Watch

Useful log lines:

```text
model_params:...
chunkdiff:enabled:1 weight:...
sparsity:step:... target:... actual:...
step:... train_loss:... chunkdiff_loss:...
step:... val_loss:... val_bpb:...
DIAGNOSTIC post_ema val_loss:... val_bpb:...
selective_prune:mode:full ...
Serialized model int6+lzma: ...
Total submission size int6+lzma: ...
final_int6_roundtrip val_loss:... val_bpb:...
```

Bad signs:

```text
chunkdiff_loss collapses near zero very early
val_bpb stalls above the previous run
sparsity step causes obvious loss jump
int6 roundtrip BPB is much worse than post_ema BPB
final int6 package misses 16MB by a lot
```

This is intentionally aggressive. It is a stress test, not clean science.
