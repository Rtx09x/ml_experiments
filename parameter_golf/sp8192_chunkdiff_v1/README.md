# SP8192 Causal ChunkDiff V1

## What this is

This is the first legal diffusion-flavored non-record experiment.

It keeps the SP8192 top-stack causal language model and adds a training-only ChunkDiff denoising objective. Evaluation and BPB scoring remain normal autoregressive next-token prediction through `forward_logits()`.

## Legal status

Legal for Parameter Golf BPB scoring as long as `forward_logits()` remains the eval path:

- validation scoring uses only prefix tokens
- no true target token is provided to scoring
- no future validation tokens are visible during BPB eval
- ChunkDiff is only an auxiliary training loss

So this is not diffusion sampling at eval. The honest description is:

`Causal SP8192 LM with auxiliary chunk-diffusion denoising training.`

## Base stack

Copied from `sp8192_top_sparse_100m`:

- SP8192 tokenizer
- 11-layer top-stack transformer
- XSA on all layers
- recurrence over layers 3-5
- parallel residuals from layer 7
- QK gain 5.25
- Muon/Adam optimizer split
- EMA
- GPTQ mixed quantization
- brotli artifact compression
- legal sliding and TTT eval paths

## ChunkDiff V1 change

Added environment-controlled training loss:

```text
CHUNKDIFF_ENABLED=1
CHUNKDIFF_WEIGHT=0.03
CHUNKDIFF_CHUNK_SIZE=64
CHUNKDIFF_MASK_MIN=0.35
CHUNKDIFF_MASK_MAX=0.80
CHUNKDIFF_EVERY=2
CHUNKDIFF_MIN_STEP=0
```

For the auxiliary loss, the script chooses a non-prefix chunk inside the training sequence, masks part of that chunk, allows bidirectional attention only inside that selected chunk, and reconstructs masked tokens. Normal AR eval/export remains unchanged.

## Default V1 run

Defaults are intentionally safer than the 102M sparse run:

```text
MODEL_DIM=704
EMBEDDING_DIM=512
NUM_LAYERS=11
SPARSE_TARGET=0.40
SPARSE_START_STEP=1000
SPARSE_END_STEP=5000
TRAIN_BATCH_TOKENS=262144
VAL_BATCH_TOKENS=262144
NPROC_PER_NODE=8
```

Expected model size is about 65M parameters plus the tiny ChunkDiff mask embedding.

## Notebook command

```bash
%%bash
cd /workspace
rm -rf ml_experiments
git clone -b boobie/sp8192-chunkdiff-v1 https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_chunkdiff_v1/runpod_8xh100.sh
./ml_experiments/parameter_golf/sp8192_chunkdiff_v1/runpod_8xh100.sh
```

With Hugging Face auth:

```bash
%%bash
export HF_TOKEN='YOUR_TOKEN_HERE'
cd /workspace
rm -rf ml_experiments
git clone -b boobie/sp8192-chunkdiff-v1 https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_chunkdiff_v1/runpod_8xh100.sh
./ml_experiments/parameter_golf/sp8192_chunkdiff_v1/runpod_8xh100.sh
```

## What to watch

Logs should include:

```text
chunkdiff:enabled:1 weight:0.03 chunk_size:64 mask:[0.35,0.8] every:2
... train_loss: ... chunkdiff_loss: ...
... val_loss: ... val_bpb: ...
Total submission size quantized+brotli: ... bytes
quantized val_loss:... val_bpb:...
```

If BPB is weak early, first tune down diffusion pressure:

```bash
CHUNKDIFF_WEIGHT=0.01 CHUNKDIFF_EVERY=4
```

If artifact is too large, tune sparsity upward carefully:

```bash
SPARSE_TARGET=0.45
```

Do not jump to 70 percent sparsity; prior 100M runs showed that hard quality collapse starts well before 70 percent.
