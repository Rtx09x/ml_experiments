# Scylla ChunkDiff Final

Non-record 16MB lane experiment: Scylla tokenizer + full GPTQ/XSA/FA3 stack with a small training-only ChunkDiff denoising auxiliary loss plus a score-first token PPM mixture diagnostic.

## Why this lane

The SP8192 diffusion+sparsity runs learned, but they did not export cleanly under 16MB with enough quality. This branch uses the strongest known substrate instead:

- Scylla tokenizer, 998 vocab
- XSA on all layers
- full Hessian GPTQ int6 + LZMA
- bigram hash features
- FlashAttention 3
- no TTT by default

ChunkDiff is added as an auxiliary training signal: corrupt a short chunk with a timestep-conditioned mask embedding and recover the original tokens. The final legal evaluator still uses the normal causal model path.

Recent live PRs show that PPM-D byte mixtures are the largest open frontier jump. This branch adds the Scylla-safe version first: a token-level score-first PPM mixture over the 998-token stream, reported as `token_ppm_mix`. It does not require CaseOps byte sidecars, so it is cheap to test immediately. If it moves the score, the next branch should build a true Scylla byte-sidecar PPM path.

## Smoke command

Paste your token, then run this on 1 H100 first:

```bash
set -euo pipefail

export HF_TOKEN='PASTE_HF_TOKEN_HERE'
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_VERBOSITY=info

cd /workspace
rm -rf ml_experiments
git clone -b boobie/scylla-chunkdiff-final https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/scylla_chunkdiff_final/runpod_8xh100.sh

RUN_MODE=smoke \
NPROC_PER_NODE=1 \
ITERATIONS=250 \
VAL_LOSS_EVERY=125 \
TRAIN_LOG_EVERY=25 \
./ml_experiments/parameter_golf/scylla_chunkdiff_final/runpod_8xh100.sh
```

## Full 8xH100 command

Use this after the smoke reaches training logs and produces an int6 roundtrip eval:

```bash
set -euo pipefail

export HF_TOKEN='PASTE_HF_TOKEN_HERE'
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_VERBOSITY=info

cd /workspace
rm -rf ml_experiments
git clone -b boobie/scylla-chunkdiff-final https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/scylla_chunkdiff_final/runpod_8xh100.sh

RUN_MODE=full \
NPROC_PER_NODE=8 \
SEED=42 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=100 \
CHUNKDIFF_WEIGHT=0.006 \
CHUNKDIFF_EVERY=4 \
CHUNKDIFF_SEQ_LEN=256 \
PPM_ENABLED=1 \
PPM_SUBSET_TOKENS=2000000 \
WARMDOWN_ITERS=4000 \
./ml_experiments/parameter_golf/scylla_chunkdiff_final/runpod_8xh100.sh
```

## If it underperforms

Do not jump to a bigger model first. Run a second seed or slightly lower auxiliary pressure:

```bash
CHUNKDIFF_WEIGHT=0.003 CHUNKDIFF_EVERY=4 SEED=1337 RUN_MODE=full NPROC_PER_NODE=8 ITERATIONS=20000 ./ml_experiments/parameter_golf/scylla_chunkdiff_final/runpod_8xh100.sh
```

The success metric is the final `final_int6_sliding_window_s64_exact` / `final_int6_roundtrip_exact` bpb plus the `token_ppm_mix` line. If `token_ppm_mix` is worse than neural-only, set `PPM_ENABLED=0`; if it is better, raise `PPM_SUBSET_TOKENS` for the final run.
