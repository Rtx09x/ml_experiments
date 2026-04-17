# BlockDiff Sparse Ticket

problem

Test a non-record Parameter Golf idea without touching the existing frontier run:
keep the working XSA/EMA/GPTQ/int6 stack, add block-diffusion training pressure,
then prune weights during training and before export.

cause

Pure diffusion language modeling does not naturally match the challenge's AR
compression score. This experiment keeps normal AR training/eval as the score
path and adds a block masked-denoising objective as training pressure.

This is real block diffusion training, not pure LLaDA-style full-sequence
diffusion. Blocks are selected, corrupted with a learned mask embedding, then
denoised at masked positions. During that auxiliary pass, previous blocks stay
causal and the selected block gets bidirectional self-attention. Normal BPB
evaluation remains purely causal and unchanged.

fix

New knobs:

```bash
BLOCKDIFF_ENABLED=1
BLOCKDIFF_WEIGHT=0.08
BLOCKDIFF_BLOCK_SIZE=16
BLOCKDIFF_MASK_MIN=0.25
BLOCKDIFF_MASK_MAX=0.75
BLOCKDIFF_EVERY=1
BLOCKDIFF_MIN_STEP=0

SPARSE_ENABLED=1
SPARSE_TARGET=0.30
SPARSE_START_STEP=1000
SPARSE_END_STEP=8000
SPARSE_EVERY=100
SPARSE_INCLUDE=mlp,attn
```

First smoke:

```bash
RUN_ID=blockdiff_sparse_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=20 \
BLOCKDIFF_ENABLED=1 \
BLOCKDIFF_WEIGHT=0.03 \
SPARSE_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 working/blockdiff_sparse_ticket/train_gpt.py
```

Then sparsity smoke:

```bash
RUN_ID=blockdiff_sparse_prune_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=1000 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
BLOCKDIFF_ENABLED=1 \
BLOCKDIFF_WEIGHT=0.03 \
SPARSE_ENABLED=1 \
SPARSE_TARGET=0.30 \
SPARSE_START_STEP=200 \
SPARSE_END_STEP=800 \
SPARSE_EVERY=50 \
torchrun --standalone --nproc_per_node=1 working/blockdiff_sparse_ticket/train_gpt.py
```

Do not start with 90% sparsity. That is not research, that is a deletion spell.

1x H100 SXM run with one-hour training cap:

Shortest path on the official Parameter Golf RunPod template:

```bash
cd /workspace
git clone https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/blockdiff_sparse_ticket/runpod_1xh100.sh
./ml_experiments/parameter_golf/blockdiff_sparse_ticket/runpod_1xh100.sh
```

The launcher writes:

```text
/workspace/blockdiff_sparse_ticket_runs/<RUN_ID>/
  README.md
  env.json
  metrics.csv
  metrics_summary.json
  runpod_1xh100.sh
  train.log
  train_gpt.py
  final_model.pt
  final_model_pre_quant.pt
  final_model.int6.ptz
  tokenizers/fineweb_1024_bpe.model

/workspace/blockdiff_sparse_ticket_runs/<RUN_ID>.zip
```

Only training is capped by `MAX_WALLCLOCK_SECONDS`. Export, final eval,
artifact copy, and zipping are not wrapped in an outer kill timeout.

Recover export from a saved pre-quant checkpoint:

```bash
cd /workspace
rm -rf ml_experiments
git clone https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/blockdiff_sparse_ticket/runpod_1xh100.sh
RUN_ID=blockdiff_sparse_ticket_1xh100_salvage \
EXPORT_ONLY_CHECKPOINT=/workspace/parameter-golf/final_model_pre_quant.pt \
./ml_experiments/parameter_golf/blockdiff_sparse_ticket/runpod_1xh100.sh
```

Manual equivalent:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
git clone https://github.com/Rtx09x/ml_experiments.git
mkdir -p parameter-golf/working/blockdiff_sparse_ticket
cp ml_experiments/parameter_golf/blockdiff_sparse_ticket/train_gpt.py parameter-golf/working/blockdiff_sparse_ticket/train_gpt.py
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
RUN_ID=blockdiff_sparse_ticket_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=3600 \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_SIZE=65536 \
VAL_LOSS_EVERY=10000 \
TRAIN_LOG_EVERY=50 \
BLOCKDIFF_ENABLED=1 \
BLOCKDIFF_WEIGHT=0.03 \
BLOCKDIFF_BLOCK_SIZE=16 \
BLOCKDIFF_MASK_MIN=0.25 \
BLOCKDIFF_MASK_MAX=0.75 \
SPARSE_ENABLED=1 \
SPARSE_TARGET=0.30 \
SPARSE_START_STEP=500 \
SPARSE_END_STEP=4000 \
SPARSE_EVERY=100 \
torchrun --standalone --nproc_per_node=1 working/blockdiff_sparse_ticket/train_gpt.py
```
