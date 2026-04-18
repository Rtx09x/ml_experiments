# SP8192 Top Stack + Sparse 60M

## Problem

The 102M run proved sparsity helps up to a point.

It also proved 70% sparsity is a brick wall for this schedule.
Cool idea. Bad score.

## Source Stack

Based on merged Parameter Golf PR:

- `Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT`
- Reported score: `val_bpb 1.0810` mean over 3 seeds
- Reported size: about `15.99 MB`
- Official PR: https://github.com/openai/parameter-golf/pull/1493

Kept:

- SP8192 tokenizer and FineWeb shards
- 11-layer GPT
- XSA over last 11 layers
- 3-layer recurrence over layers 3 to 5
- recurrence enabled after training fraction `0.35`
- parallel residuals starting at layer 7
- QK gain `5.25`
- Muon momentum `0.97`
- TTT LR `0.01`
- Muon/Adam optimizer split
- EMA `0.9965`
- GPTQ mixed quantization
- int6 matrices, int8 embeddings
- legal score-first TTT eval path

## Change

This is the practical run:

- model width `672`
- embedding width `512`
- local shape count: `59,564,504` parameters
- sparsity target `50%`
- sparse ramp from step `1000` to step `4000`
- final export keeps the reached/trained sparsity level
- sparsity thresholding stays on GPU
- training wall clock cap: `4800s` / `80 min`

Why 60M/50%:

- 102M/40% gave `1.0691` before bad export
- 102M/62% gave `1.1077`
- 102M/70% degraded hard
- 60M/50% should fit the size target with less compression violence

Extra top-stack knobs included:

- `MUON_MOMENTUM=0.97`
- `TTT_LR=0.01`

Not included yet:

- parameter banking
- fused MLP kernel
- improved cross-lane parallel residuals
- eval-time hash embedding

Those are bigger ports. Worth doing, but not inside this clean sparsity-size experiment.

## Exports

The run package includes:

- `latest_train_model.pt`
- `checkpoint_step_<step>_bpb_<score>.pt`
- `final_model_ema_before_sparse.pt`
- `final_model.pt`
- `final_model_pre_quant.pt`
- `final_model.int6.ptz`
- `train.log`
- `metrics.csv`
- `metrics_summary.json`
- tokenizer copy
- zipped run folder

The step-named checkpoint exists because overwriting the best checkpoint is how you lose money twice. We tried that. It was stupid.

## RunPod Command

Use this in a notebook cell with `%%bash`.

```bash
cd /workspace
rm -rf ml_experiments
git clone -b boobie/sp8192-top-sparse-100m https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_top_sparse_60m/runpod_1xh100_80m.sh
./ml_experiments/parameter_golf/sp8192_top_sparse_60m/runpod_1xh100_80m.sh
```

With Hugging Face auth:

```bash
export HF_TOKEN='YOUR_TOKEN_HERE'
cd /workspace
rm -rf ml_experiments
git clone -b boobie/sp8192-top-sparse-100m https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_top_sparse_60m/runpod_1xh100_80m.sh
./ml_experiments/parameter_golf/sp8192_top_sparse_60m/runpod_1xh100_80m.sh
```

## Useful Overrides

More aggressive:

```bash
export SPARSE_TARGET=0.60
export SPARSE_END_STEP=4200
```

Skip expensive TTT eval:

```bash
export TTT_ENABLED=0
```

Safer memory:

```bash
export TRAIN_BATCH_TOKENS=262144
export VAL_BATCH_TOKENS=262144
```

## Expected Logs

```text
model_params:59564504
muon_momentum: 0.97
ttt_lr: 0.01
layer_loop:enabled ...
sparsity:step:4000 target:0.500 actual:0.500
Saved latest train checkpoint: latest_train_model.pt and checkpoint_step_4000_bpb_...
pre-quantization post-ema val_loss:... val_bpb:...
Serialized model quantized+brotli: ...
Total submission size quantized+brotli: ...
quantized val_loss:... val_bpb:...
quantized_sliding_window val_loss:... val_bpb:...
quantized_ttt val_loss:... val_bpb:...
zip_path=/workspace/sp8192_top_sparse_60m_runs/sp8192_top_sparse_60m_1xh100_80m.zip
```

## Read The Result

Use this order:

1. `pre-quantization post-ema val_bpb`
2. `quantized val_bpb`
3. `quantized_sliding_window val_bpb`
4. `quantized_ttt val_bpb`
5. `Total submission size`

If pre-quant is good and quantized is bad, fix GPTQ/export.
If pre-quant is bad, sparsity or width is wrong.
If TTT helps, keep it.
If TTT barely helps, stop paying for it.
