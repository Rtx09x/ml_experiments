# SP8192 Top Stack + Sparse 100M

## Problem

We want to test the current top Parameter Golf stack, but scaled up and trained with heavy sparsity from the start-ish.

Not diffusion.
No masking-token side quest.
This is the leaderboard recipe with a blunt sparsity experiment bolted on.

## Source Stack

Based on merged Parameter Golf PR:

- `Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT`
- Reported score: `val_bpb 1.0810` mean over 3 seeds
- Reported size: about `15.99 MB`
- Official PR: https://github.com/openai/parameter-golf/pull/1493

Core ingredients kept:

- SP8192 tokenizer and FineWeb shards
- 11-layer GPT
- XSA over last 11 layers
- 3-layer recurrence over layers 3 to 5
- recurrence enabled after training fraction `0.35`
- parallel residuals starting at layer 7
- QK gain `5.25`
- Muon/Adam optimizer split
- EMA `0.9965`
- GPTQ mixed quantization
- int6 matrices, int8 embeddings
- legal score-first TTT eval path

## Change

This variant changes the hypothesis:

- scale model width from `512` to `896`
- keep embedding width at `512`
- target roughly `100M` parameters (`102.3M` by local shape count)
- add magnitude sparsity on MLP and attention/projection matrices
- ramp sparsity from `0%` to `70%`
- default ramp: step `1000` to step `7000`
- final export keeps the reached/trained sparsity level instead of hard-pruning to `70%`
- training wall clock cap: `4800s` / `80 min`

The script still exports:

- `final_model.pt`
- `final_model_pre_quant.pt`
- `final_model.int6.ptz`
- `train.log`
- `metrics.csv`
- `metrics_summary.json`
- tokenizer copy
- zipped run folder

## Important Reality

70% sparsity is aggressive.

If BPB jumps, sparsity is the prime suspect.
If pre-quant BPB is fine but int6 BPB jumps, quantization/export is the suspect.
If both are fine, this is worth trying on an even stronger leaderboard base.

## RunPod Command

Use this in a notebook cell with `%%bash`.

```bash
cd /workspace
rm -rf ml_experiments
git clone -b boobie/sp8192-top-sparse-100m https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_top_sparse_100m/runpod_1xh100_80m.sh
./ml_experiments/parameter_golf/sp8192_top_sparse_100m/runpod_1xh100_80m.sh
```

With Hugging Face auth:

```bash
export HF_TOKEN='YOUR_TOKEN_HERE'
cd /workspace
rm -rf ml_experiments
git clone -b boobie/sp8192-top-sparse-100m https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/sp8192_top_sparse_100m/runpod_1xh100_80m.sh
./ml_experiments/parameter_golf/sp8192_top_sparse_100m/runpod_1xh100_80m.sh
```

## Override Knobs

Safer memory:

```bash
export TRAIN_BATCH_TOKENS=262144
export VAL_BATCH_TOKENS=262144
```

Less brutal sparsity:

```bash
export SPARSE_TARGET=0.50
```

Skip TTT eval if GPU money is burning:

```bash
export TTT_ENABLED=0
```

Disable sparsity:

```bash
export SPARSE_ENABLED=0
```

## Expected Logs

Look for these.

```text
model_params:...
layer_loop:enabled ...
sparsity:step:1000 target:...
sparsity:final target:0.700 actual:...
pre-quantization post-ema val_loss:... val_bpb:...
Serialized model quantized+brotli: ...
Total submission size quantized+brotli: ...
quantized val_loss:... val_bpb:...
quantized_sliding_window val_loss:... val_bpb:...
quantized_ttt val_loss:... val_bpb:...
zip_path=/workspace/sp8192_top_sparse_100m_runs/sp8192_top_sparse_100m_1xh100_80m.zip
```

## Read The Result

Use this order:

1. `pre-quantization post-ema val_bpb`
2. `quantized val_bpb`
3. `quantized_sliding_window val_bpb`
4. `quantized_ttt val_bpb`
5. `Total submission size`

If `pre-quantization post-ema` is bad, training/sparsity failed.
If only quantized eval is bad, export/GPTQ failed.
If TTT helps, good.
If TTT does nothing, it is mostly expensive decoration here.

## Notes From Failed 70% Export

The first 80-minute run reached:

```text
4449 val_bpb: 1.0691
trained sparsity: about 40%
```

Then the old export path hard-pruned to `70%` and destroyed it:

```text
pre-quantization post-ema val_bpb: 1.8265
```

That proved the training path was good and the final hard prune was bad.
This version does not do that hard final jump.
