# Leaderboard Final SP8192

This is the official current leaderboard stack packaged for clean RunPod execution from `ml_experiments`.

Source record:

- `openai/parameter-golf`
- `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`
- reported 3-seed mean: `1.0810 BPB`
- seed 42: `quantized_ttt val_bpb 1.08079352`
- total artifact size: `15,991,930 bytes`

This is not a diffusion or sparse branch. It is the strongest legal leaderboard lane we have evidence for.

## Run

```bash
%%bash
set -euo pipefail

export HF_TOKEN='PASTE_TOKEN_HERE'
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_VERBOSITY=info

cd /workspace
rm -rf ml_experiments

git clone -b boobie/leaderboard-final-sp8192 https://github.com/Rtx09x/ml_experiments.git
chmod +x ml_experiments/parameter_golf/leaderboard_final_sp8192/runpod_8xh100.sh

TRAIN_SHARDS=128 \
NPROC_PER_NODE=8 \
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
./ml_experiments/parameter_golf/leaderboard_final_sp8192/runpod_8xh100.sh
```

Expected important lines:

```text
4550/20000 val_bpb: ~1.0886
Total submission size quantized+brotli: ~15,991,930 bytes
quantized_sliding_window val_bpb: ~1.0829
quantized_ttt val_bpb: ~1.0808
```

## Tiny Sweep

After seed 42 reproduces, run only tiny sweeps. The known record is already heavily tuned, so broad changes are lower-value with two days left.

Good candidates:

```bash
SEED=314 QK_GAIN_INIT=5.25 TTT_LR=0.005
SEED=999 QK_GAIN_INIT=5.25 TTT_LR=0.005
SEED=42  QK_GAIN_INIT=5.35 TTT_LR=0.005
SEED=42  QK_GAIN_INIT=5.25 TTT_LR=0.006
```

Do not spend more compute on ChunkDiff for leaderboard unless this lane is already done.
