# MI300X Quickstart

If you want the least annoying path, use [`runpod_mi300x_hf.sh`](./runpod_mi300x_hf.sh).

## What to edit once

Open the script and replace:

```bash
HF_TOKEN="${HF_TOKEN:-PASTE_YOUR_HF_TOKEN_HERE}"
```

with your actual HF token, for example:

```bash
HF_TOKEN="${HF_TOKEN:-hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx}"
```

Yes, hardcoding a token is less secure. If you do this, keep the pod private and
rotate the token afterward.

## Smoke run

```bash
cd /workspace/ml_experiments/crystalfold/matbench_discovery_v5_runpod
bash runpod_mi300x_hf.sh smoke
```

## Full run

```bash
cd /workspace/ml_experiments/crystalfold/matbench_discovery_v5_runpod
bash runpod_mi300x_hf.sh full
```

## Expected outputs

Smoke:

```text
/workspace/crystalfold_v5_artifacts/runs/mi300x_smoke/
```

Full:

```text
/workspace/crystalfold_v5_artifacts/runs/mi300x_v5/
```

Important files:

```text
best.pt
last.pt
history.json
run_config.json
scaler.json
```

