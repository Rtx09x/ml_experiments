# TRIADS Dielectric Kaggle Runner

Run `kaggle_dielectric_triads_single_cell.py` as one Kaggle notebook cell or as a
script. It targets `matbench_dielectric` with a composition-only TRIADS model.

Key properties:

- Uses the official `matbench` package and official folds.
- Uses structures only to extract composition.
- Selects hyperparameters per official fold using only an inner validation split.
- Keeps default configs under `150,000` trainable parameters.
- Trains a 3-seed ensemble for each candidate config.
- Saves checkpoints, histories, predictions, official Matbench record, summary JSON,
  and a final `triads_dielectric_outputs.zip`.

On Kaggle, paste the full script into one code cell and run it. Internet must be
enabled for installing `matbench`/`matminer` if they are not already present and
for downloading Mat2Vec embeddings.
