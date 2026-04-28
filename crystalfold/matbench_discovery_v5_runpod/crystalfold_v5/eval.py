from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FlatGraphDataset, collate_graphs
from .model import ConfigV5, CrystalFoldV5
from .relax import FIREConfig, FIRERelaxer
from .train import EnergyScaler, materialize_edge_shift_cart, materialize_positions, move_batch


def run_eval_wbm(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    cfg = ConfigV5(**{k: v for k, v in cfg_dict.items() if k in ConfigV5.__dataclass_fields__})
    model = CrystalFoldV5(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scaler = EnergyScaler.from_dataset(Path(args.train_dataset))
    ds = FlatGraphDataset(args.wbm_dataset, split="all")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_graphs,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    fire_cfg = FIREConfig(max_steps=args.relax_steps, fmax_threshold=args.fmax)
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    rows: list[tuple[int, float, int]] = []

    for i, batch in enumerate(tqdm(loader, desc="wbm_relax_eval", mininterval=5.0)):
        if args.max_batches is not None and i >= args.max_batches:
            break
        mids = batch["material_ids"]
        batch = move_batch(batch, device)
        positions = materialize_positions(batch)
        batch["edge_shift_cart"] = materialize_edge_shift_cart(batch)
        relaxer = FIRERelaxer(fire_cfg)
        relaxer.init(batch["crystal_sizes_list"], positions)
        for _ in range(fire_cfg.max_steps):
            if relaxer.all_converged:
                break
            pos = relaxer.positions.detach().requires_grad_(True)
            with autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "off" and device.type == "cuda"):
                out = model(batch, positions=pos)
                total_e = CrystalFoldV5.total_energy_from_energy_per_atom(out["final_pred"], batch["crystal_sizes"])
            pred_f_norm = -torch.autograd.grad(total_e, pos, create_graph=False, retain_graph=False)[0]
            forces_phys = pred_f_norm.detach() * scaler.iqr
            pred_epa_phys = scaler.inverse(out["final_pred"].float())
            relaxer.step(forces_phys, pred_epa_phys)
        with torch.no_grad():
            with autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "off" and device.type == "cuda"):
                out_final = model(batch, positions=relaxer.positions)
            pred = scaler.inverse(out_final["final_pred"].float()).detach().cpu().tolist()
        rows.extend(zip(mids, pred, relaxer.steps_used()))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["material_id", "pred_energy_per_atom", "relax_steps"])
        writer.writerows(rows)

    vals = np.array([x[1] for x in rows], dtype=np.float64)
    stats = {
        "n": len(rows),
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "std": float(vals.std()) if len(vals) else 0.0,
        "min": float(vals.min()) if len(vals) else 0.0,
        "max": float(vals.max()) if len(vals) else 0.0,
        "output": str(out_path),
    }
    print(json.dumps(stats, indent=2))
