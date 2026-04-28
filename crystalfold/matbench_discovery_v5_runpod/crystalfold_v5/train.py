from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FlatGraphDataset, collate_graphs
from .model import ConfigV5, CrystalFoldV5, DiscoveryLoss


def build_model_config(preset: str) -> ConfigV5:
    if preset == "base":
        return ConfigV5()
    if preset == "large":
        return ConfigV5(
            d_model=320,
            d_edge=160,
            d_comp=160,
            n_heads=8,
            d_ff=640,
            n_gat_layers=3,
            max_cycles=8,
        )
    if preset == "xlarge":
        return ConfigV5(
            d_model=384,
            d_edge=192,
            d_comp=192,
            n_heads=8,
            d_ff=768,
            n_gat_layers=4,
            max_cycles=10,
        )
    raise ValueError(f"unknown model preset: {preset}")


class EnergyScaler:
    def __init__(self, median: float, iqr: float):
        self.median = float(median)
        self.iqr = float(max(iqr, 0.1))

    @classmethod
    def from_dataset(cls, dataset_path: Path) -> "EnergyScaler":
        scaler_path = dataset_path.parent / "scaler.json"
        if scaler_path.exists():
            data = json.loads(scaler_path.read_text())
            return cls(data["energy_median"], data["energy_iqr"])
        import h5py

        with h5py.File(dataset_path, "r") as hf:
            y = hf["energy_per_atom"][:].astype(np.float64)
        q25, q75 = np.percentile(y, [25, 75])
        return cls(float(np.median(y)), float(max(q75 - q25, 0.1)))

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.median) / self.iqr

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.iqr + self.median

    def to_json(self) -> dict[str, float]:
        return {"energy_median": self.median, "energy_iqr": self.iqr}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def materialize_positions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    lattice_per_atom = batch["lattice"].index_select(0, batch["atom_graph_index"])
    return torch.bmm(batch["frac_coords"].unsqueeze(1), lattice_per_atom).squeeze(1)


def materialize_edge_shift_cart(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    lattice_per_edge = batch["lattice"].index_select(0, batch["edge_graph_index"])
    return torch.bmm(batch["edge_shift"].unsqueeze(1), lattice_per_edge).squeeze(1)


def maybe_local_copy(dataset: Path, local_root: str | None) -> Path:
    if not local_root:
        return dataset
    local_dir = Path(local_root) / "crystalfold_v5_local"
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / dataset.name
    if not dst.exists() or dst.stat().st_size != dataset.stat().st_size:
        shutil.copy2(dataset, dst)
    scaler = dataset.parent / "scaler.json"
    if scaler.exists():
        shutil.copy2(scaler, dst.parent / "scaler.json")
    return dst


def evaluate(
    model: CrystalFoldV5,
    loader: DataLoader,
    scaler: EnergyScaler,
    device: torch.device,
    amp: str,
    max_batches: int | None = None,
) -> float:
    model.eval()
    maes = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="val", leave=False, mininterval=5.0)):
            if max_batches is not None and i >= max_batches:
                break
            batch = move_batch(batch, device)
            dtype = torch.bfloat16 if amp == "bf16" else torch.float16
            with autocast(device_type=device.type, dtype=dtype, enabled=amp != "off" and device.type == "cuda"):
                out = model(batch, positions=None)
            pred = scaler.inverse(out["final_pred"].float())
            maes.append((pred - batch["energy_per_atom"]).abs().detach().cpu())
    return float(torch.cat(maes).mean())


def probe_wbm(
    model: CrystalFoldV5,
    loader: DataLoader,
    scaler: EnergyScaler,
    device: torch.device,
    amp: str,
    out_path: Path,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    rows: list[tuple[str, float]] = []
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="wbm", leave=False, mininterval=5.0)):
            if max_batches is not None and i >= max_batches:
                break
            mids = batch["material_ids"]
            batch = move_batch(batch, device)
            with autocast(device_type=device.type, dtype=dtype, enabled=amp != "off" and device.type == "cuda"):
                out = model(batch, positions=None)
            pred = scaler.inverse(out["final_pred"].float()).detach().cpu().tolist()
            rows.extend(zip(mids, pred))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["material_id", "pred_energy_per_atom"])
        writer.writerows(rows)
    vals = np.array([x[1] for x in rows], dtype=np.float64)
    order = np.argsort(vals)[: min(10, len(vals))]
    top_ids = [rows[int(i)][0] for i in order]
    return {
        "n": float(len(rows)),
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "std": float(vals.std()) if len(vals) else 0.0,
        "min": float(vals.min()) if len(vals) else 0.0,
        "max": float(vals.max()) if len(vals) else 0.0,
        "top_ids": top_ids,
    }


def run_train(args: argparse.Namespace) -> None:
    args.force_every = max(1, int(args.force_every))
    args.energy_only_epochs = max(0, int(args.energy_only_epochs))
    dataset_path = maybe_local_copy(Path(args.dataset), args.local_copy)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_ds = FlatGraphDataset(dataset_path, split="train", train_frac=args.train_frac, seed=args.seed)
    val_ds = FlatGraphDataset(dataset_path, split="val", train_frac=args.train_frac, seed=args.seed)
    wbm_ds = FlatGraphDataset(args.wbm_dataset, split="all") if args.wbm_dataset else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_graphs,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        collate_fn=collate_graphs,
        pin_memory=True,
        persistent_workers=args.workers > 1,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    wbm_loader = None
    if wbm_ds is not None:
        wbm_loader = DataLoader(
            wbm_ds,
            batch_size=max(1, args.batch_size * 2),
            shuffle=False,
            num_workers=max(1, args.workers // 2),
            collate_fn=collate_graphs,
            pin_memory=True,
            persistent_workers=args.workers > 1,
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        )

    scaler = EnergyScaler.from_dataset(dataset_path)
    (run_dir / "scaler.json").write_text(json.dumps(scaler.to_json(), indent=2))
    cfg = build_model_config(args.model_preset)
    model = CrystalFoldV5(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    criterion = DiscoveryLoss().to(device)
    optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.12,
        div_factor=25,
        final_div_factor=1000,
    )
    grad_scaler = GradScaler(enabled=args.amp == "fp16" and device.type == "cuda")
    start_epoch = 0
    best_mae = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        target_model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_mae = float(ckpt["best_mae"])

    meta = {
        "dataset": str(dataset_path),
        "wbm_dataset": str(args.wbm_dataset) if args.wbm_dataset else None,
        "device": str(device),
        "parameters": (model._orig_mod if hasattr(model, "_orig_mod") else model).count_parameters(),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "wbm_samples": len(wbm_ds) if wbm_ds is not None else 0,
        "args": vars(args),
    }
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))

    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        losses, maes = [], []
        energy_losses, force_losses = [], []
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", mininterval=5.0)
        force_epoch = epoch >= args.energy_only_epochs
        for step_idx, batch in enumerate(pbar):
            batch = move_batch(batch, device)
            target_e = scaler.transform(batch["energy_per_atom"])
            optimizer.zero_grad(set_to_none=True)
            use_force = force_epoch and (step_idx % args.force_every == 0)
            with autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "off" and device.type == "cuda"):
                if use_force:
                    positions = materialize_positions(batch).detach().requires_grad_(True)
                    batch["edge_shift_cart"] = materialize_edge_shift_cart(batch)
                    out = model(batch, positions=positions)
                    force_total = CrystalFoldV5.total_energy_from_energy_per_atom(out["final_pred"], batch["crystal_sizes"])
                else:
                    out = model(batch, positions=None)
            if use_force:
                pred_f_norm = -torch.autograd.grad(force_total, positions, create_graph=True, retain_graph=True)[0]
                target_f_norm = batch["forces"] / scaler.iqr
                with autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "off" and device.type == "cuda"):
                    loss, info = criterion(out["final_pred"], target_e, pred_f_norm, target_f_norm, out["cycle_preds"])
            else:
                with autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "off" and device.type == "cuda"):
                    loss, info = criterion.energy_only(out["final_pred"], target_e, out["cycle_preds"])
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), args.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            with torch.no_grad():
                pred_phys = scaler.inverse(out["final_pred"].float())
                mae = float((pred_phys - batch["energy_per_atom"]).abs().mean().detach().cpu())
            losses.append(float(loss.detach().cpu()))
            maes.append(mae)
            energy_losses.append(float(info["energy_loss"]))
            force_losses.append(float(info["force_loss"]))
            pbar.set_postfix(
                loss=np.mean(losses[-20:]),
                e=np.mean(energy_losses[-20:]),
                f=np.mean(force_losses[-20:]),
                mae=np.mean(maes[-20:]),
                lr=scheduler.get_last_lr()[0],
                force=int(use_force),
            )

        val_mae = evaluate(model, val_loader, scaler, device, args.amp, args.val_max_batches)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_energy_loss": float(np.mean(energy_losses)),
            "train_force_loss": float(np.mean(force_losses)),
            "train_mae": float(np.mean(maes)),
            "val_mae": val_mae,
            "seconds": time.time() - t0,
            "force_step_frac": float(sum(1 for x in force_losses if x > 0.0) / max(len(force_losses), 1)),
        }
        if wbm_loader is not None and args.wbm_every > 0 and (epoch + 1) % args.wbm_every == 0:
            probe_path = run_dir / "wbm_probe" / f"epoch_{epoch + 1:03d}.csv"
            wbm_stats = probe_wbm(model, wbm_loader, scaler, device, args.amp, probe_path, args.wbm_max_batches)
            row["wbm_probe"] = wbm_stats
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ckpt = {
            "model": target_model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mae": min(best_mae, val_mae),
            "cfg": cfg.__dict__,
            "scaler": scaler.to_json(),
        }
        torch.save(ckpt, run_dir / "last.pt")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(ckpt, run_dir / "best.pt")
        print(
            f"epoch={epoch + 1} "
            f"train_loss={row['train_loss']:.5f} "
            f"energy_loss={row['train_energy_loss']:.5f} "
            f"force_loss={row['train_force_loss']:.5f} "
            f"train_mae={row['train_mae']:.5f} "
            f"val_mae={val_mae:.5f} "
            f"best_val={best_mae:.5f} "
            f"force_frac={row['force_step_frac']:.3f} "
            f"sec={row['seconds']:.1f}"
        )
        if "wbm_probe" in row:
            w = row["wbm_probe"]
            print(
                f"wbm_probe epoch={epoch + 1} "
                f"n={int(w['n'])} mean={w['mean']:.5f} std={w['std']:.5f} "
                f"min={w['min']:.5f} max={w['max']:.5f} "
                f"csv={run_dir / 'wbm_probe' / f'epoch_{epoch + 1:03d}.csv'}"
            )

