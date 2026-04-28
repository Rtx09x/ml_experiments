from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import ijson
import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


MPTRJ_ARTICLE_ID = 23713842
MATBENCH_ARTICLE_ID = 22715158

CUTOFF = 6.0
MAX_NEIGHBORS = 16
MAX_ATOMS = 200
N_RBF = 64
N_ELEM_FEAT = 21
N_DIST_PROF = 12
N_ATOM_FEAT = N_ELEM_FEAT + N_DIST_PROF + 3 + 1
N_EDGE_STATIC = 12
N_EDGE_FEAT = N_RBF + 3 + 1 + N_EDGE_STATIC
N_GLOBAL = 20
N_COMP = 160

MAX_FORCE_EV_A = 20.0
MIN_ENERGY_PA = -15.0
MAX_ENERGY_PA = 5.0


@dataclass(frozen=True)
class SourceFile:
    article_id: int
    file_id: int
    name: str
    size: int
    md5: str
    url: str
    license: str
    citation: str


def fetch_figshare_files(article_id: int) -> tuple[dict[str, Any], list[SourceFile]]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    meta = requests.get(url, timeout=60).json()
    files = []
    for f in meta["files"]:
        files.append(
            SourceFile(
                article_id=article_id,
                file_id=int(f["id"]),
                name=f["name"],
                size=int(f["size"]),
                md5=f.get("computed_md5") or f.get("supplied_md5") or "",
                url=f["download_url"],
                license=meta.get("license", {}).get("name", ""),
                citation=meta.get("citation", ""),
            )
        )
    return meta, files


def _pick_file(files: list[SourceFile], contains: tuple[str, ...]) -> SourceFile:
    matches = [f for f in files if all(part in f.name for part in contains)]
    if not matches:
        raise RuntimeError(f"No Figshare file matched: {contains}")
    return sorted(matches, key=lambda f: f.size, reverse=True)[0]


def _md5(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(src: SourceFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = src.name.replace("/", "__")
    out = dest_dir / safe_name
    if out.exists() and out.stat().st_size == src.size:
        if src.md5 and _md5(out) != src.md5:
            out.unlink()
        else:
            return out

    tmp = out.with_suffix(out.suffix + ".part")
    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
    mode = "ab" if "Range" in headers else "wb"
    with requests.get(src.url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = src.size
        initial = tmp.stat().st_size if tmp.exists() else 0
        with tmp.open(mode) as f, tqdm(
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            desc=safe_name,
            mininterval=2.0,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(out)
    if src.md5 and _md5(out) != src.md5:
        raise RuntimeError(f"MD5 mismatch for {out}")
    return out


def download_sources(root: Path, include_wbm: bool = True) -> dict[str, Any]:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"sources": []}

    mptrj_meta, mptrj_files = fetch_figshare_files(MPTRJ_ARTICLE_ID)
    mptrj_src = _pick_file(mptrj_files, ("MPtrj", ".json"))
    mptrj_path = download_file(mptrj_src, raw)
    manifest["mptrj_json"] = str(mptrj_path)
    manifest["sources"].append(mptrj_src.__dict__)
    manifest["mptrj_article"] = {
        "doi": mptrj_meta.get("doi"),
        "license": mptrj_meta.get("license", {}),
        "citation": mptrj_meta.get("citation"),
    }

    if include_wbm:
        mb_meta, mb_files = fetch_figshare_files(MATBENCH_ARTICLE_ID)
        wbm_src = _pick_file(mb_files, ("wbm-initial-atoms.extxyz.zip",))
        summary_src = _pick_file(mb_files, ("wbm-summary.csv.gz",))
        ref_src = _pick_file(mb_files, ("mp-elemental-reference-entries",))
        manifest["wbm_initial_zip"] = str(download_file(wbm_src, raw))
        manifest["wbm_summary_csv_gz"] = str(download_file(summary_src, raw))
        manifest["mp_elemental_refs_json_gz"] = str(download_file(ref_src, raw))
        manifest["sources"].extend([wbm_src.__dict__, summary_src.__dict__, ref_src.__dict__])
        manifest["matbench_article"] = {
            "doi": mb_meta.get("doi"),
            "license": mb_meta.get("license", {}),
            "citation": mb_meta.get("citation"),
        }

    (root / "manifest_downloads.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_element_table() -> np.ndarray:
    from pymatgen.core.periodic_table import Element

    bulk = {3: 11, 4: 130, 5: 320, 6: 33, 11: 6.3, 12: 45, 13: 76, 14: 100, 19: 3.1, 20: 17}
    cohesive = {1: 2.26, 3: 1.63, 4: 3.32, 5: 5.81, 6: 7.37, 7: 4.92, 8: 2.60, 11: 1.11}
    block_map = {"s": [1, 0, 0, 0], "p": [0, 1, 0, 0], "d": [0, 0, 1, 0], "f": [0, 0, 0, 1]}
    table = np.zeros((103, N_ELEM_FEAT), dtype=np.float32)
    for z in range(1, 103):
        try:
            el = Element.from_Z(z)
            mass = float(el.atomic_mass or 1.0)
            chi = float(el.X) if el.X is not None and not math.isnan(float(el.X)) else 0.0
            group = int(el.group or 0)
            row = int(el.row or 0)
            valence = group if group <= 2 else (group - 10 if group >= 13 else 2)
            radius = float(el.atomic_radius or 1.5)
            ionic = float(getattr(el, "average_ionic_radius", None) or radius)
            vdw = float(getattr(el, "van_der_waals_radius", None) or 2.0)
            melt = float(getattr(el, "melting_point", None) or 0.0)
            ox = float(el.common_oxidation_states[0]) if el.common_oxidation_states else 0.0
            block = block_map.get(getattr(el, "block", ""), [0, 0, 0, 0])
            row = np.array([
                mass,
                1.0 / math.sqrt(max(mass, 0.01)),
                chi,
                float(el.ionization_energies[0]) if el.ionization_energies else 0.0,
                float(el.electron_affinity or 0.0),
                float(valence),
                float(max(0, 10 - (group - 2)) if getattr(el, "block", "") == "d" and 3 <= group <= 12 else 0),
                float(row),
                float(group),
                radius,
                ionic,
                vdw,
                melt,
                float(bulk.get(z, 0.0)),
                float(cohesive.get(z, 0.0)),
                ox,
                *block,
                1.0 if el.is_metal else 0.0,
            ], dtype=np.float32)
            table[z] = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            table[z, 0] = 1.0
    return table


def gaussian_rbf(distances: np.ndarray, n_bins: int = N_RBF, cutoff: float = CUTOFF) -> np.ndarray:
    centers = np.linspace(0.0, cutoff, n_bins, dtype=np.float32)
    return np.exp(-10.0 * (distances.reshape(-1, 1) - centers.reshape(1, -1)) ** 2).astype(np.float32)


def smooth_cutoff(distances: np.ndarray, cutoff: float = CUTOFF) -> np.ndarray:
    x = np.clip(distances / cutoff, 0.0, 1.0)
    return (0.5 * (np.cos(np.pi * x) + 1.0)).astype(np.float32)


def comp_features(species_z: np.ndarray, elem_table: np.ndarray) -> np.ndarray:
    z = np.clip(species_z.astype(np.int64), 1, 102)
    counts = np.bincount(z, minlength=103).astype(np.float32)
    frac = counts / max(float(counts.sum()), 1.0)
    present = np.where(counts > 0)[0]
    feat_parts = [frac[1:103]]
    stats = []
    props = elem_table[present] if len(present) else np.zeros((1, N_ELEM_FEAT), np.float32)
    weights = frac[present] if len(present) else np.ones(1, np.float32)
    for j in range(N_ELEM_FEAT):
        vals = props[:, j]
        mean = float(np.sum(vals * weights))
        stats.extend([mean, float(vals.min()), float(vals.max()), float(np.sqrt(np.sum(weights * (vals - mean) ** 2)))])
    feat_parts.append(np.array(stats, np.float32))
    z_stats = np.array([len(present), z.mean(), z.std(), z.min(), z.max(), len(species_z)], dtype=np.float32)
    feat_parts.append(z_stats)
    out = np.concatenate(feat_parts)
    if len(out) < N_COMP:
        out = np.pad(out, (0, N_COMP - len(out)))
    return out[:N_COMP].astype(np.float32)


def global_features(structure: Any, species_z: np.ndarray, energy_pa: float, stress: np.ndarray | None) -> np.ndarray:
    lat = structure.lattice
    out = np.zeros(N_GLOBAL, dtype=np.float32)
    out[:6] = [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma]
    out[6] = float(lat.volume)
    out[7] = float(lat.volume) / max(len(species_z), 1)
    out[8] = float(len(species_z))
    out[9] = float(energy_pa)
    out[10] = float(len(set(species_z.tolist())))
    if stress is not None:
        s = np.asarray(stress, dtype=np.float32).reshape(-1)
        out[11:17] = s[:6] if len(s) >= 6 else np.pad(s, (0, 6 - len(s)))
    return out


def _frame_to_task(mp_id: str, frame_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from pymatgen.core import Structure

        structure = Structure.from_dict(record["structure"])
        n = len(structure)
        if n == 0 or n > MAX_ATOMS:
            return None
        forces = np.asarray(record.get("force") or record.get("forces"), dtype=np.float32)
        if forces.shape != (n, 3) or float(np.abs(forces).max()) > MAX_FORCE_EV_A:
            return None
        energy_pa = float(record["energy_per_atom"])
        if not (MIN_ENERGY_PA <= energy_pa <= MAX_ENERGY_PA):
            return None
        magmom = record.get("magmom")
        magmom_arr = np.asarray(magmom, dtype=np.float32) if magmom is not None else np.zeros(n, np.float32)
        if magmom_arr.shape[0] != n:
            magmom_arr = np.zeros(n, np.float32)
        stress = record.get("stress")
        return {
            "material_id": mp_id,
            "frame_id": frame_id,
            "structure": structure.as_dict(),
            "energy_pa": energy_pa,
            "forces": forces,
            "magmom": magmom_arr,
            "stress": np.asarray(stress, dtype=np.float32) if stress is not None else np.zeros((3, 3), np.float32),
        }
    except Exception:
        return None


def _select(items: list[tuple[str, dict[str, Any]]], n_snapshots: int) -> list[tuple[str, dict[str, Any]]]:
    if len(items) <= n_snapshots:
        return items
    idx = np.linspace(0, len(items) - 1, n_snapshots).round().astype(int)
    return [items[int(i)] for i in np.unique(idx)]


def iter_mptrj_task_chunks(
    json_path: Path,
    n_snapshots: int,
    max_materials: int | None,
    chunk_size: int,
):
    tasks: list[dict[str, Any]] = []
    with json_path.open("rb") as f:
        materials = ijson.kvitems(f, "", use_float=True)
        for n_mat, (mp_id, frames) in enumerate(tqdm(materials, desc="MPtrj materials", mininterval=5.0)):
            if max_materials and n_mat >= max_materials:
                break
            selected = _select(list(frames.items()), n_snapshots)
            for frame_id, record in selected:
                task = _frame_to_task(mp_id, frame_id, record)
                if task is not None:
                    tasks.append(task)
                    if len(tasks) >= chunk_size:
                        yield tasks
                        tasks = []
    if tasks:
        yield tasks


def _build_graph(task: dict[str, Any], elem_table: np.ndarray) -> dict[str, Any] | None:
    try:
        from pymatgen.core import Structure

        structure = Structure.from_dict(task["structure"])
        species_z = np.array([site.specie.Z for site in structure], dtype=np.int16)
        n = len(species_z)
        srcs, dsts, dists, vecs, shifts = [], [], [], [], []
        nn_per_atom: dict[int, list[float]] = defaultdict(list)
        for i, nbrs in enumerate(structure.get_all_neighbors(CUTOFF)):
            for nb in sorted(nbrs, key=lambda x: x.nn_distance)[:MAX_NEIGHBORS]:
                if nb.nn_distance < 0.01:
                    continue
                srcs.append(i)
                dsts.append(nb.index)
                dists.append(float(nb.nn_distance))
                vecs.append(np.asarray(nb.coords - structure[i].coords, dtype=np.float32))
                shifts.append(np.asarray(nb.image, dtype=np.int8))
                nn_per_atom[i].append(float(nb.nn_distance))
        if not srcs:
            return None
        src = np.asarray(srcs, np.int32)
        dst = np.asarray(dsts, np.int32)
        dist = np.asarray(dists, np.float32)
        vec = np.asarray(vecs, np.float32)
        shift = np.asarray(shifts, np.int8)

        elem = elem_table[np.clip(species_z, 0, 102)]
        coord = np.zeros(n, np.float32)
        avg = np.zeros(n, np.float32)
        std = np.zeros(n, np.float32)
        prof = np.zeros((n, N_DIST_PROF), np.float32)
        for i in range(n):
            ds = sorted(nn_per_atom.get(i, []))
            coord[i] = len(ds)
            if ds:
                avg[i] = float(np.mean(ds))
                std[i] = float(np.std(ds)) if len(ds) > 1 else 0.0
                prof[i, : min(len(ds), N_DIST_PROF)] = ds[:N_DIST_PROF]
        magmom = np.asarray(task["magmom"], np.float32).reshape(-1, 1)
        atom_feat = np.concatenate([elem, prof, coord[:, None], avg[:, None], std[:, None], magmom], axis=1)

        rbf = gaussian_rbf(dist)
        norm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
        dirs = vec / norm
        cutoff_val = smooth_cutoff(dist).reshape(-1, 1)
        zs, zd = np.clip(species_z[src], 0, 102), np.clip(species_z[dst], 0, 102)
        chi_s, chi_d = elem_table[zs, 2], elem_table[zd, 2]
        rad_s, rad_d = elem_table[zs, 9], elem_table[zd, 9]
        mass_s, mass_d = elem_table[zs, 0], elem_table[zd, 0]
        ion_s, ion_d = elem_table[zs, 3], elem_table[zd, 3]
        val_s, val_d = elem_table[zs, 5], elem_table[zd, 5]
        d_safe = np.maximum(dist, 0.01)
        edge_static = np.stack(
            [
                np.abs(chi_s - chi_d),
                1.0 - np.exp(-0.25 * (chi_s - chi_d) ** 2),
                np.minimum(rad_s, rad_d) / np.maximum(np.maximum(rad_s, rad_d), 1e-8),
                np.minimum(mass_s, mass_d) / np.maximum(np.maximum(mass_s, mass_d), 1e-8),
                val_s * val_d,
                np.abs(ion_s - ion_d),
                chi_s * chi_d,
                np.sqrt(np.maximum(chi_s * chi_d, 0.01)) / (d_safe * d_safe),
                1.0 / (d_safe * d_safe),
                1.0 / (d_safe**6),
                (rad_s + rad_d - d_safe) / d_safe,
                (zs == zd).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        edge_feat = np.concatenate([rbf, dirs.astype(np.float32), cutoff_val, edge_static], axis=1)
        lattice = np.asarray(structure.lattice.matrix, np.float32)
        frac = np.asarray(structure.frac_coords, np.float32)
        return {
            "material_id": task["material_id"],
            "formula": structure.composition.reduced_formula,
            "n_atoms": n,
            "n_edges": len(src),
            "atom_z": species_z,
            "atom_features": atom_feat.astype(np.float32),
            "frac_coords": frac,
            "forces": np.asarray(task["forces"], np.float32),
            "edge_src": src,
            "edge_dst": dst,
            "edge_shift": shift,
            "edge_features": edge_feat.astype(np.float32),
            "energy_pa": float(task["energy_pa"]),
            "stress": np.asarray(task["stress"], np.float32).reshape(3, 3),
            "lattice": lattice,
            "comp_features": comp_features(species_z, elem_table),
            "global_features": global_features(structure, species_z, float(task["energy_pa"]), task["stress"]),
        }
    except Exception:
        return None


class FlatH5Writer:
    def __init__(self, out_path: Path, elem_table: np.ndarray, source_manifest: dict[str, Any]):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.hf = h5py.File(out_path, "w")
        self.n_samples = 0
        self.n_atoms_total = 0
        self.n_edges_total = 0
        hf = self.hf
        hf.attrs.update(
            {
                "version": "crystalfold_v5_runpod_flat_1",
                "n_samples": 0,
                "n_atoms_total": 0,
                "n_edges_total": 0,
                "atom_feat_dim": N_ATOM_FEAT,
                "edge_feat_dim": N_EDGE_FEAT,
                "comp_feat_dim": N_COMP,
                "global_feat_dim": N_GLOBAL,
                "cutoff": CUTOFF,
                "max_neighbors": MAX_NEIGHBORS,
                "target": "mp2020_corrected_total_energy_per_atom",
            }
        )
        hf.attrs["source_manifest_json"] = json.dumps(source_manifest)
        hf.create_dataset("atom_offsets", data=np.array([0], np.int64), maxshape=(None,), chunks=(8192,), compression="gzip", compression_opts=4)
        hf.create_dataset("edge_offsets", data=np.array([0], np.int64), maxshape=(None,), chunks=(8192,), compression="gzip", compression_opts=4)
        hf.create_dataset("n_atoms", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(8192,), compression="gzip", compression_opts=4)
        hf.create_dataset("n_edges", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(8192,), compression="gzip", compression_opts=4)
        hf.create_dataset("energy_per_atom", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(8192,), compression="gzip", compression_opts=4)
        hf.create_dataset("lattice", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype=np.float32, chunks=(512, 3, 3), compression="lzf")
        hf.create_dataset("stress", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype=np.float16, chunks=(512, 3, 3), compression="lzf")
        hf.create_dataset("comp_features", shape=(0, N_COMP), maxshape=(None, N_COMP), dtype=np.float16, chunks=(512, N_COMP), compression="lzf")
        hf.create_dataset("global_features", shape=(0, N_GLOBAL), maxshape=(None, N_GLOBAL), dtype=np.float16, chunks=(2048, N_GLOBAL), compression="lzf")
        hf.create_dataset("element_table", data=elem_table.astype(np.float32), compression="gzip", compression_opts=4)
        str_dt = h5py.string_dtype("utf-8")
        hf.create_dataset("material_ids", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=(8192,))
        hf.create_dataset("formulas", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=(8192,))
        atoms = hf.create_group("atoms")
        atoms.create_dataset("atom_z", shape=(0,), maxshape=(None,), dtype=np.int16, chunks=(100000,), compression="lzf")
        atoms.create_dataset("atom_features", shape=(0, N_ATOM_FEAT), maxshape=(None, N_ATOM_FEAT), dtype=np.float16, chunks=(10000, N_ATOM_FEAT), compression="lzf")
        atoms.create_dataset("frac_coords", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(10000, 3), compression="lzf")
        atoms.create_dataset("forces", shape=(0, 3), maxshape=(None, 3), dtype=np.float16, chunks=(10000, 3), compression="lzf")
        edges = hf.create_group("edges")
        edges.create_dataset("src", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(200000,), compression="lzf")
        edges.create_dataset("dst", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(200000,), compression="lzf")
        edges.create_dataset("shift", shape=(0, 3), maxshape=(None, 3), dtype=np.int8, chunks=(200000, 3), compression="lzf")
        edges.create_dataset("edge_features", shape=(0, N_EDGE_FEAT), maxshape=(None, N_EDGE_FEAT), dtype=np.float16, chunks=(10000, N_EDGE_FEAT), compression="lzf")

    @staticmethod
    def _append(ds: h5py.Dataset, values: np.ndarray) -> None:
        old = ds.shape[0]
        new = old + values.shape[0]
        ds.resize((new, *ds.shape[1:]))
        ds[old:new] = values

    def append(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        hf = self.hf
        n_atoms = np.asarray([r["n_atoms"] for r in results], dtype=np.int32)
        n_edges = np.asarray([r["n_edges"] for r in results], dtype=np.int32)
        n = len(results)

        for name, values in {
            "n_atoms": n_atoms,
            "n_edges": n_edges,
            "energy_per_atom": np.asarray([r["energy_pa"] for r in results], np.float32),
            "lattice": np.stack([r["lattice"] for r in results]).astype(np.float32),
            "stress": np.stack([r["stress"] for r in results]).astype(np.float16),
            "comp_features": np.stack([r["comp_features"] for r in results]).astype(np.float16),
            "global_features": np.stack([r["global_features"] for r in results]).astype(np.float16),
        }.items():
            self._append(hf[name], values)

        self._append(hf["material_ids"], np.asarray([r["material_id"] for r in results], dtype=object))
        self._append(hf["formulas"], np.asarray([r["formula"] for r in results], dtype=object))
        self._append(hf["atoms/atom_z"], np.concatenate([r["atom_z"] for r in results]).astype(np.int16))
        self._append(hf["atoms/atom_features"], np.concatenate([r["atom_features"] for r in results]).astype(np.float16))
        self._append(hf["atoms/frac_coords"], np.concatenate([r["frac_coords"] for r in results]).astype(np.float32))
        self._append(hf["atoms/forces"], np.concatenate([r["forces"] for r in results]).astype(np.float16))
        self._append(hf["edges/src"], np.concatenate([r["edge_src"] for r in results]).astype(np.int32))
        self._append(hf["edges/dst"], np.concatenate([r["edge_dst"] for r in results]).astype(np.int32))
        self._append(hf["edges/shift"], np.concatenate([r["edge_shift"] for r in results]).astype(np.int8))
        self._append(hf["edges/edge_features"], np.concatenate([r["edge_features"] for r in results]).astype(np.float16))

        atom_new = self.n_atoms_total + np.cumsum(n_atoms).astype(np.int64)
        edge_new = self.n_edges_total + np.cumsum(n_edges).astype(np.int64)
        self._append(hf["atom_offsets"], atom_new)
        self._append(hf["edge_offsets"], edge_new)
        self.n_samples += n
        self.n_atoms_total += int(n_atoms.sum())
        self.n_edges_total += int(n_edges.sum())
        hf.attrs["n_samples"] = self.n_samples
        hf.attrs["n_atoms_total"] = self.n_atoms_total
        hf.attrs["n_edges_total"] = self.n_edges_total
        hf.flush()

    def close(self) -> None:
        self.hf.close()


def write_graph_h5(results: list[dict[str, Any]], out_path: Path, elem_table: np.ndarray, source_manifest: dict[str, Any]) -> None:
    writer = FlatH5Writer(out_path, elem_table, source_manifest)
    try:
        writer.append(results)
    finally:
        writer.close()


def build_mptrj_h5(root: Path, workers: int, snapshots: int, max_materials: int | None, chunk_size: int) -> Path:
    manifest_path = root / "manifest_downloads.json"
    manifest = json.loads(manifest_path.read_text())
    raw_json = Path(manifest["mptrj_json"])
    elem_table = build_element_table()
    out = root / "processed" / "mpt_v5.h5"
    writer = FlatH5Writer(out, elem_table, manifest)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk in iter_mptrj_task_chunks(raw_json, snapshots, max_materials, chunk_size):
                results = []
                futures = [pool.submit(_build_graph, task, elem_table) for task in chunk]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Build MPtrj graphs", mininterval=5.0):
                    item = fut.result()
                    if item is not None:
                        results.append(item)
                writer.append(results)
                print(f"wrote samples={writer.n_samples:,} atoms={writer.n_atoms_total:,} edges={writer.n_edges_total:,}")
    finally:
        writer.close()
    if not out.exists():
        raise RuntimeError("No MPtrj graphs were built.")
    write_scaler(out, root / "processed" / "scaler.json")
    return out


def _extract_extxyz_files(zip_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".extxyz")]
        if not members:
            raise RuntimeError(f"No extxyz file in {zip_path}")
        out_paths = []
        for member in tqdm(members, desc="Extract WBM extxyz", mininterval=5.0):
            out = dest_dir / Path(member).name
            out_paths.append(out)
            if out.exists() and out.stat().st_size > 0:
                continue
            zf.extract(member, dest_dir)
            extracted = dest_dir / member
            if extracted != out:
                out.parent.mkdir(parents=True, exist_ok=True)
                extracted.replace(out)
        return out_paths


def build_wbm_h5(root: Path, workers: int, max_wbm: int | None) -> Path:
    from ase.io import iread
    from pymatgen.io.ase import AseAtomsAdaptor

    manifest = json.loads((root / "manifest_downloads.json").read_text())
    elem_table = build_element_table()
    extxyz_files = _extract_extxyz_files(Path(manifest["wbm_initial_zip"]), root / "raw" / "wbm_initial")
    tasks = []
    for path in tqdm(extxyz_files, desc="WBM files", mininterval=5.0):
        if max_wbm and len(tasks) >= max_wbm:
            break
        try:
            for atoms in iread(str(path), format="extxyz"):
                structure = AseAtomsAdaptor.get_structure(atoms)
                mid = str(atoms.info.get("material_id") or atoms.info.get("mat_id") or atoms.info.get("id") or path.stem)
                n = len(structure)
                if 0 < n <= MAX_ATOMS:
                    tasks.append(
                        {
                            "material_id": mid,
                            "structure": structure.as_dict(),
                            "energy_pa": 0.0,
                            "forces": np.zeros((n, 3), np.float32),
                            "magmom": np.zeros(n, np.float32),
                            "stress": np.zeros((3, 3), np.float32),
                        }
                    )
                if max_wbm and len(tasks) >= max_wbm:
                    break
        except Exception:
            continue
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_graph, task, elem_table) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Build WBM graphs", mininterval=5.0):
            item = fut.result()
            if item is not None:
                results.append(item)
    out = root / "processed" / "wbm_initial_v5.h5"
    write_graph_h5(results, out, elem_table, manifest)
    return out


def write_scaler(h5_path: Path, out_path: Path) -> None:
    with h5py.File(h5_path, "r") as hf:
        y = hf["energy_per_atom"][:].astype(np.float64)
    q25, q75 = np.percentile(y, [25, 75])
    scaler = {"energy_median": float(np.median(y)), "energy_iqr": float(max(q75 - q25, 0.1))}
    out_path.write_text(json.dumps(scaler, indent=2))


class FlatGraphDataset(Dataset):
    def __init__(self, h5_path: str | Path, split: str = "train", train_frac: float = 0.9, seed: int = 42):
        self.h5_path = str(h5_path)
        self.split = split
        with h5py.File(self.h5_path, "r") as hf:
            self.n_total = int(hf.attrs["n_samples"])
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in hf["material_ids"][:]]
        unique = np.array(sorted(set(ids)), dtype=object)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        train_ids = set(unique[: int(len(unique) * train_frac)])
        self.indices = [i for i, mid in enumerate(ids) if (mid in train_ids) == (split == "train")]
        self._hf: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self) -> h5py.File:
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")
        return self._hf

    def __getitem__(self, local_idx: int) -> dict[str, Any]:
        hf = self._file()
        i = self.indices[local_idx]
        a0, a1 = int(hf["atom_offsets"][i]), int(hf["atom_offsets"][i + 1])
        e0, e1 = int(hf["edge_offsets"][i]), int(hf["edge_offsets"][i + 1])
        src = hf["edges/src"][e0:e1].astype(np.int64)
        dst = hf["edges/dst"][e0:e1].astype(np.int64)
        return {
            "atom_features": hf["atoms/atom_features"][a0:a1].astype(np.float32),
            "frac_coords": hf["atoms/frac_coords"][a0:a1].astype(np.float32),
            "forces": hf["atoms/forces"][a0:a1].astype(np.float32),
            "edge_index": np.stack([src, dst], axis=0),
            "edge_shift": hf["edges/shift"][e0:e1].astype(np.float32),
            "edge_features": hf["edges/edge_features"][e0:e1].astype(np.float32),
            "edge_static": hf["edges/edge_features"][e0:e1, -N_EDGE_STATIC:].astype(np.float32),
            "lattice": hf["lattice"][i].astype(np.float32),
            "comp_features": hf["comp_features"][i].astype(np.float32),
            "global_features": hf["global_features"][i].astype(np.float32),
            "energy_per_atom": np.float32(hf["energy_per_atom"][i]),
            "n_atoms": np.int32(a1 - a0),
        }


def collate_graphs(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    sizes = [int(s["n_atoms"]) for s in samples]
    edge_indices = []
    atom_offset = 0
    atom_graph_index = []
    edge_graph_index = []
    for s in samples:
        ei = torch.as_tensor(s["edge_index"], dtype=torch.long) + atom_offset
        edge_indices.append(ei)
        atom_graph_index.append(torch.full((int(s["n_atoms"]),), len(atom_graph_index), dtype=torch.long))
        edge_graph_index.append(torch.full((ei.shape[1],), len(edge_graph_index), dtype=torch.long))
        atom_offset += int(s["n_atoms"])
    return {
        "atom_features": torch.cat([torch.as_tensor(s["atom_features"], dtype=torch.float32) for s in samples], dim=0),
        "frac_coords": torch.cat([torch.as_tensor(s["frac_coords"], dtype=torch.float32) for s in samples], dim=0),
        "forces": torch.cat([torch.as_tensor(s["forces"], dtype=torch.float32) for s in samples], dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_shift": torch.cat([torch.as_tensor(s["edge_shift"], dtype=torch.float32) for s in samples], dim=0),
        "edge_features": torch.cat([torch.as_tensor(s["edge_features"], dtype=torch.float32) for s in samples], dim=0),
        "edge_static": torch.cat([torch.as_tensor(s["edge_static"], dtype=torch.float32) for s in samples], dim=0),
        "lattice": torch.stack([torch.as_tensor(s["lattice"], dtype=torch.float32) for s in samples]),
        "atom_graph_index": torch.cat(atom_graph_index, dim=0),
        "edge_graph_index": torch.cat(edge_graph_index, dim=0),
        "comp_features": torch.stack([torch.as_tensor(s["comp_features"], dtype=torch.float32) for s in samples]),
        "global_features": torch.stack([torch.as_tensor(s["global_features"], dtype=torch.float32) for s in samples]),
        "energy_per_atom": torch.as_tensor([s["energy_per_atom"] for s in samples], dtype=torch.float32),
        "crystal_sizes": torch.as_tensor(sizes, dtype=torch.long),
        "crystal_sizes_list": sizes,
    }


def run_preprocess(args: argparse.Namespace) -> None:
    root = Path(args.root)
    manifest = download_sources(root, include_wbm=args.include_wbm)
    mpt_h5 = build_mptrj_h5(root, args.workers, args.snapshots, args.max_materials, args.chunk_size)
    outputs = {"mpt_h5": str(mpt_h5)}
    if args.include_wbm:
        outputs["wbm_h5"] = str(build_wbm_h5(root, max(1, args.workers // 2), args.max_wbm))
    (root / "processed" / "manifest.json").write_text(json.dumps({"downloads": manifest, "outputs": outputs}, indent=2))
