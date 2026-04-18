"""
TRIADS-Dielectric: Kaggle single-cell runner for matbench_dielectric.

Goal
----
Composition-only TRIADS/DeepHybridTRM sweep for matbench_dielectric using the
official Matbench folds and record API. The code is intentionally self-contained
so it can be pasted into one Kaggle notebook cell or run as a Python script.

Legality / benchmark discipline
-------------------------------
- Official train/test folds come from the `matbench` package.
- Hyperparameter selection is done only with an inner validation split carved
  out of each official train+validation fold.
- The official test fold is only used after a config is selected by inner-val
  MAE for that fold.
- Feature scaling and target scaling are fit only on the inner training split.
- This is composition-only: structures are used only to extract composition.

Default sweep
-------------
The runner trains exactly four composition-only TRIADS configs.
"""

# =========================
# 0. Kaggle bootstrap
# =========================

INSTALL_MISSING = True
TASK_NAME = "matbench_dielectric"
OUTPUT_DIR = "triads_dielectric_outputs"
ZIP_NAME = f"{OUTPUT_DIR}.zip"

SEEDS = [42]
EPOCHS = 320
SWA_START = 220
BATCH_SIZE = 1024
INNER_VAL_FRAC = 0.15
TARGET_SCALER_KIND = "robust_iqr"
USE_TORCH_COMPILE = False  # usually not worth compile overhead for this tiny net


def _ensure_packages():
    import importlib.util
    import subprocess
    import sys

    required_modules = ["matbench", "matminer", "pymatgen", "gensim", "sklearn", "monty"]
    missing = [mod for mod in required_modules if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    if not INSTALL_MISSING:
        raise ImportError(f"Missing packages: {missing}")

    print("Installing missing packages:", missing)
    # matbench==0.6 pins Python-3.12-hostile historical versions
    # (scikit-learn==1.0.1, scipy==1.7.3). Install modern runtime deps first,
    # then install matbench without dependency backtracking.
    modern_deps = []
    if any(m in missing for m in ["matminer", "pymatgen", "monty", "gensim", "sklearn"]):
        modern_deps = ["matminer", "pymatgen", "monty", "gensim", "scikit-learn"]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + modern_deps
        )
    if "matbench" in missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "matbench==0.6"]
        )


_ensure_packages()

# Gensim versions used by old Mat2Vec files may expect scipy.linalg.triu.
try:
    import scipy.linalg
    import numpy as _np_for_scipy_patch
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = _np_for_scipy_patch.triu
except Exception:
    pass


# =========================
# 1. Imports and setup
# =========================

import copy
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "checkpoints").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "predictions").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "histories").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "features").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(OUTPUT_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger("TRIADS-DIEL")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_plain(obj):
    """Convert Matbench/monty/numpy objects into JSON-safe primitives."""
    if obj.__class__.__name__ == "RecursiveDotDict":
        return {str(k): to_plain(v) for k, v in dict(obj).items()}
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        return to_plain(as_dict())
    if isinstance(as_dict, dict):
        return to_plain(as_dict)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def task_metadata_summary(task):
    return {
        "dataset_name": getattr(task, "dataset_name", TASK_NAME),
        "version": getattr(task, "version", None),
        "input_type": getattr(task.metadata, "input_type", None),
        "target": getattr(task.metadata, "target", None),
        "task_type": getattr(task.metadata, "task_type", None),
        "unit": getattr(task.metadata, "unit", None),
        "n_samples": getattr(task.metadata, "n_samples", None),
        "mad": getattr(task.metadata, "mad", None),
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    log.info("GPU: %s | %.1f GB", torch.cuda.get_device_name(0),
             props.total_memory / 1e9)
else:
    log.info("GPU not available; running on CPU.")

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# =========================
# 2. Fast GPU dataloader
# =========================

class FastTensorDataLoader:
    """Minimal dataloader for tensors already on the target device."""

    def __init__(self, *tensors, batch_size=1024, shuffle=False):
        assert tensors, "Need at least one tensor"
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.dataset_len = int(tensors[0].shape[0])
        self.n_batches = (self.dataset_len + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = tuple(t[perm] for t in self.tensors)
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self.dataset_len:
            raise StopIteration
        j = min(self._i + self.batch_size, self.dataset_len)
        batch = tuple(t[self._i:j] for t in self.tensors)
        self._i = j
        return batch

    def __len__(self):
        return self.n_batches


# =========================
# 3. Composition featurizer
# =========================

class DielectricCompositionFeaturizer:
    """Composition-only features from the proven TRIADS stack.

    Layout:
      Magpie ElementProperty: 22 properties x 6 statistics = 132 features
      Extra fast composition features: ElementFraction, Stoichiometry,
        ValenceOrbital, TMetalFraction, and lightweight dielectric chemistry
        proxies. Slow oxidation-state/search featurizers are deliberately
        avoided so one awkward formula cannot hang the Kaggle run.
      Mat2Vec pooled element embedding: 200 features
    """

    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = [
        "pretrained_embeddings",
        "pretrained_embeddings.wv.vectors.npy",
        "pretrained_embeddings.trainables.syn1neg.npy",
    ]

    def __init__(self, cache_dir="mat2vec_cache"):
        from matminer.featurizers.composition import (
            ElementFraction,
            Stoichiometry,
            ValenceOrbital,
        )
        from matminer.featurizers.composition.element import TMetalFraction

        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())
        self.mg_n_props = len(self.ep_magpie.features)
        self.mg_stat_dim = len(self.ep_magpie.stats)
        assert self.mg_n_props * self.mg_stat_dim == self.n_mg

        self.extra_featurizers = [
            ("ElementFraction", ElementFraction()),
            ("Stoichiometry", Stoichiometry()),
            ("ValenceOrbital", ValenceOrbital()),
            ("TMetalFraction", TMetalFraction()),
        ]
        self.n_extra = None
        self.extra_labels = None
        self.scaler = None

        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        for fname in self.FILES:
            dest = cache / fname
            if not dest.exists():
                log.info("Downloading Mat2Vec file: %s", fname)
                urllib.request.urlretrieve(self.GCS + fname, dest)

        try:
            from gensim.models import Word2Vec
            self.m2v = Word2Vec.load(str(cache / "pretrained_embeddings"))
            self.emb = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}
            self.mat2vec_dim = 200
            self.mat2vec_ok = True
        except Exception as exc:
            log.warning("Mat2Vec load failed; using zero Mat2Vec vectors: %s", exc)
            self.emb = {}
            self.mat2vec_dim = 200
            self.mat2vec_ok = False

    @staticmethod
    def composition_from_input(x):
        if hasattr(x, "composition"):
            return x.composition
        if isinstance(x, Composition):
            return x
        return Composition(str(x))

    def _pool_mat2vec(self, comp):
        vec = np.zeros(self.mat2vec_dim, dtype=np.float32)
        total = 0.0
        for symbol, amount in comp.get_el_amt_dict().items():
            if symbol in self.emb:
                vec += float(amount) * np.asarray(self.emb[symbol], dtype=np.float32)
                total += float(amount)
        return vec / max(total, 1e-8)

    def _dielectric_proxies(self, comp):
        from pymatgen.core.periodic_table import Element

        amounts = comp.get_el_amt_dict()
        total = max(float(sum(amounts.values())), 1e-8)
        props = {
            "X": [],
            "atomic_mass": [],
            "atomic_radius": [],
            "average_ionic_radius": [],
            "row": [],
            "group": [],
            "Z": [],
            "is_metal": [],
            "is_transition_metal": [],
        }
        weights = []
        for symbol, amount in amounts.items():
            frac = float(amount) / total
            weights.append(frac)
            try:
                el = Element(symbol)
                vals = {
                    "X": float(el.X or 0.0),
                    "atomic_mass": float(el.atomic_mass or 0.0),
                    "atomic_radius": float(el.atomic_radius or 0.0),
                    "average_ionic_radius": float(getattr(el, "average_ionic_radius", None) or 0.0),
                    "row": float(el.row or 0.0),
                    "group": float(el.group or 0.0),
                    "Z": float(el.Z or 0.0),
                    "is_metal": 1.0 if el.is_metal else 0.0,
                    "is_transition_metal": 1.0 if el.is_transition_metal else 0.0,
                }
            except Exception:
                vals = {k: 0.0 for k in props}
            for k, v in vals.items():
                props[k].append(v)

        w = np.asarray(weights, dtype=np.float32)
        feats = []
        labels = []
        for name, values in props.items():
            x = np.asarray(values, dtype=np.float32)
            mean = float(np.sum(w * x))
            mn = float(np.min(x)) if len(x) else 0.0
            mx = float(np.max(x)) if len(x) else 0.0
            rng = mx - mn
            var = float(np.sum(w * (x - mean) ** 2))
            feats.extend([mean, rng, var])
            labels.extend([f"{name}_wmean", f"{name}_range", f"{name}_wvar"])

        entropy = -float(np.sum(w * np.log(np.clip(w, 1e-8, 1.0))))
        n_elem = float(len(amounts))
        max_frac = float(np.max(w)) if len(w) else 0.0
        feats.extend([entropy, n_elem, max_frac])
        labels.extend(["comp_entropy", "n_elements", "max_element_fraction"])

        return np.asarray(feats, dtype=np.float32), labels

    def _safe_extra(self, comp):
        parts = []
        labels = []
        for name, ftzr in self.extra_featurizers:
            try:
                vals = np.asarray(ftzr.featurize(comp), dtype=np.float32)
            except Exception:
                vals = np.zeros(len(ftzr.feature_labels()), dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(vals)
            try:
                labels.extend([f"{name}:{x}" for x in ftzr.feature_labels()])
            except Exception:
                labels.extend([f"{name}:{i}" for i in range(len(vals))])

        proxy_vals, proxy_labels = self._dielectric_proxies(comp)
        parts.append(proxy_vals)
        labels.extend([f"DielectricProxy:{x}" for x in proxy_labels])

        vals = np.concatenate(parts).astype(np.float32)
        if self.n_extra is None:
            self.n_extra = int(len(vals))
            self.extra_labels = labels
        return vals

    def featurize_all(self, inputs, desc="Featurizing"):
        comps = [self.composition_from_input(x) for x in inputs]
        rows = []
        if comps:
            _ = self._safe_extra(comps[0])

        for comp in tqdm(comps, desc=desc, leave=False):
            try:
                mg = np.asarray(self.ep_magpie.featurize(comp), dtype=np.float32)
            except Exception:
                mg = np.zeros(self.n_mg, dtype=np.float32)
            mg = np.nan_to_num(mg, nan=0.0, posinf=0.0, neginf=0.0)
            extra = self._safe_extra(comp)
            m2v = self._pool_mat2vec(comp)
            rows.append(np.concatenate([mg, extra, m2v]).astype(np.float32))

        X = np.vstack(rows).astype(np.float32)
        log.info(
            "Feature blocks: Magpie=%d (%dx%d), Extra=%d, Mat2Vec=%d, Total=%d",
            self.n_mg, self.mg_n_props, self.mg_stat_dim,
            self.n_extra, self.mat2vec_dim, X.shape[1],
        )
        return X

    def fit_scaler(self, X):
        self.scaler = StandardScaler().fit(X)

    def transform(self, X):
        if self.scaler is None:
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return np.nan_to_num(
            self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    def manifest(self):
        return {
            "composition_only": True,
            "structure_usage": "structure objects are used only to extract composition",
            "n_magpie": self.n_mg,
            "magpie_n_props": self.mg_n_props,
            "magpie_stat_dim": self.mg_stat_dim,
            "n_extra": self.n_extra,
            "extra_labels": self.extra_labels,
            "mat2vec_dim": self.mat2vec_dim,
            "mat2vec_loaded": self.mat2vec_ok,
            "total_dim": self.n_mg + int(self.n_extra or 0) + self.mat2vec_dim,
        }


# =========================
# 4. Target scaler
# =========================

class TargetScaler:
    def __init__(self, kind="robust_iqr"):
        self.kind = kind
        self.center = 0.0
        self.scale = 1.0

    def fit(self, y):
        y = np.asarray(y, dtype=np.float32)
        if self.kind == "robust_iqr":
            self.center = float(np.median(y))
            q75, q25 = np.percentile(y, [75, 25])
            self.scale = max(float(q75 - q25), 1e-4)
        else:
            self.center = float(np.mean(y))
            self.scale = max(float(np.std(y)), 1e-4)

    def transform(self, y):
        return ((np.asarray(y, dtype=np.float32) - self.center) / self.scale).astype(np.float32)

    def inverse_tensor(self, y):
        return y * self.scale + self.center

    def as_dict(self):
        return {"kind": self.kind, "center": self.center, "scale": self.scale}


# =========================
# 5. TRIADS model
# =========================

class DeepHybridTRM(nn.Module):
    """V13A-style two-layer self-attention + Mat2Vec cross-attention + TRM."""

    def __init__(
        self,
        n_props=22,
        stat_dim=6,
        n_extra=0,
        mat2vec_dim=200,
        d_attn=40,
        nhead=4,
        d_hidden=64,
        ff_dim=96,
        dropout=0.15,
        max_steps=16,
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.D = int(d_hidden)
        self.n_props = int(n_props)
        self.stat_dim = int(stat_dim)
        self.n_extra = int(n_extra)
        self.mat2vec_dim = int(mat2vec_dim)

        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn),
            nn.LayerNorm(d_attn),
            nn.GELU(),
        )
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn),
            nn.LayerNorm(d_attn),
            nn.GELU(),
        )

        self.sa1 = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_attn * 2, d_attn),
        )
        self.sa1_fn = nn.LayerNorm(d_attn)

        self.sa2 = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_attn * 2, d_attn),
        )
        self.sa2_fn = nn.LayerNorm(d_attn)

        self.ca = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        pool_in = d_attn + max(0, self.n_extra)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )

        self.z_up = nn.Sequential(
            nn.Linear(d_hidden * 3, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.y_up = nn.Sequential(
            nn.Linear(d_hidden * 2, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.head = nn.Linear(d_hidden, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _attention(self, x):
        B = x.size(0)
        mg_dim = self.n_props * self.stat_dim
        mg = x[:, :mg_dim]
        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:mg_dim + self.n_extra + self.mat2vec_dim]
        else:
            extra = None
            m2v = x[:, mg_dim:mg_dim + self.mat2vec_dim]

        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        tok = self.sa1_n(tok + self.sa1(tok, tok, tok, need_weights=False)[0])
        tok = self.sa1_fn(tok + self.sa1_ff(tok))
        tok = self.sa2_n(tok + self.sa2(tok, tok, tok, need_weights=False)[0])
        tok = self.sa2_fn(tok + self.sa2_ff(tok))
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx, need_weights=False)[0])

        pooled = tok.mean(dim=1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.pool(pooled)

    def forward(self, x, deep_supervision=False, return_trajectory=False):
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device, dtype=x.dtype)
        y = torch.zeros(B, self.D, device=x.device, dtype=x.dtype)
        step_preds = []
        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], dim=-1))
            y = y + self.y_up(torch.cat([y, z], dim=-1))
            step_preds.append(self.head(y).squeeze(1))

        if deep_supervision:
            return step_preds
        if return_trajectory:
            return step_preds[-1], step_preds
        return step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def deep_supervision_loss(step_preds, targets):
    weights = torch.arange(1, len(step_preds) + 1, device=targets.device, dtype=targets.dtype)
    weights = weights / weights.sum()
    losses = torch.stack([F.smooth_l1_loss(pred, targets, beta=0.5) for pred in step_preds])
    return torch.sum(weights * losses)


# =========================
# 6. Configs
# =========================

@dataclass
class ModelConfig:
    name: str
    d_attn: int
    nhead: int
    d_hidden: int
    ff_dim: int
    dropout: float
    max_steps: int
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4


CONFIGS = [
    ModelConfig("DIEL-045K-S14-D12", d_attn=32, nhead=4, d_hidden=48,
                ff_dim=80, dropout=0.12, max_steps=14),
    ModelConfig("DIEL-075K-S16-D15", d_attn=40, nhead=4, d_hidden=64,
                ff_dim=96, dropout=0.15, max_steps=16),
    ModelConfig("DIEL-115K-S18-D18", d_attn=48, nhead=4, d_hidden=80,
                ff_dim=128, dropout=0.18, max_steps=18),
    ModelConfig("DIEL-145K-S20-D20", d_attn=56, nhead=4, d_hidden=88,
                ff_dim=144, dropout=0.20, max_steps=20),
]


def model_kwargs(cfg, feature_manifest):
    return {
        "n_props": feature_manifest["magpie_n_props"],
        "stat_dim": feature_manifest["magpie_stat_dim"],
        "n_extra": feature_manifest["n_extra"],
        "mat2vec_dim": feature_manifest["mat2vec_dim"],
        "d_attn": cfg.d_attn,
        "nhead": cfg.nhead,
        "d_hidden": cfg.d_hidden,
        "ff_dim": cfg.ff_dim,
        "dropout": cfg.dropout,
        "max_steps": cfg.max_steps,
    }


# =========================
# 7. Splitting and training
# =========================

def inner_split_regression(y, val_frac=0.15, seed=42):
    y = np.asarray(y, dtype=np.float32)
    qs = np.percentile(y, [20, 40, 60, 80])
    bins = np.digitize(y, qs)
    rng = np.random.default_rng(seed)
    tr, vl = [], []
    for b in range(5):
        idx = np.where(bins == b)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        vl.extend(idx[:n_val].tolist())
        tr.extend(idx[n_val:].tolist())
    return np.asarray(tr, dtype=np.int64), np.asarray(vl, dtype=np.int64)


@torch.inference_mode()
def predict_tensor(model, X_tensor, batch_size=4096):
    model.eval()
    preds = []
    dummy_y = torch.zeros(X_tensor.shape[0], device=X_tensor.device)
    dl = FastTensorDataLoader(X_tensor, dummy_y, batch_size=batch_size, shuffle=False)
    for bx, _ in dl:
        preds.append(model(bx).detach().float().cpu())
    return torch.cat(preds)


def evaluate_mae_original(model, X_tensor, y_orig_tensor, target_scaler):
    pred_norm = predict_tensor(model, X_tensor, batch_size=BATCH_SIZE)
    pred_orig = target_scaler.inverse_tensor(pred_norm)
    return F.l1_loss(pred_orig, y_orig_tensor.cpu()).item()


def train_one_model(
    cfg,
    seed,
    fold_num,
    X_train,
    y_train_orig,
    X_val,
    y_val_orig,
    feature_manifest,
):
    set_seed(seed + 1000 * fold_num)

    feat_scaler = StandardScaler().fit(X_train)

    target_scaler = TargetScaler(TARGET_SCALER_KIND)
    target_scaler.fit(y_train_orig)

    tr_x_np = np.nan_to_num(
        feat_scaler.transform(X_train), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    vl_x_np = np.nan_to_num(
        feat_scaler.transform(X_val), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    tr_y_norm_np = target_scaler.transform(y_train_orig)
    vl_y_orig_np = np.asarray(y_val_orig, dtype=np.float32)

    tr_x = torch.tensor(tr_x_np, dtype=torch.float32, device=device)
    tr_y = torch.tensor(tr_y_norm_np, dtype=torch.float32, device=device)
    vl_x = torch.tensor(vl_x_np, dtype=torch.float32, device=device)
    vl_y_orig = torch.tensor(vl_y_orig_np, dtype=torch.float32, device=device)

    tr_dl = FastTensorDataLoader(tr_x, tr_y, batch_size=BATCH_SIZE, shuffle=True)

    kw = model_kwargs(cfg, feature_manifest)
    model = DeepHybridTRM(**kw).to(device)
    params = model.count_parameters()
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, SWA_START), eta_min=cfg.lr * 0.1
    )
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=cfg.lr * 0.5)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    history = []
    t0 = time.time()

    pbar = tqdm(range(EPOCHS), leave=False, ncols=120,
                desc=f"{cfg.name}|F{fold_num}|s{seed}")
    for ep in pbar:
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for bx, by in tr_dl:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                step_preds = model(bx, deep_supervision=True)
                loss = deep_supervision_loss(step_preds, by)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu()) * bx.shape[0]
            train_n += bx.shape[0]

        if ep < SWA_START:
            sched.step()
            phase = "cos"
        else:
            swa_model.update_parameters(model)
            swa_sched.step()
            phase = "swa"

        val_mae = evaluate_mae_original(model, vl_x, vl_y_orig, target_scaler)
        train_loss = train_loss_sum / max(train_n, 1)
        lr_now = float(opt.param_groups[0]["lr"])

        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

        history.append({
            "epoch": ep + 1,
            "train_loss_norm": train_loss,
            "val_mae": val_mae,
            "lr": lr_now,
            "phase": phase,
        })

        if ep % 10 == 0 or ep == EPOCHS - 1:
            pbar.set_postfix(
                val=f"{val_mae:.5f}",
                best=f"{best_val:.5f}",
                ep=best_epoch + 1,
                phase=phase,
            )

    # Check SWA model explicitly; keep it only if it wins on inner validation.
    if EPOCHS > SWA_START:
        swa_val = evaluate_mae_original(swa_model, vl_x, vl_y_orig, target_scaler)
        if swa_val < best_val:
            best_val = swa_val
            best_state = copy.deepcopy(swa_model.module.state_dict())
            best_epoch = EPOCHS
            history.append({
                "epoch": "swa_final",
                "train_loss_norm": None,
                "val_mae": swa_val,
                "lr": float(opt.param_groups[0]["lr"]),
                "phase": "swa_selected",
            })

    # Load best state into a fresh uncompiled model for checkpoint portability.
    final_model = DeepHybridTRM(**kw).to(device)
    final_model.load_state_dict(best_state)
    runtime = time.time() - t0

    out = {
        "model": final_model,
        "params": params,
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch + 1) if isinstance(best_epoch, int) else best_epoch,
        "history": history,
        "feature_scaler": feat_scaler,
        "target_scaler": target_scaler,
        "runtime_sec": float(runtime),
        "model_kwargs": kw,
    }

    del model, swa_model, tr_x, tr_y, vl_x, vl_y_orig
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# =========================
# 8. Matbench run
# =========================

def load_task_and_features():
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
    task = mb.tasks_map[TASK_NAME]
    task.load()

    log.info("Loaded %s", task)
    log.info("Matbench folds: %s", list(task.folds))
    log.info("Task metadata: %s", task_metadata_summary(task))

    full_df = task.df.copy()
    input_col = task.metadata.input_type
    target_col = task.metadata.target

    inputs = full_df[input_col].tolist()
    targets = full_df[target_col].astype(float).to_numpy(np.float32)

    feat = DielectricCompositionFeaturizer()
    X_all = feat.featurize_all(inputs, desc=f"{TASK_NAME} composition features")
    manifest = feat.manifest()

    np.save(Path(OUTPUT_DIR, "features", "composition_features_raw.npy"), X_all)
    pd.DataFrame({
        "matbench_index": full_df.index.astype(str).tolist(),
        "target": targets,
    }).to_csv(Path(OUTPUT_DIR, "features", "index_and_target.csv"), index=False)
    with open(Path(OUTPUT_DIR, "features", "feature_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    index_to_pos = {idx: i for i, idx in enumerate(full_df.index)}
    return mb, task, X_all, targets, index_to_pos, manifest


def select_configs(feature_manifest):
    selected = []
    for cfg in CONFIGS:
        m = DeepHybridTRM(**model_kwargs(cfg, feature_manifest))
        params = m.count_parameters()
        del m
        selected.append((cfg, params))

    with open(Path(OUTPUT_DIR, "config_param_table.json"), "w") as f:
        json.dump({
            "selected": [{"config": asdict(c), "params": p} for c, p in selected],
        }, f, indent=2)

    log.info("Selected configs:")
    for cfg, params in selected:
        log.info("  %-20s params=%d", cfg.name, params)
    return selected


def run_benchmark():
    run_start = time.time()
    set_seed(SEEDS[0])

    mb, task, X_all, y_all, index_to_pos, feature_manifest = load_task_and_features()
    selected_configs = select_configs(feature_manifest)

    fold_rows = []
    history_rows = []
    all_prediction_rows = []
    fold_selected_predictions = {}

    # Save exact official split indices for auditability.
    split_rows = []
    for fold in task.folds:
        tr_df = task.get_train_and_val_data(fold, as_type="df")
        te_df = task.get_test_data(fold, as_type="df", include_target=True)
        for idx in tr_df.index:
            split_rows.append({"fold": fold, "split": "train_val", "matbench_index": str(idx)})
        for idx in te_df.index:
            split_rows.append({"fold": fold, "split": "test", "matbench_index": str(idx)})
    pd.DataFrame(split_rows).to_csv(Path(OUTPUT_DIR, "official_split_indices.csv"), index=False)

    for fold in task.folds:
        log.info("=" * 78)
        log.info("Official fold %s/%s", fold + 1, len(task.folds))
        tr_df = task.get_train_and_val_data(fold, as_type="df")
        te_df = task.get_test_data(fold, as_type="df", include_target=True)

        trval_pos = np.asarray([index_to_pos[idx] for idx in tr_df.index], dtype=np.int64)
        test_pos = np.asarray([index_to_pos[idx] for idx in te_df.index], dtype=np.int64)
        X_trval = X_all[trval_pos]
        y_trval = y_all[trval_pos]
        X_test = X_all[test_pos]
        y_test = y_all[test_pos]

        candidate_results = []

        for cfg, params in selected_configs:
            seed_results = []
            log.info("-- Candidate %s | params=%d", cfg.name, params)

            for seed in SEEDS:
                inner_tr, inner_val = inner_split_regression(
                    y_trval, val_frac=INNER_VAL_FRAC, seed=seed + fold * 997
                )
                result = train_one_model(
                    cfg=cfg,
                    seed=seed,
                    fold_num=fold + 1,
                    X_train=X_trval[inner_tr],
                    y_train_orig=y_trval[inner_tr],
                    X_val=X_trval[inner_val],
                    y_val_orig=y_trval[inner_val],
                    feature_manifest=feature_manifest,
                )

                hist_path = Path(
                    OUTPUT_DIR, "histories",
                    f"{TASK_NAME}_{cfg.name}_fold{fold+1}_seed{seed}_history.csv",
                )
                pd.DataFrame(result["history"]).to_csv(hist_path, index=False)
                for row in result["history"]:
                    history_rows.append({
                        "fold": fold + 1,
                        "config": cfg.name,
                        "seed": seed,
                        **row,
                    })

                ckpt_path = Path(
                    OUTPUT_DIR, "checkpoints",
                    f"{TASK_NAME}_{cfg.name}_fold{fold+1}_seed{seed}.pt",
                )
                torch.save({
                    "dataset": TASK_NAME,
                    "fold": fold + 1,
                    "seed": seed,
                    "config": asdict(cfg),
                    "params": result["params"],
                    "best_val_mae": result["best_val_mae"],
                    "best_epoch": result["best_epoch"],
                    "model_kwargs": result["model_kwargs"],
                    "model_state": result["model"].state_dict(),
                    "target_scaler": result["target_scaler"].as_dict(),
                    "feature_manifest": feature_manifest,
                    "composition_only": True,
                }, ckpt_path)

                seed_results.append(result)
                log.info(
                    "   seed=%s val_mae=%.6f best_epoch=%s runtime=%.1fs",
                    seed, result["best_val_mae"], result["best_epoch"],
                    result["runtime_sec"],
                )

            mean_val = float(np.mean([r["best_val_mae"] for r in seed_results]))
            std_val = float(np.std([r["best_val_mae"] for r in seed_results]))
            candidate_results.append({
                "cfg": cfg,
                "params": params,
                "seed_results": seed_results,
                "mean_val_mae": mean_val,
                "std_val_mae": std_val,
            })
            log.info(
                "Candidate summary: %s mean_val=%.6f std=%.6f",
                cfg.name, mean_val, std_val,
            )

        # Legal selection: choose by inner validation only.
        chosen = min(candidate_results, key=lambda r: r["mean_val_mae"])
        chosen_cfg = chosen["cfg"]
        log.info(
            "Selected for official fold %d by inner-val: %s | val=%.6f +/- %.6f",
            fold + 1, chosen_cfg.name, chosen["mean_val_mae"], chosen["std_val_mae"],
        )

        # Predict official test fold only with the validation-selected config.
        test_seed_preds = []
        test_seed_val_maes = []
        for result, seed in zip(chosen["seed_results"], SEEDS):
            X_test_scaled = np.nan_to_num(
                result["feature_scaler"].transform(X_test),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            te_x = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
            pred_norm = predict_tensor(result["model"], te_x, batch_size=BATCH_SIZE)
            pred_orig = result["target_scaler"].inverse_tensor(pred_norm).numpy()
            test_seed_preds.append(pred_orig)
            test_seed_val_maes.append(result["best_val_mae"])

            pred_path = Path(
                OUTPUT_DIR, "predictions",
                f"{TASK_NAME}_{chosen_cfg.name}_fold{fold+1}_seed{seed}_test_predictions.csv",
            )
            pred_df = pd.DataFrame({
                "matbench_index": te_df.index.astype(str).tolist(),
                "prediction": pred_orig,
                "target_for_local_scoring": y_test,
            })
            pred_df.to_csv(pred_path, index=False)

        pred_stack = np.vstack(test_seed_preds)
        ensemble_pred = pred_stack.mean(axis=0).astype(float)
        ensemble_std = pred_stack.std(axis=0).astype(float)
        local_test_mae = float(np.mean(np.abs(ensemble_pred - y_test)))

        task.record(
            fold,
            predictions=ensemble_pred.tolist(),
            std=ensemble_std.tolist(),
            params={
                "algorithm": "TRIADS composition-only",
                "selected_config": asdict(chosen_cfg),
                "params": chosen["params"],
                "selection_rule": "lowest mean inner-validation MAE across seeds",
                "inner_val_mae_mean": chosen["mean_val_mae"],
                "inner_val_mae_std": chosen["std_val_mae"],
                "seeds": SEEDS,
                "composition_only": True,
            },
        )

        fold_selected_predictions[fold] = ensemble_pred
        all_prediction_rows.extend([
            {
                "fold": fold + 1,
                "matbench_index": str(idx),
                "prediction": float(pred),
                "prediction_seed_std": float(std),
                "target_for_local_scoring": float(tgt),
                "selected_config": chosen_cfg.name,
            }
            for idx, pred, std, tgt in zip(te_df.index, ensemble_pred, ensemble_std, y_test)
        ])

        for cand in candidate_results:
            fold_rows.append({
                "fold": fold + 1,
                "config": cand["cfg"].name,
                "params": cand["params"],
                "mean_inner_val_mae": cand["mean_val_mae"],
                "std_inner_val_mae": cand["std_val_mae"],
                "selected_for_fold": cand["cfg"].name == chosen_cfg.name,
                "local_test_mae_if_selected": local_test_mae if cand["cfg"].name == chosen_cfg.name else None,
            })

        log.info(
            "Official fold %d local test MAE for selected ensemble: %.6f",
            fold + 1, local_test_mae,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Validation through Matbench.
    task.validate()
    scores = to_plain(task.scores)
    total_min = (time.time() - run_start) / 60.0

    try:
        from monty.serialization import dumpfn
        dumpfn(mb.as_dict(), Path(OUTPUT_DIR, "matbench_record.json.gz"))
    except Exception as exc:
        log.warning("Could not dump Matbench record via monty: %s", exc)
        with open(Path(OUTPUT_DIR, "matbench_record.json"), "w") as f:
            json.dump(to_plain(mb.as_dict()), f, indent=2)

    pd.DataFrame(fold_rows).to_csv(Path(OUTPUT_DIR, "fold_config_summary.csv"), index=False)
    pd.DataFrame(history_rows).to_csv(Path(OUTPUT_DIR, "all_histories.csv"), index=False)
    pd.DataFrame(all_prediction_rows).to_csv(
        Path(OUTPUT_DIR, "official_fold_predictions_selected.csv"), index=False
    )

    summary = {
        "dataset": TASK_NAME,
        "task_type": "regression",
        "metric": "MAE",
        "unit": "unitless refractive index n",
        "composition_only": True,
        "official_matbench_folds": True,
        "legal_selection": "per-fold config selected by inner validation only",
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "swa_start": SWA_START,
        "batch_size": BATCH_SIZE,
        "feature_manifest": feature_manifest,
        "scores": scores,
        "total_time_min": round(total_min, 2),
    }
    with open(Path(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 78)
    log.info("FINAL MATBENCH SCORES")
    log.info(json.dumps(scores, indent=2))
    log.info("Total time: %.2f min", total_min)

    if Path(ZIP_NAME).exists():
        Path(ZIP_NAME).unlink()
    shutil.make_archive(OUTPUT_DIR, "zip", ".", OUTPUT_DIR)
    log.info("Saved artifact zip: %s", ZIP_NAME)

    print("\n" + "=" * 78)
    print("TRIADS dielectric run complete")
    print(f"Outputs: {OUTPUT_DIR}/")
    print(f"Zip:     {ZIP_NAME}")
    print("Matbench scores:")
    print(json.dumps(scores, indent=2))
    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
