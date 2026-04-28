from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_ATOM_FEAT, N_COMP, N_EDGE_FEAT, N_EDGE_STATIC, N_GLOBAL, N_RBF


@dataclass
class ConfigV5:
    n_atom_feat: int = N_ATOM_FEAT
    n_edge_feat: int = N_EDGE_FEAT
    n_edge_static: int = N_EDGE_STATIC
    n_comp_feat: int = N_COMP
    n_global_feat: int = N_GLOBAL
    d_model: int = 192
    d_edge: int = 96
    d_comp: int = 96
    n_heads: int = 8
    d_ff: int = 384
    n_gat_layers: int = 2
    max_cycles: int = 6
    cutoff: float = 6.0
    n_rbf: int = N_RBF
    d_head: int = field(init=False)

    def __post_init__(self) -> None:
        self.d_head = self.d_model // self.n_heads


class FeedForward(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int | None = None):
        super().__init__()
        d_out = d_out or d_in
        self.net = nn.Sequential(nn.Linear(d_in, d_ff), nn.SiLU(), nn.Linear(d_ff, d_out))
        self.norm = nn.LayerNorm(d_out)
        self.residual = d_in == d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.norm(x + y) if self.residual else self.norm(y)


class GatedResidual(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        return g * x + (1.0 - g) * h


class RBFLayer(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.log_gamma = nn.Parameter(torch.full((n_rbf,), math.log(10.0)))
        self.cutoff = cutoff

    def forward(self, dists: torch.Tensor) -> torch.Tensor:
        d = dists.reshape(-1, 1)
        return torch.exp(-self.log_gamma.exp().reshape(1, -1) * (d - self.centers.reshape(1, -1)).pow(2))


def pad_by_sizes(x: torch.Tensor, sizes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    b = len(sizes)
    max_n = max(sizes)
    pad = x.new_zeros(b, max_n, x.size(-1))
    mask = torch.ones(b, max_n, dtype=torch.bool, device=x.device)
    offset = 0
    for i, sz in enumerate(sizes):
        pad[i, :sz] = x[offset : offset + sz]
        mask[i, :sz] = False
        offset += sz
    return pad, mask


def compute_dynamic_edges(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shift_cart: torch.Tensor,
    rbf: RBFLayer,
    cutoff: float,
) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    vec = positions[dst] + edge_shift_cart - positions[src]
    d = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(0.01)
    dirs = vec / d
    x = (d / cutoff).clamp(0.0, 1.0)
    cutoff_val = 0.5 * (torch.cos(math.pi * x) + 1.0)
    return torch.cat([rbf(d.squeeze(-1)), dirs, cutoff_val], dim=-1)


class GATLayer(nn.Module):
    def __init__(self, cfg: ConfigV5):
        super().__init__()
        d = cfg.d_model
        self.n_h = cfg.n_heads
        self.d_h = cfg.d_head
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.e = nn.Linear(cfg.d_edge, cfg.n_heads, bias=False)
        self.out = nn.Linear(d, d)
        self.gate = GatedResidual(d)
        self.norm = nn.LayerNorm(d)
        self.ff = FeedForward(d, d * 2)

    def forward(self, x: torch.Tensor, edge_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        q = self.q(x[dst]).view(-1, self.n_h, self.d_h)
        k = self.k(x[src]).view(-1, self.n_h, self.d_h)
        v = self.v(x[src]).view(-1, self.n_h, self.d_h)
        score = (q * k).sum(-1) / math.sqrt(self.d_h) + self.e(edge_feat)

        max_per_dst = torch.full((n, self.n_h), -torch.inf, device=x.device, dtype=score.dtype)
        max_per_dst.scatter_reduce_(0, dst[:, None].expand(-1, self.n_h), score, reduce="amax", include_self=True)
        alpha = torch.exp(score - max_per_dst[dst])
        den = torch.zeros(n, self.n_h, device=x.device, dtype=alpha.dtype)
        den.index_add_(0, dst, alpha)

        msg = (alpha[..., None] * v).reshape(-1, self.n_h * self.d_h).to(v.dtype)
        num = torch.zeros(n, self.n_h * self.d_h, device=x.device, dtype=msg.dtype)
        num.index_add_(0, dst, msg)
        agg = (num.view(n, self.n_h, self.d_h) / (den[:, :, None].to(num.dtype) + 1e-8)).reshape(n, -1)
        y = self.norm(self.gate(self.out(agg), x))
        return self.ff(y)


class FullSelfAttention(nn.Module):
    def __init__(self, cfg: ConfigV5):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.gate = GatedResidual(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_model * 2)

    def forward(self, x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        pad, mask = pad_by_sizes(x, sizes)
        out, _ = self.attn(pad, pad, pad, key_padding_mask=mask)
        pieces = []
        offset = 0
        for i, sz in enumerate(sizes):
            h = self.norm(self.gate(out[i, :sz], x[offset : offset + sz]))
            pieces.append(self.ff(h))
            offset += sz
        return torch.cat(pieces, dim=0)


class StructuralEncoder(nn.Module):
    def __init__(self, cfg: ConfigV5):
        super().__init__()
        self.atom_proj = FeedForward(cfg.n_atom_feat, cfg.d_model * 2, cfg.d_model)
        self.edge_proj = FeedForward(cfg.n_edge_feat, cfg.d_edge * 2, cfg.d_edge)
        self.gats = nn.ModuleList([GATLayer(cfg) for _ in range(cfg.n_gat_layers)])
        self.full_sa = FullSelfAttention(cfg)

    def forward(self, atom_feat: torch.Tensor, edge_feat: torch.Tensor, edge_index: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        x = self.atom_proj(atom_feat)
        e = self.edge_proj(edge_feat)
        for gat in self.gats:
            x = gat(x, e, edge_index)
        return self.full_sa(x, sizes)


class CompositionEncoder(nn.Module):
    def __init__(self, cfg: ConfigV5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_comp_feat + cfg.n_global_feat, cfg.d_comp * 2),
            nn.SiLU(),
            nn.Linear(cfg.d_comp * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, comp: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([comp, global_feat], dim=-1))


class TRMCycle(nn.Module):
    def __init__(self, cfg: ConfigV5):
        super().__init__()
        self.atom_sa = FullSelfAttention(cfg)
        self.cross_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.cross_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.gate = GatedResidual(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, comp: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        h = self.atom_sa(x, sizes)
        # There is only one composition token per crystal, so cross-attention
        # collapses to broadcasting a transformed composition context.
        comp_ctx = self.cross_out(self.cross_v(comp))
        repeat_sizes = torch.as_tensor(sizes, device=comp.device, dtype=torch.long)
        c = torch.repeat_interleave(comp_ctx, repeat_sizes, dim=0)
        return self.ff(self.norm(self.gate(c, h)))


class EnergyHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))

    def forward(self, x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        pad, mask = pad_by_sizes(x, sizes)
        logits = self.attn(pad).squeeze(-1)
        logits = logits.masked_fill(mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * pad).sum(dim=1)
        return self.mlp(pooled).squeeze(-1)


class CrystalFoldV5(nn.Module):
    def __init__(self, cfg: ConfigV5 | None = None):
        super().__init__()
        self.cfg = cfg or ConfigV5()
        self.rbf = RBFLayer(self.cfg.n_rbf, self.cfg.cutoff)
        self.struct = StructuralEncoder(self.cfg)
        self.comp = CompositionEncoder(self.cfg)
        self.warmup = TRMCycle(self.cfg)
        self.shared = TRMCycle(self.cfg)
        self.energy = EnergyHead(self.cfg.d_model)
        self.deep = nn.ModuleList([EnergyHead(self.cfg.d_model) for _ in range(self.cfg.max_cycles)])
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict[str, torch.Tensor], positions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if positions is None:
            edge_feat = batch["edge_features"]
        else:
            dyn = compute_dynamic_edges(
                positions,
                batch["edge_index"],
                batch["edge_shift_cart"],
                self.rbf,
                self.cfg.cutoff,
            )
            edge_feat = torch.cat([dyn, batch["edge_static"]], dim=-1)
        sizes = batch.get("crystal_sizes_list")
        if sizes is None:
            sizes = [int(x) for x in batch["crystal_sizes"].detach().cpu().tolist()]
        x = self.struct(batch["atom_features"], edge_feat, batch["edge_index"], sizes)
        comp = self.comp(batch["comp_features"], batch["global_features"])
        cycle_preds = []
        x = self.warmup(x, comp, sizes)
        cycle_preds.append(self.deep[0](x, sizes))
        for i in range(1, self.cfg.max_cycles):
            x = self.shared(x, comp, sizes)
            cycle_preds.append(self.deep[i](x, sizes))
        return {"final_pred": self.energy(x, sizes), "cycle_preds": torch.stack(cycle_preds)}

    @staticmethod
    def total_energy_from_energy_per_atom(energy_pa: torch.Tensor, crystal_sizes: torch.Tensor) -> torch.Tensor:
        return (energy_pa * crystal_sizes.to(dtype=energy_pa.dtype)).sum()

    def energy_and_forces(self, batch: dict[str, torch.Tensor], positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.requires_grad_(True)
        out = self.forward(batch, positions)
        total_e = self.total_energy_from_energy_per_atom(out["final_pred"], batch["crystal_sizes"])
        forces = -torch.autograd.grad(total_e, positions, create_graph=self.training, retain_graph=self.training)[0]
        return out["final_pred"], forces

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiscoveryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_energy = nn.Parameter(torch.tensor(0.0))
        self.log_var_force = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        pred_e: torch.Tensor,
        target_e: torch.Tensor,
        pred_f: torch.Tensor,
        target_f: torch.Tensor,
        cycle_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        e_loss = F.huber_loss(pred_e, target_e, delta=0.1)
        if cycle_preds is not None and cycle_preds.size(0) > 1:
            ds = torch.stack([F.huber_loss(cycle_preds[i], target_e, delta=0.1) for i in range(cycle_preds.size(0) - 1)]).mean()
            e_loss = 0.7 * e_loss + 0.3 * ds
        f_loss = F.mse_loss(pred_f, target_f)
        total = torch.exp(-self.log_var_energy) * e_loss + self.log_var_energy
        total = total + torch.exp(-self.log_var_force) * f_loss + self.log_var_force
        return total, {"energy_loss": float(e_loss.detach()), "force_loss": float(f_loss.detach())}

    def energy_only(
        self,
        pred_e: torch.Tensor,
        target_e: torch.Tensor,
        cycle_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        e_loss = F.huber_loss(pred_e, target_e, delta=0.1)
        if cycle_preds is not None and cycle_preds.size(0) > 1:
            ds = torch.stack([F.huber_loss(cycle_preds[i], target_e, delta=0.1) for i in range(cycle_preds.size(0) - 1)]).mean()
            e_loss = 0.7 * e_loss + 0.3 * ds
        total = torch.exp(-self.log_var_energy) * e_loss + self.log_var_energy
        return total, {"energy_loss": float(e_loss.detach()), "force_loss": 0.0}
