from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FIREConfig:
    dt_start: float = 0.05
    dt_max: float = 0.5
    f_inc: float = 1.1
    f_dec: float = 0.5
    alpha_start: float = 0.1
    f_alpha: float = 0.99
    n_min: int = 5
    fmax_threshold: float = 0.05
    max_steps: int = 200
    energy_tol: float = 1e-5
    energy_window: int = 5


class CrystalFIREState:
    def __init__(self, n_atoms: int, device: torch.device, cfg: FIREConfig):
        self.cfg = cfg
        self.velocity = torch.zeros(n_atoms, 3, device=device)
        self.dt = cfg.dt_start
        self.alpha = cfg.alpha_start
        self.n_pos = 0
        self.converged = False
        self.step_count = 0
        self.energy_history: list[float] = []

    def step(self, forces: torch.Tensor) -> torch.Tensor:
        if self.converged:
            return torch.zeros_like(forces)
        power = (forces * self.velocity).sum()
        if power > 0:
            self.n_pos += 1
            f_hat = forces / (forces.norm() + 1e-10)
            v_norm = self.velocity.norm()
            self.velocity = (1.0 - self.alpha) * self.velocity + self.alpha * v_norm * f_hat
            if self.n_pos > self.cfg.n_min:
                self.dt = min(self.dt * self.cfg.f_inc, self.cfg.dt_max)
                self.alpha = self.alpha * self.cfg.f_alpha
        else:
            self.n_pos = 0
            self.velocity.zero_()
            self.dt = self.dt * self.cfg.f_dec
            self.alpha = self.cfg.alpha_start
        self.velocity = self.velocity + self.dt * forces
        delta_pos = self.dt * self.velocity
        self.step_count += 1
        return delta_pos

    def update_energy(self, energy_pa: float) -> None:
        self.energy_history.append(energy_pa)
        if len(self.energy_history) > self.cfg.energy_window:
            self.energy_history.pop(0)

    def check_converged(self, forces: torch.Tensor) -> bool:
        if self.converged:
            return True
        if forces.norm(dim=-1).max().item() < self.cfg.fmax_threshold:
            self.converged = True
            return True
        if len(self.energy_history) >= self.cfg.energy_window:
            if max(self.energy_history) - min(self.energy_history) < self.cfg.energy_tol:
                self.converged = True
                return True
        if self.step_count >= self.cfg.max_steps:
            self.converged = True
            return True
        return False


class FIRERelaxer:
    def __init__(self, cfg: FIREConfig):
        self.cfg = cfg

    def init(self, crystal_sizes: list[int], positions: torch.Tensor) -> None:
        self.positions = positions.clone().detach()
        self.crystal_sizes = crystal_sizes
        self.device = positions.device
        self.states = [CrystalFIREState(sz, self.device, self.cfg) for sz in crystal_sizes]
        self.global_step = 0

    @property
    def all_converged(self) -> bool:
        return all(s.converged for s in self.states)

    def step(self, forces: torch.Tensor, energies_pa: torch.Tensor) -> int:
        delta_pos = torch.zeros_like(self.positions)
        n_new = 0
        offset = 0
        for i, (state, sz) in enumerate(zip(self.states, self.crystal_sizes)):
            f = forces[offset : offset + sz]
            e = float(energies_pa[i].item())
            state.update_energy(e)
            was_converged = state.converged
            is_converged = state.check_converged(f)
            if not was_converged and is_converged:
                n_new += 1
            if not state.converged:
                delta_pos[offset : offset + sz] = state.step(f)
            offset += sz
        self.positions = self.positions + delta_pos
        self.global_step += 1
        return n_new

    def max_forces(self, forces: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(self.crystal_sizes), device=forces.device)
        offset = 0
        for i, sz in enumerate(self.crystal_sizes):
            out[i] = forces[offset : offset + sz].norm(dim=-1).max()
            offset += sz
        return out

    def converged_mask(self) -> list[bool]:
        return [s.converged for s in self.states]

    def steps_used(self) -> list[int]:
        return [s.step_count for s in self.states]
