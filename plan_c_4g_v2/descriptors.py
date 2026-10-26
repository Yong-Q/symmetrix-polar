"""Radial and angular local descriptors for the 4G-HDNNP student.

Radial: element-resolved Gaussian RBF sums (an ACSF G2 analogue).
Angular: element-pair-resolved G4-like sums over neighbour pairs,

    G4 = sum_{j<k} exp(-eta (r_ij^2+r_ik^2+r_jk^2))
         (1 + lambda cos(theta_jik))^zeta  fc(r_ij) fc(r_ik) fc(r_jk)

with a smooth cosine cutoff.  Both descriptor sets are sums of differentiable
functions of the positions, so autograd gives exact descriptor forces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .neighbors import NeighborData


# atomic numbers supported by the six-molecule dataset
Z_SUPPORTED = (1, 6, 7, 8)
ELEMENT_INDEX = {z: i for i, z in enumerate(Z_SUPPORTED)}
N_ELEM = len(Z_SUPPORTED)

# symmetric element-pair channel table (10 channels for 4 elements)
_PAIR_CHANNEL = torch.full((N_ELEM, N_ELEM), -1, dtype=torch.long)
_k = 0
for _a in range(N_ELEM):
    for _b in range(_a, N_ELEM):
        _PAIR_CHANNEL[_a, _b] = _k
        _PAIR_CHANNEL[_b, _a] = _k
        _k += 1
N_PAIR = _k


def species_to_index(z: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(z, -1)
    for atomic_z, idx in ELEMENT_INDEX.items():
        out[z == atomic_z] = idx
    if (out < 0).any():
        raise ValueError(f"unsupported elements: {z[out < 0].unique().tolist()}")
    return out


@dataclass
class DescriptorConfig:
    r_cut_rad: float = 6.0
    n_rbf: int = 12
    r_min: float = 0.55
    rbf_width: float = 0.45
    r_cut_ang: float = 3.5
    # (eta [A^-2], zeta, lambda)
    angular_basis: Sequence[Tuple[float, float, float]] = field(
        default_factory=lambda: (
            (0.30, 2.0, 1.0), (0.30, 2.0, -1.0),
            (0.30, 4.0, 1.0), (0.30, 4.0, -1.0),
            (1.50, 2.0, 1.0), (1.50, 2.0, -1.0),
            (1.50, 4.0, 1.0), (1.50, 4.0, -1.0),
        )
    )

    @property
    def n_ang(self) -> int:
        return len(self.angular_basis)

    @property
    def cont_dim(self) -> int:
        return self.n_rbf * N_ELEM + self.n_ang * N_PAIR


def angular_basis_preset(n: int) -> Tuple[Tuple[float, float, float], ...]:
    """Return an element-resolved G4 basis of size n.

    8 is the original preset.  16/24 widen the eta/zeta grid, which is the
    part that C2H4/C2H6 need for C-C bond and torsion resolution.
    """
    if n == 8:
        return (
            (0.30, 2.0, 1.0), (0.30, 2.0, -1.0),
            (0.30, 4.0, 1.0), (0.30, 4.0, -1.0),
            (1.50, 2.0, 1.0), (1.50, 2.0, -1.0),
            (1.50, 4.0, 1.0), (1.50, 4.0, -1.0),
        )
    if n == 16:
        return tuple((eta, zeta, lam)
                     for eta in (0.10, 0.50, 1.50, 6.00)
                     for zeta in (2.0, 4.0)
                     for lam in (1.0, -1.0))
    if n == 24:
        return tuple((eta, zeta, lam)
                     for eta in (0.10, 0.50, 1.50)
                     for zeta in (2.0, 3.0, 4.0, 6.0)
                     for lam in (1.0, -1.0))
    raise ValueError(f"unsupported angular preset {n}; use 8, 16, or 24")


def _radial_centers(cfg: DescriptorConfig, device, dtype):
    mu = torch.linspace(cfg.r_min, cfg.r_cut_rad, cfg.n_rbf, device=device, dtype=dtype)
    return mu


def cutoff(r, rc):
    """Smooth cosine cutoff; zero and zero-derivative at rc."""
    return 0.5 * (torch.cos(torch.pi * r / rc) + 1.0)


def radial_features(pos, cell, nl: NeighborData, zi: torch.Tensor,
                    cfg: DescriptorConfig):
    """Return [N, n_rbf * N_ELEM] continuous radial features."""
    dist, _ = nl.geometry(pos, cell)                   # [N,K]
    mask = nl.mask
    r = torch.where(mask, dist, torch.zeros_like(dist))
    fc = cutoff(r, cfg.r_cut_rad) * mask
    mu = _radial_centers(cfg, pos.device, pos.dtype)
    rbf = torch.exp(-((r.unsqueeze(-1) - mu) / cfg.rbf_width) ** 2)   # [N,K,n_rbf]
    w = rbf * fc.unsqueeze(-1)
    z_nbr = zi[nl.nbr]                                 # [N,K]
    idx = z_nbr.unsqueeze(-1).expand(-1, -1, cfg.n_rbf)
    out = torch.zeros(pos.shape[0], N_ELEM, cfg.n_rbf,
                      device=pos.device, dtype=pos.dtype)
    out.scatter_add_(1, idx, w)
    return out.reshape(pos.shape[0], N_ELEM * cfg.n_rbf)


def _cell_context(cell: torch.Tensor, batch: torch.Tensor):
    """Return (cell_or_per_atom, inv_or_per_atom, homogeneous)."""
    if cell.dim() == 2:
        return cell, torch.linalg.inv(cell), True
    if bool(torch.allclose(cell, cell[0].expand_as(cell))):
        c0 = cell[0]
        return c0, torch.linalg.inv(c0), True
    ca = cell[batch]
    return ca, torch.linalg.inv(ca), False


def angular_features(pos, cell, nl: NeighborData, zi: torch.Tensor,
                     cfg: DescriptorConfig):
    """Return [N, n_ang * N_PAIR] continuous angular features."""
    dist, vec = nl.geometry(pos, cell)                 # [N,Ka], [N,Ka,3]
    mask = nl.mask
    n, ka = dist.shape
    if ka < 2:
        return torch.zeros(n, cfg.n_ang * N_PAIR, device=pos.device, dtype=pos.dtype)
    ctx_cell, ctx_inv, homogeneous = _cell_context(cell, nl.batch)
    iu, iv = torch.triu_indices(ka, ka, offset=1, device=pos.device)
    d1 = dist[:, iu]
    d2 = dist[:, iv]
    m = mask[:, iu] & mask[:, iv]
    v1 = vec[:, iu, :]
    v2 = vec[:, iv, :]
    # minimum image for the neighbour-neighbour vector
    dv = v1 - v2
    if homogeneous:
        fr = dv @ ctx_inv
        fr = fr - torch.round(fr)
        dv = fr @ ctx_cell
    else:
        fr = torch.einsum("npi,nij->npj", dv, ctx_inv)
        fr = fr - torch.round(fr)
        dv = torch.einsum("npi,nij->npj", fr, ctx_cell)
    r12 = torch.linalg.norm(dv, dim=-1)
    denom = d1 * d2 + 1e-12
    cos = (v1 * v2).sum(dim=-1) / denom
    cos = cos.clamp(-1.0, 1.0)
    fc1 = cutoff(d1, cfg.r_cut_ang)
    fc2 = cutoff(d2, cfg.r_cut_ang)
    fc12 = cutoff(r12, cfg.r_cut_ang)
    base = fc1 * fc2 * fc12 * m
    pair_ch = _PAIR_CHANNEL.to(pos.device)[zi[nl.nbr][:, iu], zi[nl.nbr][:, iv]]
    out = torch.zeros(n, N_PAIR, cfg.n_ang, device=pos.device, dtype=pos.dtype)
    idx = pair_ch.unsqueeze(-1).expand(-1, -1, cfg.n_ang)
    r2sum = d1 * d1 + d2 * d2 + r12 * r12
    for k, (eta, zeta, lam) in enumerate(cfg.angular_basis):
        ang = torch.exp(-eta * r2sum) * (1.0 + lam * cos).clamp(min=0.0) ** zeta * base
        out[:, :, k].scatter_add_(1, pair_ch, ang)
    return out.reshape(n, N_PAIR * cfg.n_ang)


class FeatureScaler(torch.nn.Module):
    """Per-feature standardisation with buffers (state_dict friendly)."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    def fit(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return
        self.mean.copy_(x.mean(dim=0))
        std = x.std(dim=0)
        rms = x.pow(2).mean(dim=0).sqrt()
        # A feature that is negligible in the fit set but appears in another
        # density regime must not get a near-zero scale (that produced 1e5
        # sigma bulk inputs from a gas-only fit).  Floor the scale at 1e-2 and
        # use unit scale for features whose RMS is below that.
        std = torch.where(rms > 1e-2, std.clamp(min=1e-2), torch.ones_like(std))
        self.std.copy_(std)
        self.count.fill_(float(x.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.count) == 0.0:
            return x
        return (x - self.mean) / self.std


def build_descriptors(pos, cell, nl_rad: NeighborData, nl_ang: NeighborData,
                      zi: torch.Tensor, cfg: DescriptorConfig):
    """Return (continuous [N,D], one_hot_center [N,4])."""
    g2 = radial_features(pos, cell, nl_rad, zi, cfg)
    g4 = angular_features(pos, cell, nl_ang, zi, cfg)
    cont = torch.cat([g2, g4], dim=1)
    onehot = F.one_hot(zi, N_ELEM).to(pos.dtype)
    return cont, onehot
