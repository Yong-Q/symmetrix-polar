"""4th-generation HDNNP-style student: local MLP + global QEq/PME.

Energy model
------------
    E = sum_i E_i(G_i, q_i) + 1/2 q^T E_coul q + sum_i eps_{z_i}

with QEq charges

    (E_coul + diag(J)) q + chi = -mu 1 ,   1^T q = 0

chi_i and J_i are local MLP outputs; q_i is fed back into the atomic energy
network (the 4G coupling).  Forces use the force trick: one adjoint solve of
the same QEq operator, then a single autograd pass through a surrogate energy.
The same surrogate also gives the exact parameter gradient by the envelope
theorem (the ``E_train`` straight-through value keeps the reported energy equal
to the physical prediction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .descriptors import (DescriptorConfig, FeatureScaler, N_ELEM,
                          Z_SUPPORTED, build_descriptors)
from .qeq import QEqPME
from .vdw import TkatchenkoSchefflerDispersion


@dataclass
class ModelConfig:
    hidden: int = 128
    trunk_hidden: int = 96
    head_hidden: int = 64
    property_hidden: int = 32
    q_scale: float = 0.5
    j_min: float = 0.5
    sigma: float = 0.6
    h_target: float = 0.8
    cg_iters: int = 60
    cg_tol: float = 1e-9
    # Number of hidden layers in the shared trunk. 1 reproduces the original
    # two-Linear trunk; larger values add [hidden -> hidden] SiLU blocks while
    # keeping the final trunk_hidden projection. Capacity is the binding
    # constraint on the bulk gate, so this is exposed as a first-class knob.
    n_trunk_layers: int = 1
    # Number of layers in the chi/J and energy head MLPs (1 = original).
    n_property_layers: int = 1
    # Tkatchenko-Scheffler dispersion (paper Eq. 5, third term). The first
    # implementation omitted it entirely; that omission is the measured cause
    # of the density-growing force error. Default on to match the paper.
    use_vdw: bool = True
    vdw_cutoff: float = 12.0
    vdw_damping_sR: float = 0.94
    vdw_damping_d: float = 20.0
    # RuNNer2 Eq. 12/14: the Coulomb operator includes the Gaussian self term
    # 1/(sigma sqrt(pi)) on its diagonal.  It was previously only in the CG
    # preconditioner, so the solved charges and the reported energy were
    # inconsistent.  On by default for all new training runs.
    include_self_energy: bool = True
    # RuNNer2 Sec. II E: initialise chi/J from Pearson electronegativity and
    # hardness (1-20x).  The original zero-chi / J~1.8 eV init produced
    # q_max ~ 1 e and the wrong charge scale.  Off by default so old
    # checkpoints load unchanged; new runs pass physical_init=True.
    physical_init: bool = False


class FourGStudent(nn.Module):
    def __init__(self, dcfg: Optional[DescriptorConfig] = None,
                 mcfg: Optional[ModelConfig] = None):
        super().__init__()
        self.dcfg = dcfg or DescriptorConfig()
        self.mcfg = mcfg or ModelConfig()
        m = self.mcfg
        d = self.dcfg
        self.scaler = FeatureScaler(d.cont_dim)
        in_dim = d.cont_dim + N_ELEM
        trunk = [nn.Linear(in_dim, m.hidden), nn.SiLU()]
        for _ in range(max(0, m.n_trunk_layers - 1)):
            trunk += [nn.Linear(m.hidden, m.hidden), nn.SiLU()]
        trunk += [nn.Linear(m.hidden, m.trunk_hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*trunk)

        def _head(in_features: int, out_features: int = 1):
            layers = [nn.Linear(in_features, m.property_hidden), nn.SiLU()]
            for _ in range(max(0, m.n_property_layers - 1)):
                layers += [nn.Linear(m.property_hidden, m.property_hidden), nn.SiLU()]
            layers += [nn.Linear(m.property_hidden, out_features)]
            return nn.Sequential(*layers)

        self.chi_head = _head(m.trunk_hidden)
        self.j_head = _head(m.trunk_hidden)
        self.e_head = nn.Sequential(
            nn.Linear(m.trunk_hidden + 1, m.head_hidden), nn.SiLU(),
            nn.Linear(m.head_hidden, 1),
        )
        # Element-specific Hirshfeld volume-ratio networks (paper Sec. II F 1).
        # One small MLP per supported element; selecting with the one-hot is a
        # gather, so this stays cheap on the GPU. Created only when the
        # dispersion term is enabled so the ablation is a genuine parameter
        # comparison.
        self.v_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m.trunk_hidden, m.property_hidden), nn.SiLU(),
                nn.Linear(m.property_hidden, 1),
            ) for _ in range(N_ELEM)
        ]) if m.use_vdw else None
        self.eps = nn.Parameter(torch.zeros(N_ELEM))
        # Element-resolved Pearson offsets (H, C, N, O).  chi is the Mulliken
        # electronegativity in eV; J is the Pearson hardness in eV.  Only
        # differences in chi matter because QEq enforces sum(q)=0.
        if m.physical_init:
            self.chi_elem = nn.Parameter(
                torch.tensor([7.18, 6.27, 7.54, 8.34], dtype=torch.get_default_dtype()))
            self.j_elem = nn.Parameter(
                torch.tensor([6.42, 5.00, 7.30, 6.08], dtype=torch.get_default_dtype()))
        else:
            self.register_parameter("chi_elem", None)
            self.register_parameter("j_elem", None)
        self.qeq = QEqPME(sigma=m.sigma, h_target=m.h_target,
                          cg_iters=m.cg_iters, cg_tol=m.cg_tol)
        self.vdw = TkatchenkoSchefflerDispersion(
            Z_SUPPORTED, cutoff=m.vdw_cutoff, sR=m.vdw_damping_sR,
            d=m.vdw_damping_d) if m.use_vdw else None
        self._init_heads()

    def _init_heads(self):
        with torch.no_grad():
            # small random last layers: exact zero would block the first
            # gradient to the hidden layers for one step
            self.chi_head[-1].weight.normal_(0.0, 1e-3)
            self.chi_head[-1].bias.zero_()
            self.j_head[-1].weight.normal_(0.0, 1e-3)
            self.j_head[-1].bias.fill_(1.0)          # softplus(1)+0.5 ~ 1.81 eV
            self.e_head[-1].weight.normal_(0.0, 1e-3)
            self.e_head[-1].bias.zero_()
            # volume ratio ~ 1 (free atom) at initialisation: softplus(0.33) ~ 0.9
            if self.v_heads is not None:
                for head in self.v_heads:
                    head[-1].weight.normal_(0.0, 1e-3)
                    head[-1].bias.fill_(0.33)

    # ------------------------------------------------------------------ #
    def fit_scaler(self, frames, indices, max_frames: Optional[int] = None):
        idx = list(indices)[:max_frames] if max_frames else list(indices)
        feats = []
        with torch.no_grad():
            for i in idx:
                f = frames[i]
                cont, _ = build_descriptors(f.pos, f.cell, f.nl_rad, f.nl_ang,
                                            f.zi, self.dcfg)
                feats.append(cont)
        self.scaler.fit(torch.cat(feats, dim=0))

    def set_element_reference(self, eps: torch.Tensor):
        with torch.no_grad():
            self.eps.copy_(eps.to(self.eps.device, self.eps.dtype))

    # ------------------------------------------------------------------ #
    def forward(self, pos, cell, zi, nl_rad, nl_ang, batch, G: int,
                compute_forces: bool = True, create_graph: bool = False,
                energy_only_grad: bool = False):
        d = self.dcfg
        m = self.mcfg
        if compute_forces and not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
        cont, onehot = build_descriptors(pos, cell, nl_rad, nl_ang, zi, d)
        feat = torch.cat([self.scaler(cont), onehot], dim=1)
        h = self.trunk(feat)
        chi = self.chi_head(h).squeeze(-1)
        J = F.softplus(self.j_head(h)).squeeze(-1) + m.j_min
        if self.chi_elem is not None:
            chi = chi + onehot @ self.chi_elem
        if self.j_elem is not None:
            J = J + onehot @ self.j_elem

        # ---- global QEq solve (no graph through the solver) --------------
        with torch.no_grad():
            st_detached = self.qeq.coulomb.build_stencil(
                pos.detach(), cell.detach(), batch, G)
            sol = self.qeq.solve_q(pos.detach(), cell.detach(), chi.detach(),
                                   J.detach(), batch, G, stencil=st_detached)
            q = sol.q.detach()

        qr = q.requires_grad_(True) if compute_forces else q
        e_in = torch.cat([h, (qr / m.q_scale).unsqueeze(-1)], dim=1)
        e_atom = self.e_head(e_in).squeeze(-1)
        E_s = e_atom.sum()
        # one differentiable stencil, reused for the energy and the adjoint term
        st_g = self.qeq.coulomb.build_stencil(pos, cell, batch, G)
        Ecq = self.qeq.coulomb.apply_stencil(st_g, q)
        # Gaussian self term, paper Eq. 12: E_ii = 1/(sigma sqrt(pi)).
        self_diag = (self.qeq.coulomb.self_energy_diag()
                     if m.include_self_energy else 0.0)
        E_self = 0.5 * self_diag * (q * q).sum() if m.include_self_energy \
            else pos.new_zeros(())
        E_elec = 0.5 * (q * Ecq).sum() + E_self
        # eps[zi] via one-hot matmul: indexing 4 parameters with 5,184 atomics
        # serialises on the GPU (measured 38 ms at 5,184 atoms)
        eps_atom = onehot @ self.eps
        E_eps = eps_atom.sum()

        # ---- Tkatchenko-Scheffler dispersion (paper Eq. 5, third term) ----
        # Element-specific Hirshfeld volume ratios, then pairwise C6/r^6.
        # E_vdw depends on positions both through the pair distances and
        # through V (hence h), so it is a plain differentiable term and enters
        # the force surrogate unchanged; it does not depend on the charges, so
        # the QEq adjoint construction is untouched.
        if self.vdw is not None:
            v_list = [head(h) for head in self.v_heads]
            v_stack = torch.cat(v_list, dim=-1)               # [N, N_ELEM]
            v_ratio = F.softplus((v_stack * onehot).sum(dim=-1))   # [N]
            E_vdw, atom_vdw = self.vdw.energy(pos, cell, batch, zi, v_ratio)
        else:
            v_ratio = torch.ones(pos.shape[0], device=pos.device, dtype=pos.dtype)
            E_vdw = pos.new_zeros(())
            atom_vdw = pos.new_zeros(pos.shape[0])

        E_pred = E_s + E_elec + E_eps + E_vdw
        atom_pred = (e_atom + eps_atom + 0.5 * q * Ecq
                     + 0.5 * self_diag * q * q + atom_vdw)
        E_graph_pred = torch.zeros(G, device=pos.device, dtype=pos.dtype)
        E_graph_pred.index_add_(0, batch, atom_pred)

        out = {"E": E_pred, "E_graph": E_graph_pred, "q": q, "chi": chi, "J": J,
               "E_s": E_s.detach(), "E_elec": E_elec.detach(),
               "E_eps": E_eps.detach(), "E_vdw": E_vdw.detach(),
               "v_ratio": v_ratio.detach(), "cg_iters": sol.iterations,
               "cg_residual": sol.residual,
               # charge-supervision / two-stage training needs the detached
               # operator and the graph inputs; they are not model state.
               "_stencil": st_detached, "_pos": pos, "_cell": cell,
               "_batch": batch, "_G": G}
        if not compute_forces:
            return out

        # ---- force trick --------------------------------------------------
        dEs_dq = torch.autograd.grad(E_s, qr, retain_graph=True,
                                     create_graph=create_graph)[0]
        r = dEs_dq.detach() + Ecq.detach()
        if m.include_self_energy:
            r = r + self_diag * q.detach()
        with torch.no_grad():
            y, it_y, res_y = self.qeq.solve_adjoint(
                pos.detach(), cell.detach(), r, J.detach(), batch, G,
                stencil=st_detached)
            y = y.detach()

        E_surr = (E_s + 0.5 * (q * Ecq).sum() + (y * Ecq).sum()
                  + (y * q * J).sum() + (y * chi).sum() + E_eps + E_vdw)
        if m.include_self_energy:
            E_surr = E_surr + 0.5 * self_diag * (q * q).sum() \
                + self_diag * (y * q).sum()
        F_pred = -torch.autograd.grad(E_surr, pos, create_graph=create_graph)[0]

        atom_surr = (e_atom + 0.5 * q * Ecq + y * Ecq
                     + y * q * J + y * chi + eps_atom + atom_vdw)
        if m.include_self_energy:
            atom_surr = atom_surr + 0.5 * self_diag * q * q \
                + self_diag * y * q
        E_graph_surr = torch.zeros(G, device=pos.device, dtype=pos.dtype)
        E_graph_surr.index_add_(0, batch, atom_surr)
        E_graph_train = (E_graph_pred - E_graph_surr).detach() + E_graph_surr

        out["F"] = F_pred
        out["E_train"] = (E_pred - E_surr).detach() + E_surr
        out["E_graph_train"] = E_graph_train
        out["y"] = y
        out["adj_iters"] = it_y
        out["adj_residual"] = res_y
        return out

    # ------------------------------------------------------------------ #
    def predict(self, batch: dict, compute_forces: bool = True,
                create_graph: bool = False):
        return self.forward(batch["pos"], batch["cell"], batch["zi"],
                            batch["nl_rad"], batch["nl_ang"], batch["batch"],
                            batch["G"], compute_forces=compute_forces,
                            create_graph=create_graph)

    def energy_and_forces(self, batch: dict):
        """Inference wrapper (still uses autograd for the force trick)."""
        out = self.predict(batch, compute_forces=True, create_graph=False)
        return out["E"], out["F"], out

    def charge_surrogate(self, out: dict, q_ref: torch.Tensor):
        """Differentiable charge prediction for chi/J supervision.

        The forward QEq solve is detached.  On the sum-zero subspace
        q = -A^{-1} chi, so with r = dL/dq,

            dL/dchi = -A^{-1} r,   dL/dJ_i = -q_i (A^{-1} r)_i.

        ``solve_adjoint`` returns y = -A^{-1} r (it solves A y = -r), hence

            dL/dchi = y,          dL/dJ_i = y_i q_i,

        and the surrogate

            q_surr = q + y * (chi - chi.detach())
                       + (y*q) * (J - J.detach())

        has the exact first-order gradient of L(q) while its value is q.
        """
        q = out["q"]
        chi = out["chi"]
        J = out["J"]
        r = 2.0 * (q - q_ref) / max(q_ref.numel(), 1)
        with torch.no_grad():
            y, _, _ = self.qeq.solve_adjoint(
                out["_pos"].detach(), out["_cell"].detach(), r.detach(),
                J.detach(), out["_batch"], out["_G"],
                stencil=out["_stencil"])
            y = y.detach()
        return (q + y * (chi - chi.detach())
                + (y * q) * (J - J.detach()))
