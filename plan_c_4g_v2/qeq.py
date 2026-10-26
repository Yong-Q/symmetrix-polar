"""Global QEq charge equilibration with an FFT/PME Coulomb operator.

The Coulomb operator is the periodic Gaussian-smeared interaction

    phi(r) = erf(r / (sqrt(2) sigma)) / r

represented on a uniform grid: charges are spread with a cubic cardinal
B-spline, transformed with FFT, multiplied by the Fourier-space kernel and the
B-spline deconvolution, transformed back and interpolated at the atoms.  The
Gaussian width regularises the short range and the FFT captures the 1/r tail
on the grid.  The operator is symmetric by construction.

QEq solves, on the sum-zero subspace,

    (E_coul + diag(J)) q + chi = -mu * 1 ,      1^T q = 0

with matrix-free preconditioned conjugate gradients.  The preconditioner is the
inverse of the translation-invariant part (E_coul + mean(J)) and is diagonal in
the FFT basis.  ``solve_adjoint`` solves the same system for the adjoint
variable used by the force trick (RuNNer 2.0, SI Eq. 34).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #
def _next_fast(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n (FFT friendly)."""
    n = max(int(n), 1)
    best = None
    p2 = 1
    while p2 < 4 * n:
        p3 = p2
        while p3 < 4 * n:
            p5 = p3
            while p5 < 4 * n:
                if p5 >= n and (best is None or p5 < best):
                    best = p5
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return int(best)


def _grid_shape(cell: torch.Tensor, h_target: float):
    cell = cell.reshape(3, 3)
    inv = torch.linalg.inv(cell)
    widths = 1.0 / torch.linalg.norm(inv, dim=0)
    return tuple(_next_fast(int(math.ceil(float(w) / h_target))) for w in widths)


def _bspline_weights(u: torch.Tensor):
    """Cubic cardinal B-spline weights for offsets 0..3."""
    f = torch.floor(u)
    t = u - f
    i0 = f.to(torch.long) - 1
    t2 = t * t
    t3 = t2 * t
    w0 = (1.0 - t) ** 3 / 6.0
    w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    w3 = t3 / 6.0
    return i0, torch.stack([w0, w1, w2, w3], dim=-1)


class _ScatterFn(torch.autograd.Function):
    """Sparse spread with a hand-written backward (index_select, no index_put)."""

    @staticmethod
    def forward(ctx, q, w, idx, ngrid, G, n):
        dens = torch.zeros(G * ngrid, device=q.device, dtype=q.dtype)
        dens.index_add_(0, idx.reshape(-1), (q.unsqueeze(1) * w).reshape(-1))
        ctx.save_for_backward(q, w, idx)
        ctx.ngrid, ctx.G, ctx.n = ngrid, G, n
        return dens.reshape((G,) + n)

    @staticmethod
    def backward(ctx, grad_dens):
        q, w, idx = ctx.saved_tensors
        g = grad_dens.reshape(-1)
        gsel = g[idx]
        grad_q = (gsel * w).sum(dim=1)
        grad_w = gsel * q.unsqueeze(1) if w.requires_grad else None
        return grad_q, grad_w, None, None, None, None


class _GatherFn(torch.autograd.Function):
    """Sparse gather with a hand-written backward (index_add, no index_put)."""

    @staticmethod
    def forward(ctx, phi, w, idx):
        out = (phi.reshape(-1)[idx] * w).sum(dim=1)
        ctx.save_for_backward(phi, w, idx)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        phi, w, idx = ctx.saved_tensors
        go = grad_out.unsqueeze(1)
        grad_phi = torch.zeros_like(phi).reshape(-1)
        grad_phi.index_add_(0, idx.reshape(-1), (go * w).reshape(-1))
        grad_w = phi.reshape(-1)[idx] * go if w.requires_grad else None
        return grad_phi.view_as(phi), grad_w, None


@dataclass
class Stencil:
    n: tuple
    G: int
    idx: torch.Tensor       # [N,64] long
    w: torch.Tensor         # [N,64] (may carry a position graph)
    ngrid: int

    def scatter(self, q: torch.Tensor) -> torch.Tensor:
        return _ScatterFn.apply(q, self.w, self.idx, self.ngrid, self.G, self.n)

    def gather(self, phi: torch.Tensor) -> torch.Tensor:
        return _GatherFn.apply(phi, self.w, self.idx)


class PMECoulomb(torch.nn.Module):
    """Periodic Gaussian-smeared Coulomb operator on a B-spline grid."""

    def __init__(self, sigma: float = 0.6, h_target: float = 0.8, order: int = 4,
                 eps: float = 1e-12):
        super().__init__()
        self.sigma = float(sigma)
        self.h_target = float(h_target)
        self.order = int(order)
        self.eps = float(eps)
        self._plans = {}

    # ------------------------------------------------------------------ #
    def _plan(self, cell: torch.Tensor, device, dtype):
        cell = cell.reshape(3, 3)
        key = (tuple(round(float(x), 8) for x in cell.reshape(-1).tolist()),
               str(device), str(dtype))
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        n = _grid_shape(cell, self.h_target)
        n1, n2, n3 = n
        inv = torch.linalg.inv(cell)
        m1 = torch.fft.fftfreq(n1, device=device, dtype=dtype) * n1
        m2 = torch.fft.fftfreq(n2, device=device, dtype=dtype) * n2
        m3 = torch.fft.rfftfreq(n3, device=device, dtype=dtype) * n3
        b1, b2, b3 = inv[0], inv[1], inv[2]
        k1 = 2.0 * math.pi * m1
        k2_ = 2.0 * math.pi * m2
        k3 = 2.0 * math.pi * m3
        kx = k1[:, None, None] * b1[0] + k2_[None, :, None] * b2[0] + k3[None, None, :] * b3[0]
        ky = k1[:, None, None] * b1[1] + k2_[None, :, None] * b2[1] + k3[None, None, :] * b3[1]
        kz = k1[:, None, None] * b1[2] + k2_[None, :, None] * b2[2] + k3[None, None, :] * b3[2]
        k2sq = kx * kx + ky * ky + kz * kz
        sx = torch.sinc(m1 / n1).clamp(min=1e-8) ** self.order
        sy = torch.sinc(m2 / n2).clamp(min=1e-8) ** self.order
        sz = torch.sinc(m3 / n3).clamp(min=1e-8) ** self.order
        bfac = sx[:, None, None] * sy[None, :, None] * sz[None, None, :]
        inv_k2 = torch.zeros_like(k2sq)
        mask = k2sq > self.eps
        inv_k2[mask] = 1.0 / k2sq[mask]
        gauss = torch.exp(-0.5 * self.sigma ** 2 * k2sq)
        volume = float(torch.abs(torch.linalg.det(cell)))
        dV = volume / (n1 * n2 * n3)
        k_full = 4.0 * math.pi * inv_k2 * gauss
        # effective operator multiplier is k_full/dV; spread/gather contribute
        # two B-spline factors.
        influence = k_full / (dV * bfac * bfac)
        influence[0, 0, 0] = 0.0
        k_over_dv = k_full / dV
        k_over_dv[0, 0, 0] = 0.0
        plan = {"n": n, "influence": influence.contiguous(),
                "k_over_dv": k_over_dv.contiguous(),
                "bfac2": (bfac * bfac).contiguous(),
                "self": float(1.0 / (self.sigma * math.sqrt(math.pi))),
                "dV": dV}
        self._plans[key] = plan
        return plan

    # ------------------------------------------------------------------ #
    def build_stencil(self, pos, cell, batch, G) -> Stencil:
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        plan = self._plan(cell[0], pos.device, pos.dtype)
        n = plan["n"]
        n1, n2, n3 = n
        # homogeneous cells (the normal path): one inverse for everything
        if bool(torch.allclose(cell, cell[0].expand_as(cell))):
            inv = torch.linalg.inv(cell[0])
            frac = pos @ inv
        else:
            inv = torch.linalg.inv(cell[batch])
            frac = torch.einsum("ni,nij->nj", pos, inv)
        frac = frac - torch.floor(frac)
        u = frac * torch.tensor(n, device=pos.device, dtype=pos.dtype)
        i0, w1d = _bspline_weights(u)
        ar = torch.arange(4, device=pos.device)
        ia = (i0[:, 0:1] + ar) % n1
        ib = (i0[:, 1:2] + ar) % n2
        ic = (i0[:, 2:3] + ar) % n3
        idx = (((batch[:, None, None, None] * n1 + ia[:, :, None, None]) * n2
                + ib[:, None, :, None]) * n3 + ic[:, None, None, :]).reshape(-1, 64)
        w = (w1d[:, 0, :, None, None] * w1d[:, 1, None, :, None]
             * w1d[:, 2, None, None, :]).reshape(-1, 64)
        return Stencil(n=n, G=G, idx=idx, w=w, ngrid=n1 * n2 * n3)

    # ------------------------------------------------------------------ #
    def apply_stencil(self, st: Stencil, q: torch.Tensor,
                      influence: Optional[torch.Tensor] = None) -> torch.Tensor:
        if influence is None:
            plan = self._plans_by_n(st)
            influence = plan["influence"]
        dens = st.scatter(q)
        rho = torch.fft.rfftn(dens, dim=(1, 2, 3))
        phi = torch.fft.irfftn(rho * influence.unsqueeze(0), s=st.n, dim=(1, 2, 3))
        return st.gather(phi)

    def _plans_by_n(self, st: Stencil):
        for plan in self._plans.values():
            if plan["n"] == st.n:
                return plan
        raise KeyError("no PME plan for this grid")

    def apply(self, pos, cell, q, batch, G):
        st = self.build_stencil(pos, cell, batch, G)
        return self.apply_stencil(st, q)

    def precondition(self, st: Stencil, r: torch.Tensor, Jmean: float) -> torch.Tensor:
        plan = self._plans_by_n(st)
        pre = 1.0 / (plan["bfac2"] * (plan["k_over_dv"] + Jmean))
        pre[0, 0, 0] = 0.0
        dens = st.scatter(r)
        rho = torch.fft.rfftn(dens, dim=(1, 2, 3))
        out = torch.fft.irfftn(rho * pre.unsqueeze(0), s=st.n, dim=(1, 2, 3))
        return st.gather(out)

    def self_energy_diag(self):
        return 1.0 / (self.sigma * math.sqrt(math.pi))


# --------------------------------------------------------------------------- #
# constrained preconditioned CG
# --------------------------------------------------------------------------- #
@dataclass
class QEqSolution:
    q: torch.Tensor
    Aq: torch.Tensor
    iterations: int
    residual: float


def _project(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean()


def _make_proj(batched):
    if batched is None:
        return _project
    nG = int(batched.max().item()) + 1
    counts = torch.bincount(batched, minlength=nG).to(torch.float64)

    def proj(v):
        mean = torch.zeros(nG, device=v.device, dtype=v.dtype)
        mean.index_add_(0, batched, v)
        mean = mean / counts.to(v.dtype)
        return v - mean[batched]
    return proj


def solve_constrained(Aop, rhs: torch.Tensor, diag,
                      max_iter: int = 30, tol: float = 1e-10,
                      batched: Optional[torch.Tensor] = None,
                      tol_rel: float = 1e-11, reset_every: int = 8,
                      precond: Optional[Callable] = None):
    """Solve A x = rhs with 1^T x = 0 by preconditioned CG.

    ``diag`` is a Jacobi vector or a callable z -> M^{-1} z.  The recursive
    residual can drift below the true residual in FP64, so the true residual is
    recomputed every ``reset_every`` iterations and at the end.
    """
    proj = _make_proj(batched)
    rhs_p = proj(rhs)
    rhs_norm = float(torch.sqrt((rhs_p * rhs_p).sum()))
    target = max(float(tol), float(tol_rel) * rhs_norm)
    x = torch.zeros_like(rhs_p)
    r = rhs_p.clone()
    if precond is None:
        z = proj(r / diag)
    else:
        z = proj(precond(r))
    p = z.clone()
    rz = (r * z).sum()
    it = 0
    residual = rhs_norm
    for it in range(1, max_iter + 1):
        Ap = proj(Aop(p))
        denom = (p * Ap).sum()
        if float(denom.abs()) < 1e-30:
            break
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if it % reset_every == 0 or it == max_iter:
            r = proj(rhs_p - proj(Aop(x)))
            residual = float(torch.sqrt((r * r).sum()))
            if residual <= target:
                break
            z = proj(precond(r)) if precond is not None else proj(r / diag)
            p = z.clone()
            rz = (r * z).sum()
        else:
            z = proj(precond(r)) if precond is not None else proj(r / diag)
            rz_new = (r * z).sum()
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
    r = proj(rhs_p - proj(Aop(x)))
    residual = float(torch.sqrt((r * r).sum()))
    return proj(x), it, residual


class QEqPME(torch.nn.Module):
    """QEq driver: forward charges + adjoint for the force trick."""

    def __init__(self, sigma: float = 0.6, h_target: float = 0.8,
                 cg_iters: int = 60, cg_tol: float = 1e-9):
        super().__init__()
        self.coulomb = PMECoulomb(sigma=sigma, h_target=h_target)
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)

    def _diag(self, J):
        return J + self.coulomb.self_energy_diag()

    def _Aop(self, st, J):
        # Paper Eq. 12/14: A = E + diag(J), and E contains the Gaussian
        # self term 1/(sigma sqrt(pi)) on its diagonal.  The self term was
        # previously only in the preconditioner, so the CG solve and the
        # reported energy used inconsistent operators.
        self_diag = self.coulomb.self_energy_diag()
        def op(x):
            return self.coulomb.apply_stencil(st, x) + (J + self_diag) * x
        return op

    def solve_q(self, pos, cell, chi, J, batch, G, stencil: Optional[Stencil] = None):
        if stencil is None:
            stencil = self.coulomb.build_stencil(pos, cell, batch, G)
        proj = _make_proj(batch)
        rhs = -proj(chi)
        Aop = self._Aop(stencil, J)
        q, it, res = solve_constrained(Aop, rhs, self._diag(J),
                                       max_iter=self.cg_iters, tol=self.cg_tol,
                                       batched=batch)
        Aq = Aop(q)
        return QEqSolution(q=q, Aq=Aq, iterations=it, residual=res)

    def solve_adjoint(self, pos, cell, r, J, batch, G,
                      stencil: Optional[Stencil] = None):
        if stencil is None:
            stencil = self.coulomb.build_stencil(pos, cell, batch, G)
        proj = _make_proj(batch)
        rhs = -proj(r)
        Aop = self._Aop(stencil, J)
        y, it, res = solve_constrained(Aop, rhs, self._diag(J),
                                       max_iter=self.cg_iters, tol=self.cg_tol,
                                       batched=batch)
        return y, it, res
