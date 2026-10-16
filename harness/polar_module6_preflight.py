#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑥(coulomb_energy = k 空间 GTO-Ewald 能量) numpy 预检 —— 编译前公式锁定。
(2026-10-15, 智能体 F)

目标: 对 4 个 scf 夹具盒(NH3/CH4/N2/CO2 × 32 分子盒, f0 帧), 用**纯 numpy**
逐式重算 graph_longrange v0.4.0 `GTOElectrostaticEnergy` 的能量, 与
`bench/polar_fixtures/index.json` 的 `electrostatic_energy` 目标值对照。
若吻合(<1e-8 相对)说明公式/序列化/密度容器语义已锁定, 可直接进引擎补丁;
若不吻合, 逐中间量报告差在哪一步(不凑)。

--- 输入 ---------------------------------------------------------------------
1) 序列化表: models/mace-polar/polar_symmetrix_probe.json
   polar_blocks.coulomb_energy(键树先行 dump; extractor=extract_polar_blocks.py)
2) 几何/k 空间: bench/polar_fixtures/geo3/{mol}-f0-pregeom.npz + -kvecs.npz
   (bench/polar_module3_refdump.py 产物; k_vectors/k_norm2/k0_mask/cosines/
    sines/density_basis_fs_re/im/volume_per_k/node_positions/volumes)
3) 密度: bench/polar_fixtures/scf/{mol}-f0.npz 的 final_spin_charge_density (N,2,4)
   + 交叉核对 bench/polar_fixtures/scf_steps/{mol}-step{0,1,2}.npz 与
   bench/polar_fixtures/{mol}-f0.npz 的 density_coefficients (N,4, 模型自己的
   charge_density_mul_ir = coulomb 的实际 source_feats; 最强的容器语义证据)
4) 目标: bench/polar_fixtures/index.json[["{mol}-f0"]["electrostatic_energy"]]

--- 公式锚点(逐式对照; 全部 file:line) ---------------------------------------
[E1] graph_longrange/energy.py::GTOElectrostaticEnergy.forward L70-99
     → pbc 任一为真 → _pbc_energy_batch (本盒 (1,3) 全 True, features.py 同判据)
[E2] energy.py::_pbc_energy_batch L113-154:
       inner = k·r; cosines/sines = cos/sin·mask                      (L125-129)
       density_basis_fs = self.density_basis(k, k², k0)               (L131)
       volume_per_k = volume[k_vector_batch]                          (L132)
       density = assemble_fourier_series_batch(...)                   (L133-139)
       k_factor_coulomb[k=0]=0 否则 1/k²                              (L141-143)
       potential = apply_coulomb_kernel_batch(...)                    (L144-148)
       energy = energy_product_batch(...)                             (L149-154)
[E3] energy.py::energy_product_batch L17-30:
       per_k = 2·Σ_c density·potential; E = 0.5·V·Σ_k per_k/(2π)^6
[E4] energy.py L156-166: include_self_interaction=False 时再减 0.5·Σ_n <q_n, self_field_n>。
     本模型序列化值 = True(extract 实值) ⇒ 该项**不执行**。脚本把该量作为
     "反事实诊断"落盘(若误加会差多少)。
[E5] energy.py L168-192: pbc 修正(monopole_dipole/slab), 仅对 molecule(TFF)/
     slab(TTF) 图生效 (slabs.py L185-256); 本盒 TTT ⇒ 两项均被 where 置 0。
[F1] features.py::assemble_fourier_series_batch L28-67:
       coeff_cos = C·q; coeff_sin = S·q                              (L52-53)
       rho_real = Br·coeff_cos + Bi·coeff_sin                         (L58-60)
       rho_imag = Bi·coeff_cos − Br·coeff_sin                         (L61-63)
       density = (2π)³·rho/volume_per_k                               (L66)
[F2] features.py::apply_coulomb_kernel_batch L70-83: potential = density·kfac·FIELD_CONSTANT
[F3] utils.py L12: FIELD_CONSTANT = 1/(5.526349406e-3)
[G1] gto_utils.py::RadialIntegralDirect L33-48(max_l=1):
       exp = exp(−½k²σ²); out[...,0] = pref0·exp; out[...,1] = pref1·k·exp
[G2] gto_utils.py::GTOBasis L145-178: k_moduli=sqrt(clamp_min(k²,0)), k0→0;
       fnlk = radial(k)·cl_scale; 按 expanded_l_indices 展开; ×Y_lm(k 按
       permute_indices=[1,2,0] 置换) × 相位(实/虚) → fourier_coefficients
[G3] e3nn SphericalHarmonics(max_l=1, normalize=True, normalization="integral"):
       Y00 = 1/√(4π) = 0.28209479177387814; Y1 = √(3/4π)·k̂ = 0.48860251190291992·k̂
       (normalize=True = 输入向量归一化; f1·Y1 = pref1·cl1·y1·exp·k 解析等价于
       引擎 v14 写法, k=0 处自动 0 —— 见 §16/模块③门)
[G4] cl_scale = 1/(√(4π/(2l+1))·2^((2l+1)/2)·Γ((2l+3)/2)·σ^(2l+3))(multipoles 归一)
     注: 序列化值是**checkpoint 内 f32 缓冲**的 f64 表示, 脚本直接用序列化值
     (不做解析重算; 解析重算偏差作为信息量报告)。
[D1] extensions.py L887-924: charge_density = spin_charge_density.sum(dim=1)(自旋通道求和,
     与 charges_to_mul_ir 无操作: cueq 未启用时 TransposeIrrepsLayoutWrapper 返回 None,
     wrapper_ops.py L413-434 + forward L641/L889-893) → coulomb_energy(source_feats=(N,4))
[D2] extensions.py L734-736: spin_charge_density = view(N,2,-1) → (N,2,4), 平坦序
     [sα,xα,yα,zα,sβ,xβ,yβ,zβ](§13 布局裁定)。

用法: cd mlip && CUDA_VISIBLE_DEVICES=1 /workspace/caches/conda/envs/polar/bin/python \
        bench/polar_module6_preflight.py 2>&1 | tee logs/polar_module6_preflight.log
(纯 numpy; 不 import torch; 无 GPU 需求)
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
PROBE_JSON = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
FIXDIR = f"{ROOT}/bench/polar_fixtures"
SCFDIR = f"{ROOT}/bench/polar_fixtures/scf"
STEPDIR = f"{ROOT}/bench/polar_fixtures/scf_steps"
GEODIR = f"{ROOT}/bench/polar_fixtures/geo3"
MOLS = ["nh3", "ch4", "n2", "co2"]

FIELD_CONSTANT = 1.0 / (5.526349406e-3)      # utils.py L12
Y0 = 0.28209479177387814                     # 1/sqrt(4*pi), e3nn integral 归一
Y1 = 0.48860251190291992                     # sqrt(3/(4*pi))

TWO_PI = 2.0 * np.pi
ENERGY_TARGET_REL = 1e-8                     # 预检判据(任务 A)


# ------------------------------------------------------------------ 工具 ----
def dump_key_tree(name, node, prefix="  ", depth=0):
    """打印 JSON 子树的键/形状(叶值只给标量或摘要)。"""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict):
                if {"shape", "values"} <= set(value.keys()):
                    print(f"{prefix}{key}: tensor shape={value['shape']} "
                          f"n={len(value['values'])}", flush=True)
                else:
                    print(f"{prefix}{key}: dict", flush=True)
                    if depth < 3:
                        dump_key_tree(name, value, prefix + "  ", depth + 1)
            elif isinstance(value, list):
                head = value[0] if value else None
                if isinstance(head, dict):
                    print(f"{prefix}{key}: list[{len(value)}]", flush=True)
                    if depth < 3:
                        dump_key_tree(name, head, prefix + "  ", depth + 1)
                else:
                    print(f"{prefix}{key}: list[{len(value)}] "
                          f"{value[:6]}", flush=True)
            else:
                print(f"{prefix}{key}: {type(value).__name__} = "
                      f"{value if isinstance(value, (int, float, bool, type(None))) else '…'}"
                      f"   [{name}]", flush=True)
    else:
        print(f"{prefix}{node}", flush=True)


def tensor_values(block):
    return np.asarray(block["values"], dtype=np.float64)


# ------------------------------------------------------------- 公式复现 ----
def rebuild_density_basis(ce, k_vectors, k_norm2, k0_mask):
    """从序列化常数重建 coulomb_energy.density_basis 在该 k 网格上的傅里叶表。

    对应 GTOBasis.forward(gto_utils.py L170-178) + RadialIntegralDirect(L33-48)。
    返回 (re, im, diag); re/im: (Nk,4)。
    """
    db = ce["density_basis"]
    cl = tensor_values(db["cl_scale"])                       # [1,2]→[2]
    exp_idx = np.asarray(db["expanded_l_indices"]["values"], dtype=np.int64)
    rph = tensor_values(db["real_phase_factors"])
    iph = tensor_values(db["imag_phase_factors"])
    perm = np.asarray(db["permute_indices"]["values"], dtype=np.int64)
    sigma2 = float(db["radial_spline"]["sigma2"]["values"][0])
    pref0 = float(db["radial_spline"]["pref0"]["values"][0])
    pref1 = float(db["radial_spline"]["pref1"]["values"][0])

    # gto_utils.py L145-150: k_moduli = sqrt(clamp_min(k²,0)); k=0 → 0
    kmod = np.sqrt(np.clip(k_norm2, 0.0, None))
    kmod = np.where(k0_mask > 0.0, 0.0, kmod)
    exp_d = np.exp(-0.5 * kmod ** 2 * sigma2)
    f0 = pref0 * exp_d                                       # G1 out[...,0]
    f1 = pref1 * kmod * exp_d                                # G1 out[...,1]
    fn = np.stack([f0, f1], axis=-1)                         # (Nk,2) 按 l
    expanded = fn[:, exp_idx] * cl[exp_idx]                  # (Nk,4) fnlk·cl_scale

    # Y_lm(e3nn, max_l=1, normalize=True): 输入 k 先按 permute=[1,2,0] 置换
    v = k_vectors[:, perm]                                   # (ky,kz,kx)
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    unit = np.zeros_like(v)
    nz = nrm[:, 0] > 0.0
    unit[nz] = v[nz] / nrm[nz]
    ylm = np.zeros((k_vectors.shape[0], 4))
    ylm[:, 0] = Y0
    ylm[:, 1:] = Y1 * unit

    xnlk = expanded * ylm
    re = xnlk * rph[None, :]
    im = xnlk * iph[None, :]
    diag = {"sigma2": sigma2, "pref0": pref0, "pref1": pref1,
            "cl_scale": cl.tolist(), "permute": perm.tolist()}
    return re, im, diag


def assemble_density(source_feats, cosines, sines, basis_re, basis_im, volume):
    """features.py::assemble_fourier_series_batch L28-67 (单 σ, 单图)。"""
    coeff_cos = cosines @ source_feats                       # (Nk,4)
    coeff_sin = sines @ source_feats
    rho_real = (basis_re * coeff_cos).sum(-1) + (basis_im * coeff_sin).sum(-1)
    rho_imag = (basis_im * coeff_cos).sum(-1) - (basis_re * coeff_sin).sum(-1)
    rho = np.stack([rho_real, rho_imag], axis=-1)
    rho *= TWO_PI ** 3 / volume
    return rho, coeff_cos, coeff_sin


def self_interaction_energy(ce, source_feats):
    """GTOSelfInteractionBlock (gto_utils.py L283-302) + energy.py L156-166。
    仅当 include_self_interaction=False 时参考才会执行(本模型 True → 反事实诊断)。"""
    blk = ce["self_interaction_terms"]
    ov = tensor_values(blk["overlap_constants"])
    sel = np.asarray(blk["select_indices"]["values"], dtype=np.int64)
    self_fields = ov[None, :] * source_feats[:, sel]
    node_energies = (source_feats * self_fields).sum(-1)
    return float(node_energies.sum()), self_fields


def pbc_correction_energy(ce, source_feats, positions, volume, pbc):
    """energy.py L168-192: molecule(TFF)/slab(TTF) 修正; TTT 时返回 0。
    按 slabs.py L185-256 (MonopoleDipoleCorrectionBlock) 与 L59-69(slab) 复现。"""
    is_molecule = bool(np.all(~pbc))
    is_slab = bool(np.all(np.logical_xor([False, False, True], pbc)))
    dens_l = int(ce["density_basis"]["max_l"])
    const = FIELD_CONSTANT / (4.0 * np.pi)                   # slabs.py L205
    mol_e = 0.0
    if is_molecule:
        q = source_feats[:, 0].sum()
        q_r = positions * source_feats[:, 0:1]
        total_dipole = q_r.sum(0)
        r2 = (positions ** 2).sum(-1)
        quad = float((r2 * source_feats[:, 0]).sum())
        if dens_l > 0:                                       # slabs.py L229-238
            local_dipoles = source_feats[:, [3, 1, 2]]
            total_dipole = total_dipole + local_dipoles.sum(0)
            quad += 2.0 * float((positions * local_dipoles).sum())
        Ls = volume ** 0.3333
        mol_e = 0.5 * 2.837297 * const * q * q / Ls \
            + 2.0 * const * np.pi * float((total_dipole ** 2).sum()) / (3 * volume) \
            - 2.0 * const * np.pi * q * quad / (3 * volume)
    slab_e = 0.0
    if is_slab:
        charges = source_feats[:, 0]
        dz = float((positions[:, 2] * charges).sum())
        if dens_l > 0:
            dz += float(source_feats[:, 1:4].sum(0)[1])      # slabs.py L52-55
        A = FIELD_CONSTANT / (4 * np.pi)
        slab_e = A * 2 * np.pi * dz * dz / volume
    return mol_e, slab_e, is_molecule, is_slab


# ------------------------------------------------------------------ 主流程 --
def main():
    print("=" * 78)
    print("P3 模块⑥ preflight: coulomb_energy (k 空间 GTO-Ewald) numpy 复现")
    print("=" * 78, flush=True)

    # ---- 0) 序列化键树 dump ---------------------------------------------
    print("\n[0] polar_blocks.coulomb_energy 键树 (polar_symmetrix_probe.json)", flush=True)
    with open(PROBE_JSON) as fh:
        full = json.load(fh)
    ce = full["polar_blocks"]["coulomb_energy"]
    dump_key_tree("coulomb_energy", ce)

    kspace_cutoff = float(ce["kspace_cutoff"])
    include_si = bool(ce["include_self_interaction"])
    include_pbc = bool(ce["include_pbc_corrections"])
    print(f"\n  kind={ce['kind']}  kspace_cutoff={kspace_cutoff!r}  "
          f"include_self_interaction={include_si}  include_pbc_corrections={include_pbc}",
          flush=True)

    target_index = json.load(open(f"{FIXDIR}/index.json"))

    rows = []
    for mol in MOLS:
        name = f"{mol}-f0"
        print("\n" + "-" * 78)
        print(f"[{name}]", flush=True)

        # ---- 1) 几何/k 空间夹具(in) -------------------------------------
        geo = np.load(f"{GEODIR}/{name}-pregeom.npz")
        kvec_dump = np.load(f"{GEODIR}/{name}-kvecs.npz")
        k_vectors = geo["k_vectors"]
        k_norm2 = geo["k_norm2"]
        k0_mask = geo["k0_mask"]
        positions = geo["node_positions"]
        volume = float(geo["volumes"][0])
        Nk, N = k_vectors.shape[0], positions.shape[0]

        assert np.array_equal(k_vectors, kvec_dump["k_vectors"]), "kvec dump mismatch"
        assert np.allclose(geo["cosines"], np.cos(k_vectors @ positions.T)) and \
            np.allclose(geo["sines"], np.sin(k_vectors @ positions.T))
        assert float((k_norm2 * (k0_mask <= 0)).max()) <= kspace_cutoff ** 2 * (1 + 1e-12)
        assert k0_mask[0] == 1.0 and k_norm2[0] == 0.0
        print(f"  [in] Nk={Nk} N={N} volume={volume:.6f}  cutoff={kspace_cutoff:.6f}  "
              f"|k|max={np.sqrt(k_norm2.max()):.6f}", flush=True)

        # ---- 2) 密度与容器语义交叉核对 ----------------------------------
        scf = np.load(f"{SCFDIR}/{name}.npz")
        fid = np.asarray(scf["final_spin_charge_density"], dtype=np.float64)  # (N,2,4)
        step2 = np.load(f"{STEPDIR}/{mol}-step2.npz")
        step2_scd = step2["spin_charge_density"]
        step2_flat = step2["spin_charge_density_flat"]
        f0fix = np.load(f"{FIXDIR}/{name}.npz")
        dens_coeff = np.asarray(f0fix["density_coefficients"], dtype=np.float64)  # (N,4)

        d_step = float(np.abs(fid - step2_scd).max())
        d_flat = float(np.abs(step2_flat - fid.reshape(N, -1)).max())
        d_model = float(np.abs(dens_coeff - fid.sum(axis=1)).max())
        d_alt_ch0 = float(np.abs(dens_coeff - fid[:, 0, :]).max())
        d_alt_flat4 = float(np.abs(dens_coeff - fid.reshape(N, -1)[:, :4]).max())
        print(f"  [sem] (N,2,4) 语义核对: scf.final == scf_steps.step2 spin_charge_density "
              f"max|Δ|={d_step:.2e}; flat==reshape(N,8) max|Δ|={d_flat:.2e}", flush=True)
        print(f"  [sem] {name}.density_coefficients(模型自己 = coulomb 实际 source_feats) "
              f"vs Σ_spin {d_model:.2e} | ch0 {d_alt_ch0:.3e} | flat[:,:4] {d_alt_flat4:.3e}",
              flush=True)
        spin_tot = fid.sum(axis=1)          # (N,4)  = 参考 charge_density (extensions.py L887)
        source_a = dens_coeff               # 源 A: 与目标同一次运行的模型出口
        source_b = spin_tot                 # 源 B: scf 夹具 final(不同运行, 独立核对)

        # ---- 3) 表重建(序列化常数 → 傅里叶基) --------------------------
        basis_re, basis_im, dbg = rebuild_density_basis(ce, k_vectors, k_norm2, k0_mask)
        ref_re = geo["density_basis_fs_re"][:, 0, :]
        ref_im = geo["density_basis_fs_im"][:, 0, :]
        d_re = float(np.abs(basis_re - ref_re).max())
        d_im = float(np.abs(basis_im - ref_im).max())
        scale = max(float(np.abs(ref_re).max()), float(np.abs(ref_im).max()), 1e-30)
        print(f"  [tbl] density_basis 重建 vs geo3 dump: max|Δ| re={d_re:.3e} im={d_im:.3e} "
              f"(ref 量级 {scale:.3f}; 相对 {max(d_re, d_im)/scale:.2e})", flush=True)
        # 解析重算(仅信息量; 序列化值是 checkpoint f32 缓冲)
        sigma = float(ce["density_basis"]["sigmas"][0])
        pref_const = 4 * np.pi * np.sqrt(np.pi / 2)              # gto_utils.py L26
        rel_analytic = {
            "pref0": abs(dbg["pref0"] - pref_const * sigma ** 3) / abs(dbg["pref0"]),
            "pref1": abs(dbg["pref1"] - pref_const * sigma ** 5) / abs(dbg["pref1"]),
        }
        print(f"  [tbl] 序列化常数 vs 解析重算(信息量; 差异=checkpoint f32 缓冲): " +
              " ".join(f"{k}={v:.2e}" for k, v in rel_analytic.items()), flush=True)

        # cos/sin 独立复算
        d_cos = float(np.abs(geo["cosines"] - np.cos(k_vectors @ positions.T)).max())
        d_sin = float(np.abs(geo["sines"] - np.sin(k_vectors @ positions.T)).max())
        kfac = np.zeros_like(k_norm2)
        nz = k0_mask <= 0.0
        kfac[nz] = 1.0 / k_norm2[nz]                          # energy.py L141-143
        d_kfac = float(np.abs(kfac - geo["k_factor_coulomb"]).max())
        d_vol = float(np.abs(geo["volume_per_k"] - volume).max())
        print(f"  [tbl] cos {d_cos:.2e} sin {d_sin:.2e} k_factor_coulomb {d_kfac:.2e} "
              f"volume_per_k {d_vol:.2e}", flush=True)

        # ---- 4) 能量链(逐式) --------------------------------------------
        results = {}
        for tag, src in (("A", source_a), ("B", source_b)):
            rho, cc, cs = assemble_density(src, geo["cosines"], geo["sines"],
                                           basis_re, basis_im, volume)
            potential = rho * kfac[:, None] * FIELD_CONSTANT   # apply_coulomb_kernel
            per_k = 2.0 * (rho * potential).sum(-1)            # energy_product_batch
            energy = 0.5 * volume * per_k.sum() / TWO_PI ** 6
            results[tag] = {"energy": float(energy), "rho": rho,
                            "potential": potential, "per_k": per_k}

        e_si, _ = self_interaction_energy(ce, source_a)
        mol_c, slab_c, is_mol, is_slab = pbc_correction_energy(
            ce, source_a, positions, volume, np.array([True, True, True]))
        corr = (mol_c if is_mol else 0.0) + (slab_c if is_slab else 0.0)

        e_a = results["A"]["energy"]
        e_b = results["B"]["energy"]
        target = float(target_index[name]["electrostatic_energy"])
        d_a = abs(e_a - target)
        d_b = abs(e_b - target)
        rel_a = d_a / abs(target)
        rel_b = d_b / abs(target)

        print(f"  [nrg] E_pycharge(A=模型 density_coefficients) = {e_a:.15f}", flush=True)
        print(f"  [nrg] E_pycharge(B=scf final Σ_spin)          = {e_b:.15f}", flush=True)
        print(f"  [nrg] target(index.json)                      = {target:.15f}", flush=True)
        print(f"  [nrg] |Δ|A={d_a:.3e} (rel {rel_a:.2e}) | |Δ|B={d_b:.3e} (rel {rel_b:.2e})",
              flush=True)
        print(f"  [nrg] 反事实: include_self_interaction 若为 False 将再减 0.5·Σ<q,self> = "
              f"{0.5 * e_si:+.6f} eV; 实际序列化值 True ⇒ 不执行(现 |Δ|≈0 自证)", flush=True)
        print(f"  [nrg] pbc 修正选择器: is_molecule={is_mol} is_slab={is_slab} "
              f"(TTT) → correction=0 (mol={mol_c:.3e}, slab={slab_c:.3e} 未参与)",
              flush=True)
        # 关键中间量前 3 个 k 点
        rho = results["A"]["rho"]
        print(f"  [mid] rho(k) 前3点: {np.array2string(rho[:3], precision=6)}", flush=True)
        print(f"  [mid] per_k 前3点: {np.array2string(results['A']['per_k'][:3], precision=6)}"
              f"  Σper_k={results['A']['per_k'].sum():.6f}", flush=True)

        # ---- 5) 过渡门参考: scf_steps step0 密度 = field_independent_spin_charge_density
        #      (= 引擎模块②后当前 accumulator 状态; 模块⑤未落地前 ⑥ 可先用它做隔离门)。
        step0 = np.load(f"{STEPDIR}/{mol}-step0.npz")["spin_charge_density"]
        rho0, _, _ = assemble_density(step0.sum(axis=1), geo["cosines"], geo["sines"],
                                      basis_re, basis_im, volume)
        per_k0 = 2.0 * (rho0 * rho0 * kfac[:, None] * FIELD_CONSTANT).sum(-1)
        e0 = float(0.5 * volume * per_k0.sum() / TWO_PI ** 6)
        print(f"  [itm] 过渡门(step0=field-independent 密度, 引擎当前可产出): "
              f"E_es(step0)={e0:.15f} eV", flush=True)

        rows.append({"case": name, "target": target, "E_A": e_a, "E_B": e_b,
                     "E_step0_interim": e0,
                     "abs_A": d_a, "rel_A": rel_a, "abs_B": d_b, "rel_B": rel_b,
                     "density_semantics": {"step_vs_final": d_step, "flat_vs_reshape": d_flat,
                                           "model_vs_spin_sum": d_model,
                                           "model_vs_ch0": d_alt_ch0,
                                           "model_vs_flat4": d_alt_flat4},
                     "basis_rebuild": {"re": d_re, "im": d_im},
                     "self_int_if_false": 0.5 * e_si})

    # ---- 5) 汇总 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("汇总 (source A = 模型自身 density_coefficients, 与目标同一次运行)")
    print(f"{'case':8s} {'target':>18s} {'E_numpy':>18s} {'|Δ|':>10s} {'rel':>10s} "
          f"{'E_es(step0)':>18s}")
    for r in rows:
        print(f"{r['case']:8s} {r['target']:18.12f} {r['E_A']:18.12f} "
              f"{r['abs_A']:10.2e} {r['rel_A']:10.2e} {r['E_step0_interim']:18.12f}")
    worst = max(r["rel_A"] for r in rows)
    worst_b = max(r["rel_B"] for r in rows)
    verdict = "PASS (公式已锁定)" if worst <= ENERGY_TARGET_REL else "MISMATCH"
    print(f"\n最大相对误差: A {worst:.3e} | B {worst_b:.3e}  (判据 {ENERGY_TARGET_REL:.0e})")
    print(f"VERDICT: {verdict}")

    def dump_corr_design():
        """序列化完备性审计: 前向路径全部参数量 vs JSON 实际提供。"""
        db = ce["density_basis"]
        checks = [
            ("density_max_l(=density_basis.max_l)", db["max_l"] == 1, "density_basis.max_l"),
            ("density_smearing_width(=sigmas[0])", len(db["sigmas"]) == 1,
             f"{db['sigmas']}"),
            ("kspace_cutoff", np.isfinite(kspace_cutoff), f"{kspace_cutoff!r}"),
            ("include_self_interaction", True, f"{include_si}"),
            ("include_pbc_corrections", True, f"{include_pbc}"),
            ("basis.cl_scale", True, db["cl_scale"]["shape"]),
            ("basis.expanded_l_indices", True, db["expanded_l_indices"]["shape"]),
            ("basis.real/imag_phase_factors", True,
             [db["real_phase_factors"]["shape"], db["imag_phase_factors"]["shape"]]),
            ("basis.permute_indices", True, db["permute_indices"]["shape"]),
            ("basis.radial sigma2/pref0/pref1", True,
             [db["radial_spline"][k]["shape"] for k in ("sigma2", "pref0", "pref1")]),
            ("self_interaction.overlap_constants", True,
             ce["self_interaction_terms"]["overlap_constants"]["shape"]),
            ("self_interaction.select_indices", True,
             ce["self_interaction_terms"]["select_indices"]["shape"]),
            ("aux.realspace_energy.raw_state_dict", True,
             list(ce["aux_blocks"]["realspace_energy"]["raw_state_dict"].keys())),
            ("aux.monopole_dipole_correction.raw_state_dict", True,
             list(ce["aux_blocks"]["monopole_dipole_correction"]["raw_state_dict"].keys())),
        ]
        print("\n序列化完备性(前向 PBC 路径):")
        for label, ok, info in checks:
            print(f"  [{'OK ' if ok else 'GAP'}] {label}: {info}")
        print("  运行时输入(非序列化; 引擎侧由几何/SCF 现场提供): k_vectors/k_norm2/"
              "k_vector_batch/k0_mask, source_feats(N,4), node_positions, volume, pbc")
        print("  解析常数(引擎硬编码, 与模块③同口径): FIELD_CONSTANT=1/(5.526349406e-3) "
              "(utils.py L12); CUBIC_MADELUNG=2.837297(L15); monopole_dipole 的 "
              "const=FIELD_CONSTANT/(4π)(slabs.py L205)")

    dump_corr_design()
    print("\nMODULE6-PREFLIGHT-DONE", flush=True)
    return 0 if worst <= ENERGY_TARGET_REL else 1


if __name__ == "__main__":
    sys.exit(main())
