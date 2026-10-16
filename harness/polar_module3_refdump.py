#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块③参考侧数据落盘: k 网格 + GTO 几何预计算 (2026-10-15)。

引擎模块③（GTO precompute：k 网格 / 结构因子）的"Python 参考侧"夹具生成器。
对 4 个 scf 夹具盒（NH3/CH4/N2/CO2, 32 分子盒, 与 scf/*.npz 完全同构型）:

  1) 直接用 graph_longrange.kspace.compute_k_vectors_flat 生成 k 网格
     → bench/polar_fixtures/geo3/{mol}-kvecs.npz
  2) 用 mace_polar 载入模型, 取 model.electric_potential_descriptor, 按
     extensions.py::PolarMACE.forward 的**确切调用方式**调用 precompute_geometry
     （几何预计算 cache: 结构因子 cos/sin、GTO 基表、体积因子、库仑/投影因子）
     → bench/polar_fixtures/geo3/{mol}-pregeom.npz
  3) 汇总 bench/polar_fixtures/geo3/index.json（名字/形状/dtype/来源 file:line/用途）

--- recipe 依据（源码行号, 2026-10-15 核对）---------------------------------
[K1] mace/modules/extensions.py L710-718:
        (k_vectors, kv_norms_squared, k_vectors_batch, k_vectors_0mask) =
            compute_k_vectors_flat(self.kspace_cutoff, cell.view(-1,3,3),
                                   data["rcell"].view(-1,3,3))
[K2] rcell 约定: mace/data/atomic_data.py L369-372
        volume = torch.linalg.det(cell)
        rcell  = 2*pi * torch.linalg.inv(cell.mT)     # cell.mT = cell.T（行=晶格矢量）
      ⇒ rcell 行 i = b_i = 2π (A^{-1})^T 行 i, 满足 b_i·a_j = 2π δ_ij;
      k = n @ rcell = Σ_i n_i b_i（kspace.py L86-88 einsum "ni,bij->bnj"）。
      本盒为立方 cell=L·I, 故 rcell = (2π/L)·I; 脚本仍按上式构造并落盘 cell/rcell。
[K3] kspace.py L86-127: k 只取半空间（k 与 -k 等价; origin + half_line + half_plane +
      half_sphere）, 逐图按 |k|² ≤ cutoff² 过滤后 flatten; k0_mask[0]=1 标记每图首个
      （即原点）k 矢量。
[G1] extensions.py L720-730: precompute_geometry(
        k_vectors=..., k_norm2=kv_norms_squared, k_vector_batch=k_vectors_batch,
        k0_mask=k_vectors_0mask, node_positions=positions, batch=data["batch"],
        volume=data["volume"], pbc=data["pbc"].view(-1,3),
        force_pbc_evaluator=use_pbc_evaluator)          # 本模型默认 False
[G2] features.py L475-489: pbc 任一为真 → _pbc_precompute_geometry（本盒全 PBC）。
[G3] features.py L568-615 (_pbc_precompute_geometry 的产物清单):
        inner_products = k·r                 [Nk, N]
        mask = k_vector_batch==batch         [Nk, N]
        cosines/sines = cos/sin(inner) * mask        ← e^{ik·r} 的实/虚部
        density_basis_fs = density_basis(k, k², k0)  [Nk,1,4,2]  (实/虚两路)
        feature_basis_fs = feature_basis(k, k², k0)  [Nk,2,4,2]
        volume_per_k = volume[batch_of_k]            [Nk,]
        k_factor_coulomb[k=0]=0, else 1/k²           [Nk,]
        k_factor_proj[k=0]=0.5, else 1               [Nk,]
        volumes, batch, node_positions, pbc, correction_mode(=0 PBC)
     其中 density_basis_fs / feature_basis_fs 即 GTOBasis 在 k 网格上求值的
     "Y_lm(k)+径向增宽权重+归一化" 合成表（gto_utils.py L152-176:
     fnlk=radial_spline(k)*cl_scale → 按 l 展开 → ×Y_lm(k, e3nn 归一) → ×相位因子）。
     引擎侧同一张表的解析常数已在探针 JSON polar_blocks.electric_potential_descriptor
     .density_basis / .feature_basis 中序列化; 本夹具落盘的是"在该 k 网格上的定值"。
[G4] volume: data["volume"] = det(cell)（atomic_data.py L369）; 单图标量。
[G5] pbc: (1,3) all-True。

--- 自证（脚本内执行, 结果写入 index.json）-----------------------------------
  用 cache + forward_dynamic 重算 SCF 第 0 步 α/β 通道的势特征, 与 scf 夹具
  potential_feats_step_0/1 逐位对照（±1e-10 级 = cache/recipe 与模型内部完全一致）。
  需要 field_independent 密度 → 复用 bench/polar_module2_gate.py::ref_fukui_correct。

用法: cd mlip; CUDA_VISIBLE_DEVICES=1 /workspace/caches/conda/envs/polar/bin/python \
        bench/polar_module3_refdump.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")
import mace_md as mm  # noqa: E402

POLAR_MODEL = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
PROBE_JSON = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
FIXDIR = f"{ROOT}/bench/polar_fixtures/scf"
OUTDIR = f"{ROOT}/bench/polar_fixtures/geo3"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]

# 数组清单: name -> (来源 file:line, 用途)
SOURCES = {
    "k_vectors": ("graph_longrange/kspace.py::compute_k_vectors_flat L80-88,123",
                  "k 矢量 [Nk,3]=n@rcell（半空间）; 引擎 k 网格第一产出"),
    "k_norm2": ("graph_longrange/kspace.py L91-94,124",
                "|k|² [Nk]; 库仑核/高斯增宽/筛选判据"),
    "k_vector_batch": ("graph_longrange/kspace.py L113-120,125",
                       "k→图 id [Nk] (long); 单图时全 0; 批内分组"),
    "k0_mask": ("graph_longrange/kspace.py L108-109,126",
                "k=0 标记 [Nk] (1.0 仅每图首项); k=0 项处理(0.5/0 特殊化)"),
    "cell": ("atomic_data.py L369 + atomic_data.py L427 传入 cell",
             "(1,3,3) 晶格矢量(行); 门侧复算 k 网格输入"),
    "rcell": ("mace/data/atomic_data.py L369-372",
              "(1,3,3) 倒格: rcell = 2π·inv(cell.T), 行 i = b_i, b_i·a_j=2πδ_ij; 门侧复算输入"),
    "volume": ("mace/data/atomic_data.py L369",
               "det(cell) [1]; 结构因子归一 (2π)³/V 的体积因子"),
    "positions": ("fixtures scf/{mol}-f0.npz positions (build_gas 重建)",
                  "(N,3) 原子坐标; 结构因子 e^{ik·r} 输入"),
    "kspace_cutoff": ("models/mace-polar/polar_symmetrix_probe.json "
                      "polar_blocks.electric_potential_descriptor.kspace_cutoff",
                      "k 空间截断 1.8467166423797607; 引擎硬编码校验值"),
    "cosines": ("graph_longrange/features.py L579-582 (E{ikr} 实部)",
                "cos(k·r)·mask [Nk,N]; 傅里叶求和/反投影结构因子"),
    "sines": ("graph_longrange/features.py L579-582 (E{ikr} 虚部)",
              "sin(k·r)·mask [Nk,N]; 傅里叶求和/反投影结构因子"),
    "density_basis_fs_re": ("graph_longrange/features.py L585 + gto_utils.py L152-176",
                            "密度基 GTO 表实部 [Nk,1,4]; assemble_fourier_series 的 rho(k) 基"),
    "density_basis_fs_im": ("同上 (虚部 = 相位因子 imag 路)",
                            "密度基 GTO 表虚部 [Nk,1,4]"),
    "feature_basis_fs_re": ("graph_longrange/features.py L586 + gto_utils.py L152-176",
                            "势特征 GTO 表实部 [Nk,2,4] (2 增宽 sigma=1.5,3.0)"),
    "feature_basis_fs_im": ("同上 (虚部)", "势特征 GTO 表虚部 [Nk,2,4]"),
    "volume_per_k": ("graph_longrange/features.py L588",
                     "每 k 图的体积 [Nk]; rho(k) 归一 (2π)³/V"),
    "k_factor_coulomb": ("graph_longrange/features.py L589-591",
                         "库仑核因子 [Nk]: k=0→0, 否则 1/k²"),
    "k_factor_proj": ("graph_longrange/features.py L592-593",
                      "反投影 k=0 因子 [Nk]: k=0→0.5, 否则 1"),
    "volumes": ("graph_longrange/features.py L606", "每图体积 [n_graphs]; 非周期修正项用"),
    "pbc": ("graph_longrange/features.py L609 (data['pbc'].view(-1,3))",
            "(1,3) all-True; correction_mode 判据"),
    "node_positions": ("graph_longrange/features.py L608",
                       "(N,3) 与结构因子同源坐标 (cache 内副本)"),
    "batch": ("graph_longrange/features.py L607", "(N,) 图归属 (long), 单图全 0"),
    "correction_mode": ("graph_longrange/features.py L595 + L22-25",
                        "0=PBC（本盒）; 非周期修正分支选择"),
    "output_permutation": ("graph_longrange/features.py L399-406, L638",
                           "[8] 势特征通道重排 (index_select); 模块④输出顺序依据"),
}


def build_box(mol, seed, n_mol=32, rho=0.01):
    """确定性重建夹具盒（与 polar_scf_hooks_dump.py 相同调用）。

    build_gas 返回 (atoms, mol_index, info)。
    """
    base, mol_index, info = mm.build_gas(mol, n_mol, rho, seed, 2.0)
    return base, info


def geometry_inputs(mol, seed, cutoff, dev):
    """重建盒 + k 网格 + precompute_geometry 的全部输入（recipe 供两脚本共用）。

    返回 dict; 张量均 float64 在 dev 上。调用点 = extensions.py L710-730。
    同时做 build_gas 与夹具 positions 一致性断言（atol 1e-10）。
    """
    import torch
    from graph_longrange.kspace import compute_k_vectors_flat

    name = f"{mol.lower()}-f0"
    fix = np.load(f"{FIXDIR}/{name}.npz", allow_pickle=False)
    atoms, info = build_box(mol, seed)
    pos_fix = np.asarray(fix["positions"], dtype=np.float64)
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    if not np.allclose(pos_fix, pos, atol=1e-10):
        raise RuntimeError(f"{name}: build_gas 与夹具 positions 不一致 "
                           f"(max|Δ|={np.abs(pos_fix - pos).max():.3e})")
    cell_np = np.asarray(atoms.cell.array, dtype=np.float64)[None, :, :]  # (1,3,3)
    volume_np = np.array([np.linalg.det(cell_np[0])], dtype=np.float64)

    cell_t = torch.tensor(cell_np, dtype=torch.float64, device=dev)
    # rcell = 2π·inv(cell.T)  (atomic_data.py L371; cell 行=晶格矢量)
    rcell_t = 2.0 * torch.pi * torch.linalg.inv(cell_t.mT)
    positions_t = torch.tensor(pos, dtype=torch.float64, device=dev)
    batch_t = torch.zeros(pos.shape[0], dtype=torch.long, device=dev)
    volume_t = torch.tensor(volume_np, dtype=torch.float64, device=dev)
    pbc_t = torch.ones(1, 3, dtype=torch.bool, device=dev)

    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
        cutoff, cell_t, rcell_t)
    return {
        "name": name, "atoms": atoms, "info": info, "fixture": fix,
        "pos": pos, "cell_np": cell_np, "rcell_np": rcell_t.cpu().numpy(),
        "volume_np": volume_np,
        "k_vectors": k_vectors, "k_norm2": k_norm2, "k_batch": k_batch,
        "k0_mask": k0_mask,
        "positions_t": positions_t, "batch_t": batch_t,
        "volume_t": volume_t, "pbc_t": pbc_t,
    }


def precompute_geometry_cache(desc, gi):
    """按 extensions.py L720-730 的确切调用方式执行 precompute_geometry。"""
    return desc.precompute_geometry(
        k_vectors=gi["k_vectors"],
        k_norm2=gi["k_norm2"],
        k_vector_batch=gi["k_batch"],
        k0_mask=gi["k0_mask"],
        node_positions=gi["positions_t"],
        batch=gi["batch_t"],
        volume=gi["volume_t"],
        pbc=gi["pbc_t"],
        force_pbc_evaluator=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    import torch
    from mace.calculators import mace_polar
    from polar_module2_gate import ref_fukui_correct  # noqa: E402 (bench/)

    cutoff = float(json.load(open(PROBE_JSON))["polar_blocks"]
                   ["electric_potential_descriptor"]["kspace_cutoff"])
    print(f"[cfg] kspace_cutoff = {cutoff!r}", flush=True)

    calc = mace_polar(model=POLAR_MODEL, device=args.device,
                      default_dtype="float64")
    model = calc.models[0]
    desc = model.electric_potential_descriptor
    print(f"[cfg] model num_recursion_steps={model.num_recursion_steps} "
          f"charges_irreps={model.charges_irreps} "
          f"field_irreps={model.field_irreps} "
          f"charges_to_mul_ir={model._charges_to_mul_ir}", flush=True)
    dev = next(model.parameters()).device
    os.makedirs(args.outdir, exist_ok=True)

    index = {
        "generator": "bench/polar_module3_refdump.py",
        "date": "2026-10-15",
        "kspace_cutoff": cutoff,
        "model": POLAR_MODEL,
        "recipe": {
            "k_grid": "extensions.py L710-718 -> compute_k_vectors_flat(cutoff, "
                      "cell.view(-1,3,3), rcell.view(-1,3,3))",
            "rcell": "atomic_data.py L371: rcell = 2*pi*inv(cell.mT)",
            "geom_cache": "extensions.py L720-730 -> descriptor.precompute_geometry("
                          "k_vectors,k_norm2,k_vector_batch,k0_mask,"
                          "node_positions,batch,volume,pbc,force_pbc_evaluator=False)",
            "dtype": "float64",
        },
        "arrays": {},
        "cases": {},
    }

    for mol, seed in MOLS:
        # 几何重建 + k 网格（recipe 与 forward L710-730 同源, 见 geometry_inputs）
        gi = geometry_inputs(mol, seed, cutoff, dev)
        name, fix, pos = gi["name"], gi["fixture"], gi["pos"]
        cell_np, volume_np, L = gi["cell_np"], gi["volume_np"], gi["info"]["cell_L_A"]
        k_vectors, k_norm2 = gi["k_vectors"], gi["k_norm2"]
        k_batch, k0_mask = gi["k_batch"], gi["k0_mask"]
        Nk = k_vectors.shape[0]

        # ---- 1) k 网格落盘 ------------------------------------------------
        kv = {
            "k_vectors": k_vectors.cpu().numpy(),
            "k_norm2": k_norm2.cpu().numpy(),
            "k_vector_batch": k_batch.cpu().numpy(),
            "k0_mask": k0_mask.cpu().numpy(),
            "cell": cell_np,
            "rcell": gi["rcell_np"],
            "volume": volume_np,
            "positions": pos,
            "kspace_cutoff": np.array([cutoff]),
        }
        np.savez(f"{args.outdir}/{name}-kvecs.npz", **kv)
        print(f"[kvec] {name}: Nk={Nk}  |k|max="
              f"{float(k_norm2.max())**0.5:.6f} <= {cutoff:.6f}  "
              f"k0_mask[0]={float(k0_mask[0])}  "
              f"pos_bitwise_eq_fixture="
              f"{bool(np.array_equal(pos, np.asarray(fix['positions'])))}",
              flush=True)

        # ---- 2) precompute_geometry (与 extensions.py L720-730 完全同参) ----
        with torch.no_grad():
            cache = precompute_geometry_cache(desc, gi)
        assert cache["mode"] == "pbc", cache.get("mode")

        # 复数对 (数值表) 拆 real/imag 两路落盘; 其余原样
        pg = {
            "cosines": cache["cosines"].cpu().numpy(),
            "sines": cache["sines"].cpu().numpy(),
            "density_basis_fs_re": cache["density_basis_fs"][..., 0].cpu().numpy(),
            "density_basis_fs_im": cache["density_basis_fs"][..., 1].cpu().numpy(),
            "feature_basis_fs_re": cache["feature_basis_fs"][..., 0].cpu().numpy(),
            "feature_basis_fs_im": cache["feature_basis_fs"][..., 1].cpu().numpy(),
            "volume_per_k": cache["volume_per_k"].cpu().numpy(),
            "k_factor_coulomb": cache["k_factor_coulomb"].cpu().numpy(),
            "k_factor_proj": cache["k_factor_proj"].cpu().numpy(),
            "volumes": cache["volumes"].cpu().numpy(),
            "node_positions": cache["node_positions"].cpu().numpy(),
            "batch": cache["batch"].cpu().numpy(),
            "pbc": cache["pbc"].cpu().numpy(),
            "correction_mode": np.array([int(cache["correction_mode"])]),
            "output_permutation":
                desc.output_permutation.cpu().numpy(),
            "k_norm2": k_norm2.cpu().numpy(),
            "k0_mask": k0_mask.cpu().numpy(),
            "k_vectors": k_vectors.cpu().numpy(),
        }
        np.savez(f"{args.outdir}/{name}-pregeom.npz", **pg)

        # ---- 3) 自证: cache + forward_dynamic 复算 SCF 首步势特征 ------------
        init = (np.asarray(fix["lr_src_0"])[:, 0, :]
                + np.asarray(fix["lr_src_1"])[:, 0, :])       # (N,8)
        fid = ref_fukui_correct(init, np.asarray(fix["fukui_src"]), 0.0, 1.0)
        den = fid.reshape(-1, 2, 4)
        check = {}
        for ch, key in ((0, "potential_feats_step_0"), (1, "potential_feats_step_1")):
            src = torch.tensor(den[:, ch, :][:, None, :], dtype=torch.float64,
                               device=dev)
            with torch.no_grad():
                feats = desc.forward_dynamic(cache=cache, source_feats=src,
                                             pbc=gi["pbc_t"])
            ref = np.asarray(fix[key], dtype=np.float64)
            dd = np.abs(feats.cpu().numpy() - ref)
            check[key] = {"max_abs": float(dd.max()),
                          "mean_abs": float(dd.mean()),
                          "ref_absmax": float(np.abs(ref).max())}
            print(f"[check] {name} {key} vs forward_dynamic(cache): "
                  f"max|Δ|={dd.max():.3e}", flush=True)

        # 形状/摘要入 index
        case = {"n_atoms": int(pos.shape[0]), "cell_L": L, "Nk": int(Nk),
                "npz": {"kvecs": f"geo3/{name}-kvecs.npz",
                        "pregeom": f"geo3/{name}-pregeom.npz"},
                "cutoff_check": {
                    "nk_cutoff_respected": bool(
                        float(k_norm2.max()) <= cutoff * cutoff * (1 + 1e-12)),
                    "k0_first": bool(float(k0_mask[0]) == 1.0)},
                "forward_dynamic_check": check,
                "arrays": {}}
        for k, v in {**kv, **pg}.items():
            case["arrays"][k] = {"shape": list(np.asarray(v).shape),
                                 "dtype": str(np.asarray(v).dtype),
                                 "source": SOURCES.get(k, ("?", "?"))[0],
                                 "use": SOURCES.get(k, ("?", "?"))[1]}
        index["cases"][name] = case

    index["arrays"] = {k: {"source": s[0], "use": s[1]}
                       for k, s in SOURCES.items()}
    with open(f"{args.outdir}/index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"[done] {len(MOLS)} cases -> {args.outdir} "
          f"(kvecs.npz / pregeom.npz / index.json)", flush=True)
    print("MODULE3-REFDUMP-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
