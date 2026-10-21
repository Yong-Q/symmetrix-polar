#!/usr/bin/env python3
"""Rebuild textbook figures from committed measurements; no synthetic benchmarks."""
from pathlib import Path
import json
import statistics as st
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tutorial"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".build/mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FONT = ROOT / "docs/technical/fonts/NotoSansSC-Medium-static.ttf"
if not FONT.exists():
    FONT = Path("/home/qiuyong/.local/share/fonts/noto-cjk-resume/NotoSansCJKsc-Regular.otf")
font_manager.fontManager.addfont(str(FONT))
for f in [Path("/home/qiuyong/fonts/arial.ttf"), Path("/home/qiuyong/fonts/ARIALBD.TTF")]:
    if f.exists():
        font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.edgecolor": "#b6c7d3", "axes.linewidth": .8,
    "text.color": "#183046", "axes.labelcolor": "#183046",
    "xtick.color": "#496376", "ytick.color": "#496376",
    "svg.fonttype": "path", "pdf.fonttype": 3, "axes.unicode_minus": False,
})
B, T, G, R = "#537da9", "#188b86", "#a6b5c1", "#c56d54"
DATA = ROOT / "docs/results/data"
def read(n): return json.loads((DATA / n).read_text())
def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

gather = read("ab_gather_tile_1296.json")["summary"]
edge = read("ab_edge_features_1296.json")["summary"]
sandbox = read("ab_path_grouped_forward_sandbox.json")["summary"]
up = read("ab_edge_up_scatter_1296.json")
selected = {x["tag"]: x["median_ms"] for x in up["results"]}
final_ms = st.median([selected["comp2_r1"], selected["comp2_r2"]])
pairs = [
    ("消息汇总：重排线程", gather["base_old"]["median_of_medians_ms"],
     gather["tile128"]["median_of_medians_ms"], "ab_gather_tile_1296.json", "two clean rounds"),
    ("边特征：连续访存", edge["base"]["median_of_medians_ms"],
     edge["comp2"]["median_of_medians_ms"], "ab_edge_features_1296.json", "two clean rounds"),
    ("张量积：复用预收缩", sandbox["main_full"]["median"] * 1.296,
     sandbox["cand_full"]["median"] * 1.296, "ab_path_grouped_forward_sandbox.json", "two clean sandbox rounds"),
    ("边更新：融合与重排", selected["base_r2"], final_ms,
     "ab_edge_up_scatter_1296.json", "post-hoc: baseline r2; candidate r1,r2"),
]
derived = {
    "source_commit": __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "atoms": 1296, "precision": "FP64", "hardware": "one A100 80GB",
    "final_selected_round_medians_ms": [selected["comp2_r1"], selected["comp2_r2"]],
    "final_selected_ms": final_ms, "final_selected_us_per_atom": final_ms / 1.296,
    "all_round_median_ms": up["summary"]["comp2"]["median_of_medians_ms"],
    "all_round_median_us_per_atom": up["summary"]["comp2"]["median_of_medians_ms"] / 1.296,
    "selection_note": "Exploratory post-hoc stable-round selection. All excluded rounds remain in the source and all-round plot. Not a prospective dedicated-GPU qualification.",
    "pairs": [{"name": n, "before_ms": a, "after_ms": b,
               "reduction_pct": 100 * (1-b/a), "source": s, "selection": sel}
              for n,a,b,s,sel in pairs],
    "historical_endpoint_ms": [599.54, final_ms],
    "historical_endpoint_reduction_pct": 100 * (1-final_ms/599.54),
    "historical_endpoint_speed_ratio": 599.54/final_ms,
    "pytorch_128_ms": 94.0, "b30_128_ms": 30.557,
    "pytorch_128_ratio": 94.0/30.557,
}
(OUT / "derived_metrics.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2)+"\n")

# Contract: matched local A/B comparisons, distinct from the historical ladder.
fig, ax = plt.subplots(figsize=(9.6, 4.5))
y = np.arange(4)[::-1]
for yi,(name,a,b,_,_) in zip(y,pairs):
    ax.plot([b/1.296,a/1.296],[yi,yi],color="#dce5ec",lw=8,solid_capstyle="round")
    ax.scatter([a/1.296],[yi],s=70,color=G,zorder=3)
    ax.scatter([b/1.296],[yi],s=70,color=T,zorder=4)
    ax.text(a/1.296+1.3,yi,f"{a/1.296:.1f} → {b/1.296:.1f}   −{100*(1-b/a):.1f}%",
            va="center",fontsize=9)
ax.set_yticks(y, [p[0] for p in pairs])
ax.set_xlim(210,318); ax.set_ylim(-.55,3.55)
ax.set_xlabel("µs/atom/step；1,296 原子 H₂O，完整 PolarMACE，FP64，单张 A100")
ax.set_title("每次局部改动分别节省了多少时间",loc="left",pad=16)
ax.grid(axis="x",alpha=.17)
ax.text(0,-.25,"灰点：改动前；青点：改动后。最后一行使用事后选取的稳定轮；见正文完整样本。",transform=ax.transAxes,fontsize=8.5,color="#60798b")
fig.tight_layout(); save(fig,"01_clean_ab")

# Contract: honest historical progression; invalid intermediate team build omitted.
history = [
    ("历史基线",599.54),("修复后的 team + B4",348.97),
    ("消息汇总",333.17),("边特征",320.96575),
    ("张量积预收缩\n独立 sandbox",sandbox["cand_full"]["median"]*1.296),
    ("边更新\n稳定轮",final_ms)]
fig,ax=plt.subplots(figsize=(9.6,4.8))
yy=np.arange(len(history))[::-1]
vals=[v/1.296 for _,v in history]
ax.barh(yy,vals,color=[G]+[B]*4+[T],height=.6)
for a,v in zip(yy,vals):ax.text(v+5,a,f"{v:.1f}",va="center")
ax.set_yticks(yy,[n for n,_ in history]);ax.set_xlim(0,525)
ax.set_xlabel("µs/atom/step；历史阶段概览，不是同场逐项消融")
ax.set_title("历史端点约减少 51.6% 耗时；中间各阶段来自不同测量批次",loc="left",pad=14)
ax.grid(axis="x",alpha=.15)
fig.tight_layout();save(fig,"02_history")

# Contract: show every stored timing sample, including slow selected-out rounds.
fig,axs=plt.subplots(1,2,figsize=(10.4,4.5),sharey=True)
for ax,filename,title in zip(axs,["ab_path_grouped_forward_1296.json","ab_edge_up_scatter_1296.json"],
                            ["张量积主线 A/B：全部轮次","边更新 A/B：全部轮次"]):
    d=read(filename)
    for k,row in enumerate(d["results"]):
        cand=row["tag"].startswith("comp")
        color=T if cand else G
        vals=np.array(row["samples_ms"])/1.296
        offsets=np.linspace(-.10,.10,len(vals))
        ax.scatter(k+offsets,vals,s=22,color=color,alpha=.75)
        ax.plot([k-.2,k+.2],[row["median_ms"]/1.296]*2,color=color,lw=2)
    ax.set_xticks(range(len(d["results"])),[r["tag"].replace("comp2","改后").replace("base","改前") for r in d["results"]],rotation=40,ha="right",fontsize=8)
    ax.set_title(title,loc="left");ax.grid(axis="y",alpha=.18)
axs[0].set_ylabel("µs/atom/step；每点是一条已存测量")
fig.suptitle("慢轮没有删除：稳定轮统计属于探索性筛选",x=.08,ha="left",fontsize=12,y=1.02)
fig.tight_layout();save(fig,"03_all_samples")

# Contract: one profiler snapshot, mutually exclusive families, no mix with later state.
prof=read("cupti_b4team_1296_summary.json")
families=prof["families"]
assert abs(sum(x["total_ms"] for x in families)-prof["gpu_busy_ms"])<1e-5
mapping={"polar pipeline":"极化更新流水线","backbone nonlinear":"非线性骨干","tensor product":"张量积",
         "cuBLAS/external":"cuBLAS / 外部算子","MLP/dense":"MLP / 稠密变换","e3nn linear":"等变线性层","other":"其他"}
large=families[:7]
fig,ax=plt.subplots(figsize=(9.4,4.6))
yy=np.arange(len(large))[::-1]
for yi,f in zip(yy,large):
    pct=100*f["total_ms"]/prof["gpu_busy_ms"]
    ax.barh(yi,pct,color=T if yi>=4 else B,height=.62)
    ax.text(pct+.7,yi,f"{pct:.1f}%",va="center",fontsize=9)
ax.set_yticks(yy,[mapping.get(f["family"],f["family"]) for f in large]);ax.set_xlim(0,49)
ax.set_xlabel("占同一次 trace 的 GPU kernel 时长总和的比例")
ax.set_title("较早的 B4+team 快照：极化、骨干和张量运算共同占时",loc="left",pad=14)
ax.grid(axis="x",alpha=.15)
fig.tight_layout();save(fig,"04_profile")

# Contract: 128 atom same-model reference separated from 1296 atom benchmark.
fig,ax=plt.subplots(figsize=(9.2,3.4))
names=["PyTorch 参考","早期 C++/Kokkos 引擎","Plan B-30ms"]
vv=[94./128*1000,183.8/128*1000,30.557/128*1000]
ax.barh([2,1,0],vv,color=[G,B,T],height=.58)
for y,v in zip([2,1,0],vv):ax.text(v+22,y,f"{v:.1f}",va="center",fontsize=9)
ax.set_yticks([2,1,0],names);ax.set_xlim(0,1820)
ax.set_xlabel("µs/atom/step；128 原子 NH₃，FP64，单张 A100")
ax.set_title("128 原子分支：相对所记录 PyTorch 参考约 3.08×",loc="left",pad=12)
ax.grid(axis="x",alpha=.15);fig.tight_layout();save(fig,"05_reference_128")

# Contract: independent paper reproduction; minima and medians shown separately.
fig,ax=plt.subplots(figsize=(9.2,3.9))
xx=np.arange(3); mins=[4.53,4.47,4.50]; meds=[6.37,4.49,5.43]
ax.plot(xx,meds,"o-",color=B,label="各轮中位数")
ax.scatter(xx,mins,s=75,marker="D",color=T,label="各轮最小值",zorder=4)
ax.axhline(4.50,color=R,ls="--",label="论文 4.50")
for x,v in zip(xx,meds):ax.text(x+.03,v+.06,f"{v:.2f}",color=B)
for x,v in zip(xx,mins):ax.text(x-.1,v-.22,f"{v:.2f}",color=T)
ax.set_xticks(xx,["进程 1","进程 2","进程 3"]);ax.set_ylim(3.9,7)
ax.set_ylabel("µs/atom/step");ax.set_title("OFF23 / 5,184 原子水 / FP32：最小值接近论文，中位数仍有波动",loc="left",pad=12)
ax.legend(ncol=3,loc="upper center",frameon=False,fontsize=9);ax.grid(axis="y",alpha=.15)
fig.tight_layout();save(fig,"06_floor_vs_median")

# Contract: trajectory parity, potential energy is not conserved total NVE energy.
md=read("gate_md_10step_128.json")
fig,(a,b)=plt.subplots(2,1,figsize=(9,4.7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
orig=np.array(md["orig"]["pe"]);fixed=np.array(md["fixed"]["pe"])
a.plot(orig[:,0],orig[:,1]-orig[0,1],"o-",color=B,label="原始引擎")
a.plot(fixed[:,0],fixed[:,1]-fixed[0,1],"x",color=T,markersize=8,label="修复后引擎")
a.set_ylabel("势能相对初值 / eV");a.legend(frameon=False,ncol=2);a.grid(alpha=.15)
b.plot(orig[:,0],fixed[:,1]-orig[:,1],"o-",color=T)
b.set_ylabel("两实现势能差 / eV");b.set_xlabel("积分步（0.5 fs / 步，共 5 fs）");b.grid(alpha=.15)
a.set_title("短轨迹实现一致；这不是长期 NVE 能量守恒证明",loc="left",pad=12)
fig.tight_layout();save(fig,"07_md_parity")

# Contract: illustrative bound, explicitly not a measured benchmark.
f=np.linspace(0,1,250); s=2.22
fig,ax=plt.subplots(figsize=(8.7,3.8))
ax.plot(f,100*f*(1-1/s),color=T,lw=2)
ax.scatter([.25],[100*.25*(1-1/s)],color=R,zorder=3)
ax.text(.28,15,"占全步 25% → 全步耗时减少 13.7%",fontsize=9)
ax.set_xlabel("被优化部分占原始全步耗时的比例 f");ax.set_ylabel("全步耗时减少 / %")
ax.set_title("Amdahl 算例：单个算子加速 2.22×，整体收益取决于原先占比",loc="left",pad=12)
ax.grid(alpha=.15);fig.tight_layout();save(fig,"08_amdahl_example")
print(json.dumps({"output":str(FIG),"final_ms":final_ms,"final_us":final_ms/1.296,
                  "figures":8},ensure_ascii=False))
