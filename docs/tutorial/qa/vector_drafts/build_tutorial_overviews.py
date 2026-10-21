#!/usr/bin/env python3
"""Large-label vector overview diagrams; explanatory, not measured timings."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/tutorial/figures'; OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'docs/tutorial/.build/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
FONT=Path('/home/qiuyong/.local/share/fonts/noto-cjk-resume/NotoSansCJKsc-Regular.otf')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=str(FONT)).get_name(),'DejaVu Sans'],'font.size':14,'pdf.fonttype':3,'svg.fonttype':'path','text.color':'#183046'})
B='#537da9';T='#188b86'; INK='#183046';GRAY='#60798b'
def init(h=8):
 fig,ax=plt.subplots(figsize=(14,h));ax.set_xlim(0,14);ax.set_ylim(0,h);ax.axis('off');fig.subplots_adjust(0.015,0.015,.985,.985);return fig,ax
def box(ax,x,y,w,h,text,color=B,fs=14,fill='#edf3f8',lw=1.5):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=.10',fc=fill,ec=color,lw=lw));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.55,color=INK)
def arrow(ax,a,b,c=B,style='-|>',lw=1.8,rad=0):ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=16,color=c,lw=lw,connectionstyle=f'arc3,rad={rad}'))
def save(fig,name):
 for fmt in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{fmt}',dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.15)
 plt.close(fig)

fig,ax=init(8.7)
ax.text(.25,8.35,'一次 PolarMACE 能量与力评估：数据怎样流动？',fontsize=22,weight='bold')
ax.text(.25,7.92,'读图顺序：输入 → 局部描述 → 两层全局场更新 → 能量 → 反向求力 → MD 积分',fontsize=13,color=GRAY)
box(ax,.3,6.45,3.1,1.05,'输入\n元素、坐标 R、晶胞；Q、spin',fs=14)
box(ax,4.0,6.45,4.05,1.05,'局部 MACE 骨干\n局部能量、特征 h、初始自旋多极矩',fs=14)
arrow(ax,(3.45,6.98),(3.94,6.98))
box(ax,9.0,6.45,4.6,1.05,'每层场更新都读取当前多极矩\n从局部表征接入全局静电',T,14,'#e9f5f2')
arrow(ax,(8.1,6.98),(8.94,6.98),T)
# Explicit bounded field stack, fixed two layers, no self-consistency assertion.
ax.add_patch(FancyBboxPatch((.3,3.65),13.3,2.18,boxstyle='round,pad=.04',fc='#f4faf8',ec=T,lw=1.7))
ax.text(.55,5.45,'场更新模块：固定 2 层（并非迭代至自洽收敛）',fontsize=16,color=T,weight='bold')
steps=[('构建全局\n电荷／自旋密度',.65,2.6),('Ewald 全局静电\n电势 → 场特征',3.83,2.8),('局部多极矩更新\n接收全局场信息',7.16,2.55),('Fukui 归一化\n满足总 Q 与 spin',10.24,2.95)]
for text,x,w in steps:box(ax,x,3.98,w,.95,text,T,14,'white')
for a,b in [(3.3,3.77),(6.68,7.10),(9.76,10.18)]:arrow(ax,(a,4.45),(b,4.45),T)
arrow(ax,(11.3,6.4),(11.3,5.9),T)
box(ax,.4,1.73,5.05,1.05,'汇总三项能量\nE = E_local + E_nonlocal + E_elec',fs=15)
box(ax,6.15,1.73,3.65,1.05,'完整反向求导\nF = −∇ᴿ E',fs=16)
box(ax,10.5,1.73,3.0,1.05,'MD 积分器\n更新速度与坐标',color=GRAY,fs=15,fill='#f1f3f5')
arrow(ax,(2.9,3.6),(2.9,2.85),T)
arrow(ax,(5.5,2.25),(6.09,2.25));arrow(ax,(9.85,2.25),(10.44,2.25),GRAY)
# Timing boundary visually excludes right-most MD box.
ax.plot([.4,.4,9.8,9.8],[1.35,1.12,1.12,1.35],color=T,lw=2)
ax.text(5.1,.72,'本报告“能量 + 力评估”计时：模型前向 + 完整反向；不含 MD 积分',ha='center',fontsize=14,color=T)
ax.text(.35,.15,'多极矩携带角向信息；场更新让远处原子通过全局静电影响局部表征。',fontsize=13,color=GRAY)
save(fig,'00_full_workflow')

fig,ax=init(9.4)
ax.text(.25,9.03,'仓库里的 linear / nonlinear：是两条执行路径的简称',fontsize=22,weight='bold')
box(ax,4.1,7.8,5.8,.72,'共同输入：原子元素、坐标、晶胞',fs=16)
arrow(ax,(5.0,7.75),(3.4,7.2));arrow(ax,(9.0,7.75),(10.6,7.2),T)
# Panels
for x,c in [(.3,B),(7.2,T)]:ax.add_patch(FancyBboxPatch((x,3.5),6.5,3.68,boxstyle='round,pad=.04',fc='#f7fafc' if c==B else '#f4faf8',ec=c,lw=1.5))
ax.text(3.55,6.82,'标准 MACE-OFF23 路径',ha='center',fontsize=19,color=B,weight='bold')
ax.text(3.55,6.38,'工程简称：linear',ha='center',fontsize=14,color=GRAY)
ax.text(10.45,6.82,'PolarMACE 路径',ha='center',fontsize=19,color=T,weight='bold')
ax.text(10.45,6.38,'工程名称：MACE_Nonlinear',ha='center',fontsize=14,color=GRAY)
box(ax,.68,5.12,5.75,.82,'局部图：等变、多体、含非线性变换',B,15,'white')
box(ax,7.58,5.12,5.75,.82,'局部几何骨干 + 初始电荷／自旋多极矩',T,15,'white')
arrow(ax,(3.55,5.06),(3.55,4.78));arrow(ax,(10.45,5.06),(10.45,4.78),T)
box(ax,.68,3.85,5.75,.89,'局部能量前向 + 反向求力\n当前配置：FP32，direct / JIT streamed',B,14,'white')
box(ax,7.58,3.85,5.75,.89,'2 次全局场更新 / Fukui → 三项能量 → 反向\n当前配置：FP64，materialized',T,13.5,'white')
ax.text(.35,3.03,'为什么 Polar 路径可能更慢？需由实际 profile 判断各项占比。',fontsize=16,weight='bold')
reasons=[('① 更多局部计算\n算子更多／通道更宽',.35),('② 额外全局场更新\n前向与反向都要计算',4.95),('③ 中间数据与调度\n宽数组、访存、kernel 启动',9.55)]
for s,x in reasons:box(ax,x,1.5,4.1,1.05,s,T,14,'#e9f5f2')
box(ax,.35,.23,13.3,.85,'不是“线性能量模型 vs 非线性能量模型”；精度与执行路径是当前实现配置。\n不能据此认定 Ewald 是主瓶颈，也不能认定 Polar 原理上无法 streaming。',GRAY,14,'#f2f4f6')
save(fig,'00_paths_comparison')
print('Saved 00_full_workflow and 00_paths_comparison in',OUT)
