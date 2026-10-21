# Symmetrix-Polar 中文教程

[打开 PDF](SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) · [打开离线网页](index.html) · [阅读 Markdown](SYMMETRIX_POLAR_TUTORIAL_2026-10-21.md)

本教程用于学习与汇报，从原子、能量和力讲起，再解释 MACE 的等变表示、PolarMACE 的全局响应、Symmetrix-XL 的执行方式、GPU 基础、本项目优化、正确性验证和结果比较。末尾包含公式串讲、术语表、汇报问答和来源索引。

主体为中文，保留 MACE、PolarMACE、Ewald 等通用英文名称。性能统一使用 **µs/atom/step**，数学公式独立排版。原论文图、Python 数据图和 AI 概念插图区分说明，不把概念图当成计算结果。

## 文件

- PDF：带页码目录与书签；数学公式为 TeX 矢量排版。
- index.html：内嵌图片、CSS 与原生 MathML；下载后可离线打开，无需 CDN。
- figures/：8 张 Python 图的 PNG/PDF/SVG、2 张概念图、4 张原论文图的提取副本。
- derived_metrics.json：从已有 JSON 派生的统计与换算，保留筛选规则。
- image_provenance.json：概念图提示词、工具与用途。
- qa/：编译与版式检查记录。图表附件使用 Type3 字体／SVG 文字轮廓确保中文完整；最终教程使用高清 PNG，数学公式仍为矢量。

## 证据范围

本次重新核对已有论文和已提交测试记录，没有重新运行 GPU benchmark。128 原子与 1,296 原子来自不同实验分支，不构成同一条规模曲线。所有轮次与稳定轮统计分别报告；原始 JSON 保留不变。具体限制与进一步验证计划见正文。

## 重新生成

在仓库根目录运行：

~~~bash
python harness/build_tutorial_figures.py
python harness/build_tutorial.py
~~~

需要 Python、matplotlib、numpy、Pillow、PyMuPDF、pandoc、XeLaTeX 与可用中文字体。图表使用已提交数据；文献原图提取副本已包含在阅读包中。源文与引用依据见教程附录 D。

原论文插图仍归相应权利人，图注保留作者、版本和图号；本仓库代码许可不改变这些图的权利状态。
