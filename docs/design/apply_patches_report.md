# ⑥+⑦ 引擎补丁应用器 dry-run 小结 — `bench/apply_polar_patches.py`（2026-10-15）

**脚本**: `mlip/bench/apply_polar_patches.py`（默认 `--dry-run`；`--apply` 已实现但**未**在真实树执行）
**运行命令**:
```bash
cd /workspace/group/shared/bza/task3_sampling_database/mlip
python3 bench/apply_polar_patches.py --dry-run --log logs/apply_patches_dryrun.log
```
**完整输出**: `mlip/logs/apply_patches_dryrun.log`（121 行，退出码 0）。
**引擎树**: `/workspace/group/shared/bza/external-repos/symmetrix-xl`（只读；运行后 md5 无变化，
无 `.bak`/`.polar_tmp` 残留；全程未调用 git、未构建）。

---

## 1. 块数与应用顺序

**13 块**（⑦ 8 步 + ⑥ 5 步），顺序 = `merge_preflight_67.md` §5.1/§6（**先⑦后⑥**）。
插入内容与锚点文本均由脚本**程序化提取**自两份 patch_final 文档的代码块（逐字）；
⑥-A/⑥-B/⑥-C2 的锚点为文档行内原文（预检 §6 定位）；⑥-D 锚点按 §5.2(a) **顺延规则**
取 ⑦-4.5 插入后的当前链尾终端块（原 `polar_scf_final_density` 块应用时刻 count=0，已验证）。

## 2. 逐块命中状态（dry-run：锚点 count 均为 1，全部通过）

| # | 块 | 文件 | 锚点命中 | 插入/替换 | 来源 |
|---|---|---|---|---|---|
| 1 | ⑦-4.1a | hpp | count=1 @ 行 381 | 后插 13 行 | module7 §4.1a / preflight §6.1 |
| 2 | ⑦-4.1b | hpp | count=1 @ 行 813 | 后插 9 行 | module7 §4.1b |
| 3 | ⑦-4.2a | tpp | count=1 @ 行 3596 | 中插 88 行（锚点第 3/4 行间） | module7 §4.2a |
| 4 | ⑦-4.2b | tpp | count=1 @ 行 3237 | 中插 4 行（闭括号前） | module7 §4.2b |
| 5 | ⑦-4.2c | tpp | count=1 @ 行 3744 | 中插 4 行（闭括号前） | module7 §4.2c |
| 6 | ⑦-4.3 | tpp | count=1 @ 行 6512 | 前插 145 行 | module7 §4.3 |
| 7 | ⑦-4.4 | tpp | count=1 @ 行 6680 | 后插 11 行 | module7 §4.4 |
| 8 | ⑦-4.5 | cpp | count=1 @ 行 576 | 整块替换 3→39 行 | module7 §4.5 |
| 9 | ⑥-A | hpp | count=1 @ 行 399（⑦块尾紧邻其上） | 前插 10 行 | module6 §1；顺延 §5.2(a) |
| 10 | ⑥-B | tpp | count=1 @ 行 3691（⑦-4.2a 块尾紧邻其上） | 前插 30 行 | module6 §2；顺延 §5.2(a) |
| 11 | ⑥-C1 | tpp | count=1 @ 行 6533（整段 8 行锚点） | 中插 108 行（末行 `    }` 前） | module6 §3.2 |
| 12 | ⑥-C2 | tpp | count=1 @ 行 8452（上方空行+`});`） | 前插 14 行 | module6 §3.3 |
| 13 | ⑥-D | cpp | count=1 @ 行 611（⑦ 链尾块） | 链尾展开 4→17 行 | module6 §4；顺延 §5.2(a) |

（行号 = 顺序模拟中该块应用时刻的定位，随顺序位移，仅作参考；判据为文本锚点 count==1。）

## 3. 应用后行数变化与 `git diff --stat` 预估

预估由内存 difflib 行差计算（**未调用 git**；相对当前工作树的增量）：

```
 libsymmetrix/source/mace_nonlinear_kokkos.hpp       |  32 ++++      815 -> 847  (+32)
 libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp  | 404 ++++++   8296 -> 8700 (+404)
 symmetrix/source/cpp/mace_nonlinear_kokkos.cpp      |  49 ++++       587 -> 636  (+49)
 3 files changed, 485 insertions(+), 0 deletions(-)
```

## 4. 应用后探针（内存结果，全部 OK）

- 结构：`P3 模块⑦` hpp=2/tpp=2、`P3 模块⑥` hpp=1/tpp=3、`polar_le_enabled`(tpp)=6、
  `launch_le_uvu`=3、`.def("polar_coulomb`(cpp)=3、⑤⑥⑦ 各步 grep（preflight §6）全部命中期望值。
- 装配序：`scale nonlinear energies` < `polar le energy add` < `if(has_zbl) {`（tpp 行 6818<6827<8466）；
  ⑤ 写 `polar scf final density` < ⑥ 读 `coulomb_density=polar_scf_final_density`（6533<6564）。
- 幂等标记应用前全 0（树未被本补丁改动）。

## 5. 不一致与处理

1. **tpp 树漂移（非阻塞）**：preflight §1 定稿读数 8292 行 / `a4c3b5a5…`；运行时刻实测
   **8296 行 / `12bb0c5b…`**（编写前 2 小时内又发生 +4 行漂移）。按 preflight §9 规则
   "以文本锚点为准"，脚本在应用时刻对全部 13 个锚点重跑唯一性断言——**13/13 count=1 命中**，
   无锚点失配。hpp/cpp 与 §1 一致（`fa20b4fd…`/815、`ba096f9a…`/587）。
2. **preflight §6 step 13 计数口径（文档 nit）**：`grep -c -F 'polar_coulomb' cpp` 预检写作
   `→ 3`；实测补丁后为 **6 行/6 次**（3 条 `.def` 名 + 3 处 `view2vector(self.polar_coulomb_*)`）。
   "3 条 `.def`" 的意图由脚本探针 `.def("polar_coulomb` → 3（+`self.polar_coulomb_` → 3）验证，
   结论不受影响。
3. **⑦-4.1a 注释 `(N,512)`**（preflight §8-1 已知）：实为 N×2048。按预检要求**原样保留**，
   脚本不改写文档代码一字。
4. **E1/E2（extractor）不属本 13 步**：preflight §1 已判定 E1/E2 已施行、JSON 已于 09:30
   重生成；本次不含提取器改动。
5. **`--apply` 未在真实树执行**：按要求仅实现；已在 `/tmp` 临时副本上自测通过
   （备份 `.bak`、写入、字节一致 + 探针复验、重复应用被幂等标记拒绝、锚点失配打印诊断并中止，
   退出码 2），临时副本已清理。真实树无任何写操作残留。

## 6. 后续（由用户/主线程执行）

dry-run 全绿后：`python3 bench/apply_polar_patches.py --apply`（写入前重跑全部断言并要求
`.bak` 不存在；失败自动回滚）→ 按 `merge_preflight_67.md` §7 重建 → 装载冒烟 → ⑥门 → ⑦门 → 端到端。
