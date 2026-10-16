#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""polar_lint — symmetrix-xl polar 代码静态护栏(编译前全类扫描, 2026-10-15)。

背景: v16-v17 的 6 类编译/运行期错误(每类只被编译暴露一次, 每轮 20-40 分钟)。
本脚本把全部已知错误类变成正则/结构规则, 改动后编译前一次全扫, "一次清一类"。

规则(全部源自实战教训, 每条带案例锚点):
  R1 `auto a=X, b=Y;` 多声明符类型不一致(v16: View* vs View**)
  R2 `Kokkos::realloc(tag, view, storage, ...)` 五参误用(v17b: 应 ensure_view)
  R3 lambda 参数 `Kokkos::View<Precision**>` 缺 LayoutRight(v17c: 拷贝构造失败)
  R4 KOKKOS_LAMBDA 体内直引成员视图(隐式 this→设备读主机内存; v17d 运行期崩)
  R5 KOKKOS_LAMBDA 体内出现 std::vector/std::string 等主机容器(v17b 泛型 lambda 之同类)
  R6 设备 lambda 内未加别名的 `polar_*` 成员标量/视图调用(同 R4 广义)
用法: python3 bench/polar_lint.py [tpp路径]   # 默认引擎树 tpp; 退出码=发现数(>0 即拦截)
"""
import re
import sys

DEFAULT_TPP = ("/workspace/group/shared/bza/external-repos/symmetrix-xl/"
               "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp")


def lambda_spans(lines):
    """返回 [(start_line, end_line)] 覆盖所有 KOKKOS_LAMBDA 体(brace 配平)。"""
    spans = []
    for idx, line in enumerate(lines):
        for m in re.finditer(r"KOKKOS_LAMBDA", line):
            depth = 0
            i, col = idx, m.end()
            while i < len(lines):
                while col < len(lines[i]):
                    ch = lines[i][col]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            spans.append((idx, i))
                            break
                    col += 1
                else:
                    i += 1
                    col = 0
                    continue
                break
    return spans


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TPP
    src = open(path).read()
    lines = src.split("\n")
    findings = []

    # R1: 多声明符 auto 类型不一致 —— 从 hpp 构建"视图形状注册表"后按名判定
    import os as _os
    hpp_path = _os.path.join(_os.path.dirname(path), "mace_nonlinear_kokkos.hpp")
    view_shape = {}
    if _os.path.exists(hpp_path):
        hpp = open(hpp_path).read()
        # 抓 "Kokkos::View<...> a,b,c;" 声明块, 判定 ** → view2d, * → view1d
        for m in re.finditer(r"Kokkos::View<([^;>]*(?:>[^;]*)*?)>\s*([A-Za-z0-9_,\s]+);", hpp):
            dtype_block = m.group(1)
            names = [n.strip() for n in m.group(2).split("\n")]
            flat = []
            for chunk in m.group(2).replace("\n", " ").split(","):
                n = chunk.strip()
                if n and re.match(r"^[A-Za-z_]\w*$", n):
                    flat.append(n)
            shape = "view2d" if "**" in dtype_block else ("view1d" if "*" in dtype_block else "other")
            for n in flat:
                view_shape[n] = shape
    for idx, line in enumerate(lines):
        m = re.match(r"\s*auto\s+(\w+)\s*=\s*([^,;]+)(,.+)?;\s*$", line)
        if not m or not m.group(3):
            continue
        decls = re.split(r",(?![^()]*\))", line[line.index("auto") + 4:].rstrip(";"))
        shapes = set()
        for d in decls:
            rhs = d.split("=", 1)[1].strip() if "=" in d else ""
            if rhs in view_shape:
                shapes.add(view_shape[rhs])
        if len(shapes) > 1:
            findings.append(("R1", idx + 1, "多声明符 auto 形状不一致: " + line.strip()[:90]))

    # R2: realloc 五参误用(view, storage, ...)
    for idx, line in enumerate(lines):
        if "realloc(Kokkos::WithoutInitializing," in line:
            nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
            if nxt.count(",") >= 2 and "_storage," in nxt:
                # 形如 "view,storage,d0," —— 编译不过
                head = nxt.split(",")
                if len(head) >= 2 and "_storage" in head[1]:
                    findings.append(("R2", idx + 1, "realloc 五参误用(应 ensure_view): " + nxt.strip()[:80]))

    # R3: lambda 参数缺 LayoutRight
    for idx, line in enumerate(lines):
        if re.search(r"Kokkos::View<Precision\*\*>\s*\w", line):
            findings.append(("R3", idx + 1, "View<Precision**> 参数缺 LayoutRight: " + line.strip()[:80]))

    # 预先收集全部局部名(任何 auto/type 声明), 用于 R4/R6 成员判定
    local_names = set(re.findall(r"\b(?:auto|const\s+\w+|int|bool|double)\s+(\w+)\s*=", src))
    local_names |= set(re.findall(r"\b(\w+)\s*=\s*\w+\s*;", src))  # 赋值型别名兜底

    spans = lambda_spans(lines)
    for start, end in spans:
        body = "\n".join(lines[start:end + 1])
        # R4/R6: 体内 polar_* 直引(后接 ( [ .) 且不是本函数已知局部名
        for m in re.finditer(r"\b(polar_[a-z_0-9]+)\s*(?=[\(\[\.])", body):
            name = m.group(1)
            if name in local_names:
                continue
            ln = start + body[:m.start()].count("\n") + 1
            findings.append(("R4", ln, f"lambda 内直引成员/未别名 {name}"))
        # R5: 主机容器
        for m in re.finditer(r"std::(vector|string|array)<", body):
            ln = start + body[:m.start()].count("\n") + 1
            findings.append(("R5", ln, "lambda 体内出现主机容器 " + m.group(0)))

    if not findings:
        print(f"polar_lint: CLEAN ({path.split('/')[-1]})")
        return 0
    for rule, ln, msg in sorted(findings):
        print(f"polar_lint: [{rule}] L{ln}: {msg}")
    print(f"polar_lint: {len(findings)} 处发现 (退出码=发现数)")
    return min(len(findings), 100)


if __name__ == "__main__":
    raise SystemExit(main())
