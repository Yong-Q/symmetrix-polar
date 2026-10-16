# ⑧-7 le 反向 slice — 落树代码(定稿文本;2026-10-16)

依据: docs/POLAR_REVERSE_DESIGN.md §2(7)/§3.2 表;docs/module8_patch_final.md §4.1 投递契约。
语义: E_le=Σ_n mlp_out(n);mlp_out=MLP(invar);invar=[dot_q|dot_v];
dot_q=uvu(x1=nf_last,x2=q_up);dot_v=uvu(x1=nf_last,x2=v_up);
q_up=Lin_q(q_in);v_up=Lin_v(P_last);q_in=d_final+d_init(d_init=polar_field_independent_density)。
产出: polar_reverse_seed_terms|=1;∂E/∂d_final→polar_seed_density_adjoint;
∂E/∂d_init→polar_seed_density0_adjoint;∂E/∂P_last→polar_seed_potential_adjoint;
∂E/∂nf_last→polar_inject_node_features_adjoint(末层)。

**实现形态(最终)**: 单块插入 `compute_polar_reverse_forces` 函数体内、lambda 助手区之后、
"// ---------- 2) 种子早退判定" 之前。直接复用函数内既有的 `uvu_reverse`(L1569)与局部
`to_host_vector` 风格;无需 hpp 改动、无需新成员函数、无新视图。le 视图均在转发
eval 中 ensure(extent 守卫兜底)。

## 插入点(唯一): tpp — 在 "// ---------- 2) 种子早退判定" 注释行之前

```cpp
    // ---------- 1.5) [⑧-7] le 反向种子生产 ----------
    // 投递契约 module8_patch_final.md §4.1;语义 POLAR_REVERSE_DESIGN.md §2(7)。
    // 位置 = readout 播种之后、种子判定之前(§4.0-A 接缝);同一 eval ≤1 次。
    // 产出: bit0 置位 + ∂E_le/∂d_final / ∂d_init(同值双分支) + ∂E_le/∂P_last
    //       + n̄f_last(⑧-A 注入末层;atomic_add=init=false 语义)。
    if(polar_le_enabled&&polar_le_add) {
        if(static_cast<int>(polar_le_invar.extent(0))!=num_nodes
            ||static_cast<int>(polar_le_q_up.extent(0))!=num_nodes
            ||static_cast<int>(polar_le_v_up.extent(0))!=num_nodes
            ||static_cast<int>(polar_node_feats_e3nn.extent(0))!=num_nodes)
            throw std::runtime_error(
                "polar le reverse: forward buffers are not resident "
                "(local_electron_energy did not run this evaluation?).");
        const int le_dot_width=static_cast<int>(polar_le_q_mask.extent(0));
        const int le_up_width=static_cast<int>(polar_le_q_up.extent(1));
        // (1) MLP 反向: mlp_out̄≡1(e0 装配含 le 不加权) → invar̄(内部重算+带式)。
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_mlp_out_bar("polar le mlp_out adjoint",num_nodes,1);
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_invar_bar("polar le invar adjoint",num_nodes,2*le_dot_width);
        {
            auto bar=le_mlp_out_bar;
            Kokkos::parallel_for("polar le mlp_out adjoint init",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    bar(node,0)=Precision(1);
                });
        }
        polar_le_mlp.reverse(polar_le_invar,le_mlp_out_bar,le_invar_bar);
        // (2) invar̄ 拆两路。
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_dot_q_bar("polar le dot_q adjoint",num_nodes,le_dot_width);
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_dot_v_bar("polar le dot_v adjoint",num_nodes,le_dot_width);
        {
            auto qb=le_dot_q_bar;
            auto vb=le_dot_v_bar;
            auto ib=le_invar_bar;
            Kokkos::parallel_for("polar le dot adjoint split",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,le_dot_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    qb(node,column)=ib(node,column);
                    vb(node,column)=ib(node,le_dot_width+column);
                });
        }
        // (3) uvu 反向(复用 uvu_reverse;x1=nf_last ⇒ 两路 n̄f 累加)。
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_q_up_bar("polar le q_up adjoint",num_nodes,le_up_width);
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_v_up_bar("polar le v_up adjoint",num_nodes,le_up_width);
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_nf_bar("polar le nf adjoint",num_nodes,polar_feature_width);
        {
            const auto le_to_host=[&](const auto& view) {
                const auto mirror=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                std::vector<typename decltype(mirror)::value_type>
                    values(mirror.size());
                for(std::size_t i2=0;i2<values.size();++i2)
                    values[i2]=mirror.data()[i2];
                return values;
            };
            const std::vector<int> le_q_meta=le_to_host(polar_le_q_meta);
            const std::vector<Precision> le_q_pw=
                le_to_host(polar_le_q_path_weight);
            const std::vector<int> le_v_meta=le_to_host(polar_le_v_meta);
            const std::vector<Precision> le_v_pw=
                le_to_host(polar_le_v_path_weight);
            auto zero_fill=[&](const char* name,
                    Kokkos::View<Precision**,Kokkos::LayoutRight> v) {
                const int columns=static_cast<int>(v.extent(1));
                Kokkos::parallel_for(name,
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,columns}),
                    KOKKOS_LAMBDA(int node,int column) {
                        v(node,column)=Precision(0);
                    });
            };
            zero_fill("polar le q_up adjoint zero",le_q_up_bar);
            zero_fill("polar le v_up adjoint zero",le_v_up_bar);
            zero_fill("polar le nf adjoint zero",le_nf_bar);
            const auto le_nf_source=polar_node_feats_e3nn;
            const auto le_q_up_view=polar_le_q_up;
            const auto le_v_up_view=polar_le_v_up;
            uvu_reverse(le_q_meta,le_q_pw,polar_le_q_weight,
                polar_le_q_mask,le_nf_source,le_q_up_view,
                le_dot_q_bar,le_nf_bar,le_q_up_bar);
            uvu_reverse(le_v_meta,le_v_pw,polar_le_v_weight,
                polar_le_v_mask,le_nf_source,le_v_up_view,
                le_dot_v_bar,le_nf_bar,le_v_up_bar);
        }
        // (4) Lin_q/Lin_v 反向(仿射;权重内建)。
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_q_in_bar("polar le q_in adjoint",num_nodes,8);
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            le_p_last_bar("polar le P_last adjoint",num_nodes,16);
        polar_le_linear_up_q.reverse(le_q_up_bar,le_q_in_bar,
            /*initialize_input_adjoint=*/true);
        polar_le_linear_up_v.reverse(le_v_up_bar,le_p_last_bar,
            /*initialize_input_adjoint=*/true);
        // (5) 种子投递(§4.1;ensure+清零后累加;q_in=d_final+d_init 双分支同值)。
        ensure_view(polar_seed_density_adjoint,
            polar_seed_density_adjoint_storage,num_nodes,8);
        ensure_view(polar_seed_density0_adjoint,
            polar_seed_density0_adjoint_storage,num_nodes,8);
        ensure_view(polar_seed_potential_adjoint,
            polar_seed_potential_adjoint_storage,num_nodes,16);
        ordered_kokkos_deep_copy(polar_seed_density_adjoint,Precision(0));
        ordered_kokkos_deep_copy(polar_seed_density0_adjoint,Precision(0));
        ordered_kokkos_deep_copy(polar_seed_potential_adjoint,Precision(0));
        {
            auto sd=polar_seed_density_adjoint;
            auto sd0=polar_seed_density0_adjoint;
            auto sp=polar_seed_potential_adjoint;
            auto qb=le_q_in_bar;
            auto pb=le_p_last_bar;
            Kokkos::parallel_for("polar le seeds density write",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    sd(node,column)+=qb(node,column);
                    sd0(node,column)+=qb(node,column);
                });
            Kokkos::parallel_for("polar le seeds potential write",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,16}),
                KOKKOS_LAMBDA(int node,int column) {
                    sp(node,column)+=pb(node,column);
                });
        }
        polar_reverse_seed_terms|=1;
        // (6) n̄f_last → ⑧-A 注入末层。
        polar_inject_node_features_adjoint(
            static_cast<int>(states.size())-1,le_nf_bar);
    }
```

## 落树核对清单

1. 锚点唯一性: "// ---------- 2) 种子早退判定" 在 tpp 中 count=1(grep 复核)。
2. 函数内既有名: `uvu_reverse`(L1569)、`ensure_view`/`ordered_kokkos_deep_copy`(文件级)、
   `num_nodes`/`polar_feature_width`/`states`(成员)/`polar_le_*`/`polar_node_feats_e3nn`/
   `polar_seed_*` 成员——全部已存在。
3. R1: 本块无多声明符 auto 混型;R3: 无缺 LayoutRight 的 View 参数(所有形参显式 LayoutRight);
   R4: lambda 体内无成员直引(`le_to_host`/`zero_fill` 都不直引成员;`le_nf_source` 等别名
   均在 lambda 外函数域取好)。
4. `uvu_reverse` 的 mask 语义已核对(前向 output*=mask 在累加后 ⇒ 伴随列乘 mask)。
5. `AffineMLP::reverse(input,out_adj,in_adj)`=内部重算 forward+带式反向,无 tape 失效顾虑。
6. `E3Linear::reverse(out_adj,in_adj,init=true)` 与 §8 mixer 反向同款调用(先例在树内)。
7. ⑧-A 注入为 atomic_add(init=false 语义),置于 readout 播种之后 ✓。
8. 形状: invar=(N,2*dot_width);q_up/v_up=(N,up_width);q_in=(N,8);P_last=(N,16);seeds=(N,8)/(N,16)。
9. 产出后 `polar_reverse_terms_last` 应以全模型 JSON=3(⑥ 位自动 | ⑦ 位本次置)。
10. 预期门: ⑧ 门 post 档 FD≤1e-3(全量解析)、parity≤1e-3(对 python 总力)。
```
