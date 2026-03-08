#!/usr/bin/env python3
"""
OntoGNN —— 本体感知图神经网络 (Ontology-Aware GNN)

架构别名：三明治架构 (Sandwich Architecture)
  面包片①  Stage 0: 特征投影     (Input Projections)
  ─────────────────────────────────────────────────────
  馅料①    Stage 1: 语义编码     (Semantic Encoding)
            业务实体之间互相交流：Shift ↔ Shift (same_day 边)
  馅料②    Stage 2: 语义注入     (Semantic Injection)
            业务特征聚合到 variable：Employee/Shift → variable
  ─────────────────────────────────────────────────────
  面包片②  Stage 3: 数学推理     (Mathematical Reasoning) [下半部实现]
            variable ↔ constraint 二分图消息传递
  ─────────────────────────────────────────────────────
  芯       Stage 4: 决策头       (Scoring Head) [下半部实现]
            每个 variable 输出一个分数，越高越优先被 SCIP 选为分支变量

Author: OntoBranch-2026 Team
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# 常量：与数学层对接的固定维度
# ─────────────────────────────────────────────────────────────────────────────

# Ecole NodeBipartite 观测中 variable_features 的固定列数（19 列）
# 在 Stage 3 数学推理中会用到，这里提前声明以便 __init__ 分配正确大小的 Linear
_ECOLE_VAR_DIM  = 19   # variable  原始观测维度
_ECOLE_CON_DIM  = 5    # constraint 原始观测维度（row_features）


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 辅助：语义注入聚合器
# ─────────────────────────────────────────────────────────────────────────────

class _MeanAggregator(nn.Module):
    """
    对一个节点收到的所有语义消息做均值聚合。

    之所以不直接用 SAGEConv，是因为语义注入不需要 SAGEConv 内部那套
    "邻居均值 concat 自身" 的逻辑——我们希望自己精确控制拼接顺序，以便
    在 forward 里把多种类型的实体消息 concat 成一个长向量。

    输入：
      msgs: Tensor [N_var, hidden_dim]   ← 已经过 Linear 变换的聚合消息
    输出：
      same tensor（这里只是占位；真正聚合在 HeteroConv 内部完成）
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class OntoGNN(nn.Module):
    """
    三明治架构 GNN，上半部（Stage 0-2）实现。

    参数
    ----
    hidden_dim : int
        统一隐层宽度。所有节点类型投影后都对齐到此维度，
        这样 Stage 2 的 concat 结果维度才是确定的。
    num_semantic_layers : int
        Stage 1（语义编码）重复堆叠的 HeteroConv 层数。
        1~2 层通常已足够让同日班次完成邻域感知。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_semantic_layers: int = 1,
    ):
        super().__init__()

        self.hidden_dim          = hidden_dim
        self.num_semantic_layers = num_semantic_layers

        # ─────────────────────────────────────────────────────────────────────
        # Stage 0: 特征投影字典 (LazyLinear 延迟推断输入维度)
        #
        # 设计动机（见《数据宪法》领域无关性原则）：
        #   不同业务场景、不同节点类型的 features 维度各不相同。
        #   普通 nn.Linear(in, out) 需要在代码里硬编码 in_features，
        #   而 nn.LazyLinear(out) 在收到第一批真实数据时自动推断 in_features。
        #   对模型代码而言，只需在 forward 里把特征喂进去，无需与 Generator 耦合。
        #
        # 注意：LazyLinear 在 .to(device) 后、第一次 forward 前处于"未初始化"状态。
        #       请确保在调用 optimizer 前先做一次 forward（哪怕是 dummy 数据）。
        #
        # 此处预先注册四种已知节点类型的投影头。
        # 若未来引入新类型（如 "Location"），只需在这里追加一行，或改用动态注册。
        # ─────────────────────────────────────────────────────────────────────
        self.proj = nn.ModuleDict({
            # 业务实体节点（语义层）：来自 JSON nodes，维度由 Generator 决定
            "Employee"   : nn.LazyLinear(hidden_dim),
            "Shift"      : nn.LazyLinear(hidden_dim),
            # 数学层节点：来自 Ecole 观测，维度固定
            "variable"   : nn.LazyLinear(hidden_dim),
            "constraint" : nn.LazyLinear(hidden_dim),
        })

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1: 语义编码层列表
        #
        # 使用 GATConv（图注意力网络）替代 SAGEConv（均值聚合）。
        #
        # GATConv 的核心升级：
        #   SAGEConv 对所有邻居一视同仁（权重均等）。
        #   GATConv 为每条边学习一个注意力系数 α_ij ∈ (0,1)：
        #     α_ij = softmax_j( LeakyReLU(aᵀ · [W·hᵢ || W·hⱼ]) )
        #   聚合结果 = Σⱼ α_ij · W·hⱼ
        #   模型可以自动发现"冲突更严重的班次对"，给它更高权重（学会区分邻居重要性）。
        #
        # heads=4, concat=False：
        #   使用 4 个注意力头并行计算，最后取均值（concat=False）。
        #   输出维度 = hidden_dim（不随 heads 数倍增），便于后续层衔接。
        #   若 concat=True，输出维度 = hidden_dim * heads，需同步调整后续层。
        #
        # dropout=0.1：
        #   对注意力系数随机置零，防止少数"枢纽"边过拟合。
        #
        # add_self_loops=False：
        #   HeteroConv 异构图场景中，src 和 dst 节点属于同一类型（Shift），
        #   但 PyG 的自环添加逻辑在异构图中需要关闭以避免索引错误。
        # ─────────────────────────────────────────────────────────────────────
        self.semantic_convs = nn.ModuleList([
            HeteroConv(
                {
                    # Shift ↔ Shift 双向（JSON 生成器已输出双向边）
                    ("Shift", "same_day", "Shift"): GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=4,              # 4 个并行注意力头
                        concat=False,         # 头输出取均值，维度保持 hidden_dim
                        dropout=0.1,          # 注意力 dropout
                        add_self_loops=False, # 异构图场景关闭自环
                    ),
                    # 未来可扩展，例如：
                    # ("Employee", "colleague", "Employee"): GATConv(
                    #     hidden_dim, hidden_dim, heads=4, concat=False
                    # ),
                },
                aggr="sum",
            )
            for _ in range(num_semantic_layers)
        ])

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2: 语义注入的"接受器"线性层
        #
        # 每种业务实体类型会通过反向 mapped_to 边把特征注入到 variable。
        # 注入消息 = 对方特征过一个线性变换（维度保持 hidden_dim）。
        # 最后把所有来源的注入消息 concat 起来 + variable 自身特征，
        # 共同投影成最终的 variable 表示。
        #
        # inject_proj  : 每种业务类型对应一个线性变换，做消息传递中的 "Φ(m)" 变换
        # inject_fuse  : 把 concat 后的长向量压缩回 hidden_dim
        #
        # 注意 inject_fuse 的输入维度：
        #   variable 自身 hidden_dim
        # + 来自 Employee 的 hidden_dim
        # + 来自 Shift    的 hidden_dim
        # = 3 * hidden_dim
        #
        # 若将来注入来源增加（例如 VRP 里有 Vehicle 和 Location），
        # inject_fuse 的输入维度也需要相应调整，可以改成 LazyLinear。
        # ─────────────────────────────────────────────────────────────────────
        self.inject_proj = nn.ModuleDict({
            "Employee" : nn.Linear(hidden_dim, hidden_dim),
            "Shift"    : nn.Linear(hidden_dim, hidden_dim),
        })

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2 注意力机制：轻量级点积注意力（Scaled Dot-Product Attention）
        #
        # 原来的均值聚合问题：
        #   一个 variable 可能关联多个 Shift（批量排班），但不同 Shift
        #   对这个变量的重要程度并不相同——权重应根据匹配度动态决定。
        #
        # 新设计（Q-K-V 三路分离）：
        #   V（值向量）= inject_proj[etype](h_etype)   ← 决定"带来什么信息"
        #   Q（查询）  = inject_attn_q[etype](h_variable[i])  ← variable 问"我需要什么"
        #   K（键）    = inject_attn_k[etype](h_etype[j])     ← 实体答"我能提供什么"
        #
        #   注意力得分 e_ij = (Q_i · K_j) / sqrt(H)
        #   对每个 variable 的所有邻居做 softmax → α_ij ∈ (0,1)
        #   聚合：agg_msg[i] = Σⱼ α_ij · V_j
        #
        # 为什么 Q/K 独立于 inject_proj（V）？
        #   V 负责内容（聚合后的语义），Q/K 负责选择（打分），职责分离，
        #   梯度互不干扰，训练更稳定（参考 Transformer 的 Q-K-V 分离设计）。
        #
        # 注：bias=False 是 Q/K 的惯例（位置无关的打分不需要偏置）。
        # ─────────────────────────────────────────────────────────────────────
        self.inject_attn_q = nn.ModuleDict({   # variable 侧：生成查询向量 Q
            "Employee" : nn.Linear(hidden_dim, hidden_dim, bias=False),
            "Shift"    : nn.Linear(hidden_dim, hidden_dim, bias=False),
        })
        self.inject_attn_k = nn.ModuleDict({   # 实体侧：生成键向量 K
            "Employee" : nn.Linear(hidden_dim, hidden_dim, bias=False),
            "Shift"    : nn.Linear(hidden_dim, hidden_dim, bias=False),
        })

        # 缩放因子（固定值，不作为参数）：防止点积过大导致 softmax 梯度消失
        import math
        self._attn_scale = math.sqrt(hidden_dim)

        # concat(variable_self, emp_msg, shift_msg) → hidden_dim
        # 这里用 LazyLinear 允许未来注入来源数量动态变化
        self.inject_fuse = nn.LazyLinear(hidden_dim)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3: 数学推理 (数学层，下半部实现)
        # ─────────────────────────────────────────────────────────────────────
        # TODO: Stage 3 将在下半部添加
        #   variable ↔ constraint 二分图双向消息传递
        #   参考：Gasse et al. (2019) 的 GCN 二分图层

        # ─────────────────────────────────────────────────────────────────────
        # Stage 4: 决策头 (下半部实现)
        # ─────────────────────────────────────────────────────────────────────
        # TODO: Stage 4 将在下半部添加
        #   输出 shape: [num_variables]，值越高该变量越优先被选作分支变量

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 0 辅助方法：投影所有类型的节点
    # ─────────────────────────────────────────────────────────────────────────

    def _project_all(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        把 x_dict 中每种节点的原始特征投影到统一的 hidden_dim 空间。

        输入  x_dict["Employee"] : Tensor[E, feat_emp]   (feat_emp 由 Generator 决定)
              x_dict["Shift"]    : Tensor[S, feat_shift]
              x_dict["variable"] : Tensor[N, 19]          (Ecole 固定)
              x_dict["constraint"]: Tensor[C, 5]           (Ecole 固定)
        输出  h_dict["Employee"] : Tensor[E, hidden_dim]
              h_dict["Shift"]    : Tensor[S, hidden_dim]
              h_dict["variable"] : Tensor[N, hidden_dim]
              h_dict["constraint"]: Tensor[C, hidden_dim]

        只投影 self.proj 中已注册的类型，图里出现但未注册的类型直接跳过。
        这样未来新增类型不会导致 KeyError。
        """
        h_dict: Dict[str, Tensor] = {}
        for ntype, x in x_dict.items():
            if ntype in self.proj:
                # ReLU 激活：引入非线性，让投影不退化为纯线性变换
                h_dict[ntype] = F.relu(self.proj[ntype](x))
            else:
                # 未注册的类型：原样透传（维度不统一，Stage 2/3 可能无法使用）
                h_dict[ntype] = x
        return h_dict

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: 语义编码
    # ─────────────────────────────────────────────────────────────────────────

    def _semantic_encoding(
        self,
        h_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        """
        在业务实体之间做多跳消息传递（如 Shift ↔ Shift same_day 边）。

        参数
        ----
        h_dict          : 各节点类型的投影后特征，Tensor[num_nodes, hidden_dim]
        edge_index_dict : PyG 边索引字典，key = (src_type, rel, dst_type)

        返回
        ----
        更新后的 h_dict（非 variable/constraint 部分可能被更新）

        防崩溃设计：
          HeteroConv 只处理 edge_index_dict 里真实存在的边类型。
          如果某张图里没有 same_day 边（例如只有一个班次），
          HeteroConv 会跳过该卷积核，不会报错，Shift 特征维持不变。
          这里我们额外用 .get() 防止 h_dict 中键不存在的情况。
        """
        for conv in self.semantic_convs:
            # HeteroConv.forward 签名：
            #   forward(x_dict, edge_index_dict) → out_dict
            # out_dict 只包含作为消息接收方出现过的节点类型。
            # 若某类型没有接收到任何消息（e.g. 图里无 same_day 边），
            # 该类型不会出现在 out_dict 里。
            out = conv(h_dict, edge_index_dict)

            # 用更新结果覆盖 h_dict，未被更新的类型保持原值
            for ntype, h_new in out.items():
                # Stage 1 残差连接：新表示 = F(旧表示) + 旧表示
                # 好处：梯度流更稳，层数越多越不容易退化
                if ntype in h_dict and h_dict[ntype].shape == h_new.shape:
                    h_dict[ntype] = F.relu(h_new) + h_dict[ntype]
                else:
                    # 形状不一致（理论上不应发生）：直接覆盖
                    h_dict[ntype] = F.relu(h_new)

        return h_dict

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: 语义注入
    # ─────────────────────────────────────────────────────────────────────────

    def _semantic_injection(
        self,
        h_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        """
        将业务实体的语义特征"注入"到 variable 节点。

        原理（以 Employee → variable 为例）：
          边 ('variable', 'mapped_to', 'Employee') 的 edge_index:
            row 0 = [0, 1, 2, ...,  299]   ← variable 的局部索引
            row 1 = [0, 0, 1, ...,   14]   ← Employee 的局部索引

          我们需要方向反过来（Employee → variable），所以先把 edge_index 翻转：
            src = edge_index[1]  ← Employee 局部索引
            dst = edge_index[0]  ← variable 局部索引

          然后对每个 variable v，用注意力加权聚合所有入边的实体特征：
            e_vj    = (Q_v · K_j) / sqrt(H)          ← 点积打分
            α_vj    = softmax_v(e_vj)                 ← 按 variable 分组归一化
            msg_emp[v] = Σⱼ α_vj · V_j               ← 加权求和

          同理得 msg_shift[v]。

          最后：
            h_variable_new[v] = inject_fuse(
                concat(h_variable[v], msg_emp[v], msg_shift[v])
            )
            维度： hidden_dim + hidden_dim + hidden_dim = 3 * hidden_dim → hidden_dim

        参数
        ----
        h_dict          : Stage 1 输出的节点特征字典
        edge_index_dict : PyG 边索引字典

        返回
        ----
        h_dict（variable 部分被更新，其余类型不变）

        防崩溃设计：
          - 若该图里没有某种 mapped_to 边（未来某些场景变量可能只关联一种实体），
            对应的注入消息用全零填充，不影响其他来源的注入。
          - variable 节点数 N_var 从 h_dict["variable"] 的行数动态读取。
        """
        # 当前图的 variable 数量（动态读取，不硬编码）
        N_var   = h_dict["variable"].shape[0]
        device  = h_dict["variable"].device
        H       = self.hidden_dim

        # 收集来自各业务实体的注入消息
        # inject_msgs: 最终会 concat 的消息列表，顺序固定（variable自身在最前）
        inject_msgs = [h_dict["variable"]]   # shape: [N_var, H]

        # 遍历所有已注册的注入来源类型
        for etype, proj_layer in self.inject_proj.items():
            # 构造边类型三元组的键（variable → etype 方向在 JSON 中的表示）
            edge_key = ("variable", "mapped_to", etype)

            if edge_key not in edge_index_dict:
                # ——— 防崩溃分支 ———
                # 该图没有 variable→etype 的边（可能该场景不涉及此类型实体）
                # 用全零占位，保持 concat 维度一致
                zero_msg = torch.zeros(N_var, H, device=device)
                inject_msgs.append(zero_msg)
                continue

            edge_index = edge_index_dict[edge_key]
            # edge_index shape: [2, num_edges]
            #   edge_index[0] = variable 的局部索引（src in JSON，dst in 注入）
            #   edge_index[1] = etype 节点的局部索引（dst in JSON，src in 注入）

            # 语义注入方向：etype → variable，所以：
            src_indices = edge_index[1]   # etype 局部索引，shape: [num_edges]
            dst_indices = edge_index[0]   # variable 局部索引，shape: [num_edges]

            # ── V：值向量（最终聚合的内容）──────────────────────────────────
            # inject_proj 把实体特征变换为"消息内容"
            # h_etype_val shape: [N_etype, H]
            h_etype_val   = proj_layer(h_dict[etype])          # [N_etype, H]
            vals_per_edge = h_etype_val[src_indices]           # [num_edges, H]

            # ── Q：查询向量（variable 侧）───────────────────────────────────
            # Q_all shape:      [N_var, H]
            # Q_per_edge shape: [num_edges, H]  ← 取出每条边对应的 variable
            Q_all      = self.inject_attn_q[etype](h_dict["variable"])  # [N_var, H]
            Q_per_edge = Q_all[dst_indices]                              # [num_edges, H]

            # ── K：键向量（实体侧）──────────────────────────────────────────
            # K_all shape:      [N_etype, H]
            # K_per_edge shape: [num_edges, H]  ← 取出每条边对应的实体
            K_all      = self.inject_attn_k[etype](h_dict[etype])       # [N_etype, H]
            K_per_edge = K_all[src_indices]                              # [num_edges, H]

            # ── 点积打分 e_ij = (Q_i · K_j) / sqrt(H) ──────────────────────
            # einsum 'ij,ij->i'：对两个 [num_edges, H] 矩阵逐行点积 → [num_edges]
            scores_per_edge = torch.einsum(
                "ij,ij->i", Q_per_edge, K_per_edge
            ) / self._attn_scale                                         # [num_edges]

            # ── Sparse Softmax：对每个 variable 的入边独立归一化 ─────────────
            #
            # 标准 softmax 要求"按组"做，这里每组 = 一个 variable 的所有入边。
            # 用 scatter_reduce_ 实现朴素版，避免引入 torch_scatter 外部依赖。
            #
            # 步骤①：求每个 variable 入边得分的最大值（数值稳定 trick）
            max_scores = torch.full((N_var,), float("-inf"), device=device)
            max_scores.scatter_reduce_(
                0, dst_indices, scores_per_edge,
                reduce="amax", include_self=True
            )
            # 取出每条边对应的组最大值，用于减法稳定
            max_per_edge = max_scores[dst_indices]                       # [num_edges]

            # 步骤②：exp(e_ij - max_i)，避免数值上溢
            exp_scores = torch.exp(scores_per_edge - max_per_edge)       # [num_edges]

            # 步骤③：按 variable 分组累加 exp，clamp 避免除以零
            sum_exp = torch.zeros(N_var, device=device)
            sum_exp.index_add_(0, dst_indices, exp_scores)
            sum_exp_per_edge = sum_exp[dst_indices].clamp(min=1e-9)      # [num_edges]

            # 步骤④：归一化得到注意力权重 α_ij
            alpha = exp_scores / sum_exp_per_edge                        # [num_edges]

            # ── 加权聚合：agg_msg[i] = Σⱼ α_ij · V_j ───────────────────────
            # alpha unsqueeze 成 [num_edges, 1] 再与 [num_edges, H] 广播相乘
            weighted_msgs = vals_per_edge * alpha.unsqueeze(-1)          # [num_edges, H]

            agg_msg = torch.zeros(N_var, H, device=device)
            agg_msg.index_add_(0, dst_indices, weighted_msgs)            # [N_var, H]

            inject_msgs.append(agg_msg)

        # concat 所有注入消息：[N_var, H] × (1 + num_inject_types) → [N_var, H*(1+K)]
        # 例：K=2 时，cat 后 shape = [N_var, 3*H]
        fused = torch.cat(inject_msgs, dim=-1)   # [N_var, (1+K)*H]

        # 通过 inject_fuse 压缩回 hidden_dim
        # inject_fuse 是 LazyLinear，第一次调用时自动探测输入维度 (1+K)*H
        h_dict["variable"] = F.relu(self.inject_fuse(fused))  # [N_var, H]

        return h_dict

    # ─────────────────────────────────────────────────────────────────────────
    # 前向传播主函数
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, data: HeteroData) -> Tensor:
        """
        前向传播骨架，串联 Stage 0-2（Stage 3-4 为 stub）。

        参数
        ----
        data : PyG HeteroData，由 OntologyGraphBuilder.build() 生成

        返回
        ----
        scores : Tensor[num_variables]
            每个决策变量的分支优先级分数（值越高，越先被 SCIP 选为分支变量）。
            Stage 4 实现前，这里暂时返回 variable 特征的 L2 范数作为占位分数。
        """
        # 从 HeteroData 提取特征字典和边索引字典
        # x_dict       : {node_type: Tensor[num_nodes, feat_dim]}
        # edge_index_dict: {(src,rel,dst): Tensor[2, num_edges]}
        x_dict          = data.x_dict
        edge_index_dict = data.edge_index_dict

        # ── Stage 0: 特征投影 ────────────────────────────────────────────────
        # 所有节点类型统一投影到 hidden_dim 空间
        # 输入各节点特征维度不同 → 输出全部变为 [num_nodes, hidden_dim]
        h_dict = self._project_all(x_dict)

        # ── Stage 1: 语义编码 ────────────────────────────────────────────────
        # 业务实体内部消息传递（如 Shift same_day Shift）
        # 只影响语义层节点（Employee, Shift 等），不涉及 variable/constraint
        h_dict = self._semantic_encoding(h_dict, edge_index_dict)

        # ── Stage 2: 语义注入 ────────────────────────────────────────────────
        # 业务实体特征 → variable 节点
        # 只影响 variable，语义层节点不变
        h_dict = self._semantic_injection(h_dict, edge_index_dict)

        # ── Stage 3: 数学推理 ────────────────────────────────────────────────
        # TODO: variable ↔ constraint 二分图消息传递（下半部实现）
        # h_dict = self._math_reasoning(h_dict, edge_index_dict)

        # ── Stage 4: 决策头 ──────────────────────────────────────────────────
        # TODO: 输出每个 variable 的分支优先级分数（下半部实现）
        # scores = self.scoring_head(h_dict["variable"])   # [N_var]
        # return scores

        # ── 占位输出（Stage 3-4 实现前的临时返回）────────────────────────────
        # 用 variable 特征向量的 L2 范数作为伪分数，保证 shape=[N_var] 正确
        pseudo_scores = h_dict["variable"].norm(dim=-1)   # [N_var]
        return pseudo_scores
