#!/usr/bin/env python3
"""
OntoGNN v3.0 —— 终极静态模型 (The Universal Brain)

========================== 架构核心思想 ==========================
  本模型拥有 **固定的参数结构**，但具备 **领域泛化力**。
  不再接受 metadata 参数，不再动态注册任何层。
  所有权重在 __init__ 时一次性确定，永远不会因为输入领域不同
  而改变模型的参数结构。

  这是通过 USG（通用语义图）范式实现的：
    - 所有业务节点统一为 "entity" 类型
    - 所有语义边统一为 "relates_to" 关系
    - 图拓扑固定为 3 节点 + 3 边

架构别名：三明治架构 / The Universal Brain

  Stage 0: 特征投影     (Feature Projection)
     3 个绝对静态的投影层：
       proj_var(19, H)           — variable 节点
       proj_con(5, H)            — constraint 节点
       proj_ent(GLOBAL_ENT_DIM, H) — entity 节点 (128 → H)
     

  Stage 1: 语义编码     (Semantic Encoding)
     GATConv 处理 ("entity", "relates_to", "entity") 边
     entity 节点间的信息传递 + 残差连接

  Stage 2: 语义注入     (Semantic Injection)
     处理 ("variable", "mapped_to", "entity") 的反向传递（entity → variable）
     纯手工 Q-K-V 稀疏注意力（不依赖任何第三方 attention 模块）

  Stage 3: 数学推理     (Mathematical Reasoning)
     variable ↔ constraint 二分图双向消息传递
     GATConv 二分图模式 + 残差连接

  Stage 4: 决策头       (Scoring Head)
     MLP: Linear → ReLU → Linear
     输出 [num_variables] 的一维分数张量

"""

from __future__ import annotations
import math
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv   #注意力
from src.generator.base_generator import GLOBAL_ENT_DIM


# ═════════════════════════════════════════════════════════════════════
#  Ecole 标准特征维度常量
# ═════════════════════════════════════════════════════════════════════
_VAR_DIM = 19    # Ecole NodeBipartite variable_features 的固定维度
_CON_DIM = 5     # Ecole NodeBipartite row_features 的固定维度


class OntoGNN(nn.Module):
    """
    终极静态三明治架构 GNN —— 参数结构完全固定，支持领域泛化。
    ★ 不出现任何领域特化的字符串或条件分支 ★

    参数
    ----
    hidden_dim : int
        统一隐层宽度 H。所有节点类型投影后对齐到此维度。
    num_semantic_layers : int
        Stage 1 语义编码的 GATConv 层数。
    num_math_layers : int
        Stage 3 数学推理的 variable↔constraint 消息传递轮数。
    gat_heads : int
        GATConv 的注意力头数（concat=False，输出维度 = H）。
    gat_dropout : float
        GATConv 的注意力 dropout 比率。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_semantic_layers: int = 1,
        num_math_layers: int = 2,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
    ):
        super().__init__()

        H = hidden_dim
        self.hidden_dim = H

        # ─────────────────────────────────────────────────────────────
        # Stage 0: 特征投影 (Feature Projection)
        #
        # 3 个绝对静态的 Linear 层，维度在编译期就完全确定：
        #   proj_var : [N_var, 19]  → [N_var, H]
        #   proj_con : [N_con, 5]   → [N_con, H]
        #   proj_ent : [N_ent, 128] → [N_ent, H]
        #
        # ★ 彻底杜绝 LazyLinear ★
        # ─────────────────────────────────────────────────────────────
        self.proj_var = nn.Linear(_VAR_DIM, H)
        self.proj_con = nn.Linear(_CON_DIM, H)
        self.proj_ent = nn.Linear(GLOBAL_ENT_DIM, H)

        # ─────────────────────────────────────────────────────────────
        # Stage 1: 语义编码 (Semantic Encoding)
        #
        # 单一类型的 GATConv，专门处理：
        #   ("entity", "relates_to", "entity")
        #
        # 多层堆叠，每层带残差连接。
        # heads × concat=False → 多头并行取均值，输出 = H
        # add_self_loops=True  → entity 同构图场景可安全使用自环
        # ─────────────────────────────────────────────────────────────
        self.semantic_convs = nn.ModuleList([
            GATConv(
                H, H,
                heads=gat_heads,
                concat=False,
                dropout=gat_dropout,
                add_self_loops=True,
            )
            for _ in range(num_semantic_layers)
        ])

        # ─────────────────────────────────────────────────────────────
        # Stage 2: 语义注入 (Semantic Injection)
        #
        # 处理 ("variable", "mapped_to", "entity") 边的反向传递：
        #   entity → variable（entity 是信息源，variable 是接收方）
        #
        # 纯手工 Q-K-V 稀疏注意力（不使用 GATConv / HeteroConv）：
        #   Q = W_q × h_variable     [N_var, H]  查询向量
        #   K = W_k × h_entity       [N_ent, H]  键向量
        #   V = W_v × h_entity       [N_ent, H]  值向量
        #
        #   对每条 mapped_to 边 (var_i, ent_j)：
        #     score = (Q[var_i] · K[ent_j]) / √H
        #   按 variable 分组做 sparse softmax，加权聚合 V
        #
        # 最终：concat(h_variable, agg) → Linear → h_variable_new
        # ─────────────────────────────────────────────────────────────
        self.inject_q = nn.Linear(H, H, bias=False)
        self.inject_k = nn.Linear(H, H, bias=False)
        self.inject_v = nn.Linear(H, H)
        self._attn_scale = math.sqrt(H)
        # 融合层：concat(h_variable, agg) = 2H → H
        self.inject_fuse = nn.Linear(2 * H, H)

        # ─────────────────────────────────────────────────────────────
        # Stage 3: 数学推理 (Mathematical Reasoning)
        #
        # 经典 L2B 二分图消息传递（参考 Gasse et al. 2019）：
        #   每轮两步：
        #     ① v2c: variable → constraint
        #     ② c2v: constraint → variable
        #   GATConv 二分图模式：forward(x=(x_src, x_dst), edge_index)
        #   堆叠 num_math_layers 轮，每步带残差连接。
        # ─────────────────────────────────────────────────────────────
        self.math_v2c_convs = nn.ModuleList([
            GATConv(
                H, H,
                heads=gat_heads, concat=False,
                dropout=gat_dropout, add_self_loops=False,
            )
            for _ in range(num_math_layers)
        ])
        self.math_c2v_convs = nn.ModuleList([
            GATConv(
                H, H,
                heads=gat_heads, concat=False,
                dropout=gat_dropout, add_self_loops=False,
            )
            for _ in range(num_math_layers)
        ])

        # ─────────────────────────────────────────────────────────────
        # Stage 4: 决策头 (Scoring Head)
        #
        # MLP：H → H → 1
        # 输出 shape = [N_var]（不加 Sigmoid，交给外部 loss 决定）
        # ─────────────────────────────────────────────────────────────
        self.scoring_head = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    # ─────────────────────────────────────────────────────────────────
    # Stage 0: 特征投影
    # ─────────────────────────────────────────────────────────────────

    def _project_all(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        将 3 种节点的原始特征投影到统一的 hidden_dim 空间。

        输入维度固定：
          x_dict["variable"]   : [N_var, 19]
          x_dict["constraint"] : [N_con, 5]
          x_dict["entity"]     : [N_ent, 128]

        输出维度统一：
          h_dict[*]            : [N_*, H]
        """
        return {
            "variable":   F.relu(self.proj_var(x_dict["variable"])),
            "constraint": F.relu(self.proj_con(x_dict["constraint"])),
            "entity":     F.relu(self.proj_ent(x_dict["entity"])),
        }

    # ─────────────────────────────────────────────────────────────────
    # Stage 1: 语义编码
    # ─────────────────────────────────────────────────────────────────

    def _semantic_encoding(
        self,
        h_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        """
        entity 节点之间的多跳消息传递。

        只处理 ("entity", "relates_to", "entity") 一种边。
        残差连接：h_new = ReLU(conv(h, ei)) + h_old

        若图中无 relates_to 边（edge_index 为空），GATConv 的
        add_self_loops=True 确保 entity 节点仍能获得自身信息。
        """
        edge_key = ("entity", "relates_to", "entity")
        ei = edge_index_dict.get(edge_key, None)

        if ei is None:
            return h_dict

        h_ent = h_dict["entity"]
        for conv in self.semantic_convs:
            h_ent_new = conv(h_ent, ei)
            h_ent = F.relu(h_ent_new) + h_ent   # 残差连接

        h_dict["entity"] = h_ent
        return h_dict

    # ─────────────────────────────────────────────────────────────────
    # Stage 2: 语义注入（手工 Q-K-V 稀疏注意力）
    # ─────────────────────────────────────────────────────────────────

    def _semantic_injection(
        self,
        h_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        """
        将 entity 的语义特征注入到 variable 节点。

        通过 ("variable", "mapped_to", "entity") 边的 **反向** 传递：
          信息流方向：entity → variable
          edge_index[0] = variable 局部索引
          edge_index[1] = entity 局部索引
          反向：src = entity (edge_index[1]), dst = variable (edge_index[0])

        Q-K-V 稀疏注意力实现：
          1. Q = W_q × h_variable, K = W_k × h_entity, V = W_v × h_entity
          2. 按边计算 score = (Q[dst] · K[src]) / √H
          3. 按 variable 分组做 sparse softmax（数值稳定版）
          4. 加权聚合 V → agg [N_var, H]
          5. fused = concat(h_variable, agg) → Linear → h_variable_new
        """
        edge_key = ("variable", "mapped_to", "entity")
        ei = edge_index_dict.get(edge_key, None)

        N_var  = h_dict["variable"].shape[0]
        H      = self.hidden_dim
        device = h_dict["variable"].device

        if ei is None or ei.shape[1] == 0:
            # 无桥接边：entity 信息无法注入，全零聚合
            agg = torch.zeros(N_var, H, device=device)
            fused = torch.cat([h_dict["variable"], agg], dim=-1)  # [N_var, 2H]
            h_dict["variable"] = F.relu(self.inject_fuse(fused))
            return h_dict

        # ── 提取边的源（entity）和目标（variable）索引 ──
        # 注入方向是 entity → variable，所以：
        var_idx = ei[0]    # variable 节点索引  [E]（消息的接收方）
        ent_idx = ei[1]    # entity 节点索引    [E]（消息的发送方）

        # ── 计算 Q, K, V ──
        Q_all = self.inject_q(h_dict["variable"])    # [N_var, H]
        K_all = self.inject_k(h_dict["entity"])      # [N_ent, H]
        V_all = self.inject_v(h_dict["entity"])      # [N_ent, H]

        Q_edge = Q_all[var_idx]   # [E, H]  接收方查询
        K_edge = K_all[ent_idx]   # [E, H]  发送方键
        V_edge = V_all[ent_idx]   # [E, H]  发送方值

        # ── 点积打分 + 缩放 ──
        e = torch.einsum("ij,ij->i", Q_edge, K_edge) / self._attn_scale   # [E]

        # ── Sparse Softmax（数值稳定版，按 variable 分组归一化）──
        #  ① 减去每个 variable 组内最大值（防止 exp 溢出）
        max_e = torch.full((N_var,), float("-inf"), device=device)
        max_e.scatter_reduce_(0, var_idx, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_e[var_idx])                              # [E]

        #  ② 按组求和
        sum_exp = torch.zeros(N_var, device=device)
        sum_exp.index_add_(0, var_idx, exp_e)

        #  ③ 归一化得注意力权重 α
        alpha = exp_e / sum_exp[var_idx].clamp(min=1e-9)                   # [E]

        # ── 加权聚合 V ──
        weighted = V_edge * alpha.unsqueeze(-1)                            # [E, H]
        agg = torch.zeros(N_var, H, device=device)
        agg.index_add_(0, var_idx, weighted)                               # [N_var, H]

        # ── 融合：concat(h_variable, agg) → Linear → h_variable_new ──
        fused = torch.cat([h_dict["variable"], agg], dim=-1)               # [N_var, 2H]
        h_dict["variable"] = F.relu(self.inject_fuse(fused))               # [N_var, H]

        return h_dict

    # ─────────────────────────────────────────────────────────────────
    # Stage 3: 数学推理（variable ↔ constraint 二分图消息传递）
    # ─────────────────────────────────────────────────────────────────

    def _math_reasoning(
        self,
        h_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        """
        经典 L2B 二分图消息传递：variable ↔ constraint。

        每一轮包含两步：
          ① v2c: variable → constraint
             GATConv(x=(h_var, h_con), ei_v2c) + 残差
          ② c2v: constraint → variable
             GATConv(x=(h_con, h_var), ei_c2v) + 残差

        若图中没有 ("variable", "constrains", "constraint") 边，直接跳过。
        """
        math_edge_key = ("variable", "constrains", "constraint")
        ei = edge_index_dict.get(math_edge_key, None)

        if ei is None:
            return h_dict

        # v2c: src=variable(ei[0]), dst=constraint(ei[1])
        ei_v2c = ei
        # c2v: 翻转方向
        ei_c2v = torch.stack([ei[1], ei[0]], dim=0)

        for v2c_conv, c2v_conv in zip(self.math_v2c_convs, self.math_c2v_convs):
            # ── ① variable → constraint ──
            h_con_old = h_dict["constraint"]
            h_con_new = v2c_conv(
                (h_dict["variable"], h_dict["constraint"]),
                ei_v2c,
            )
            h_dict["constraint"] = F.relu(h_con_new) + h_con_old

            # ── ② constraint → variable ──
            h_var_old = h_dict["variable"]
            h_var_new = c2v_conv(
                (h_dict["constraint"], h_dict["variable"]),
                ei_c2v,
            )
            h_dict["variable"] = F.relu(h_var_new) + h_var_old

        return h_dict

    # ─────────────────────────────────────────────────────────────────
    # 前向传播主函数
    # ─────────────────────────────────────────────────────────────────

    def forward(self, data: HeteroData) -> Tensor:
        """
        前向传播：Stage 0 → 1 → 2 → 3 → 4。

        参数
        ----
        data : PyG HeteroData，由 UniversalGraphBuilder.build() 生成
               结构固定：3 节点 (variable, constraint, entity)
                        3 边 (constrains, relates_to, mapped_to)

        返回
        ----
        scores : Tensor[num_variables]
            每个决策变量的分支优先级分数（不加 Sigmoid）。
        """
        x_dict          = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Stage 0: 全部节点投影到 hidden_dim
        h_dict = self._project_all(x_dict)

        # Stage 1: entity 节点内部消息传递
        h_dict = self._semantic_encoding(h_dict, edge_index_dict)

        # Stage 2: entity → variable 语义注入
        h_dict = self._semantic_injection(h_dict, edge_index_dict)

        # Stage 3: variable ↔ constraint 二分图推理
        h_dict = self._math_reasoning(h_dict, edge_index_dict)

        # Stage 4: 决策头 → 每个 variable 一个分数
        scores = self.scoring_head(h_dict["variable"]).squeeze(-1)  # [N_var]
        return scores
