#!/usr/bin/env python3
"""
UniversalGraphBuilder通用语义图构建器

========================== 架构核心思想 ==========================
  由于 USG 范式下所有业务节点统一为 "entity"、所有语义边统一为
  "relates_to"，图构建器不再需要动态按类型分桶。

  输出的 HeteroData 拓扑结构 **绝对固定**：
    3 种节点：variable, constraint, entity
    3 种边  ：constrains, mapped_to, relates_to

  这意味着 OntoGNN 模型可以拥有完全静态的参数结构——不再需要
  metadata 驱动的动态层注册。

构建三层图：
  ① 数学层   : variable[N, 19] + constraint[C, 5]
               + ("variable", "constrains", "constraint")
  ② 语义层   : entity[E, 128]
               + ("entity", "relates_to", "entity")
  ③ 桥接层   : ("variable", "mapped_to", "entity")

ID 映射机制：
  单一映射表 id_to_idx["emp_0"] = 0, id_to_idx["shift_0"] = 20, ...
  所有 entity 共享同一个局部索引空间。

Author: OntoBranch-2026 Team
"""

import json
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from torch_geometric.data import HeteroData
from src.generator.base_generator import GLOBAL_ENT_DIM

class UniversalGraphBuilder:
    """
    USG 极简构建器：将 (JSON, Ecole obs) 转换为结构固定的 HeteroData。

    输出保证恰好包含 3 种节点 + 3 种边，无论输入来自哪个业务领域。
    """

    def __init__(self, json_data: Dict[str, Any], verbose: bool = False):
        self.raw     = json_data
        self.verbose = verbose

        # ── entity 节点列表与 ID → 局部索引映射 ──
        # 因为 USG 下所有业务节点都是 "entity"，只需要一个 flat 映射
        self._entity_nodes: List[Dict] = []
        self._id_to_idx: Dict[str, int] = {}
        self._parse_nodes()

    # ─────────────────────────────────────────────────────────────────
    #  节点解析
    # ─────────────────────────────────────────────────────────────────

    def _parse_nodes(self):
        """
        遍历 JSON nodes 列表，构建 entity 节点索引。

        USG 契约保证所有节点的 type == "entity"，因此不需要按类型分桶。
        id_to_idx 是一个 flat 字典：
          {"emp_0": 0, "emp_1": 1, ..., "shift_0": 20, "shift_1": 21, ...}
        """
        for node in self.raw.get("nodes", []):
            node_id = node["id"]
            local_idx = len(self._entity_nodes)
            self._entity_nodes.append(node)
            self._id_to_idx[node_id] = local_idx

        if self.verbose:
            print(f"  [parse_nodes] entity: {len(self._entity_nodes)} nodes")
    # ─────────────────────────────────────────────────────────────────
    #  语义层：entity 节点特征 + relates_to 边
    # ─────────────────────────────────────────────────────────────────
    def _build_semantic_layer(self, data: HeteroData):
        """
        构建 entity 节点特征矩阵和 relates_to 语义边。

        entity 特征：
          每个 node["features"] 已由 Generator 对齐到 GLOBAL_ENT_DIM (128) 维。
          直接堆叠为 Tensor[num_entities, 128]。

        relates_to 边：
          USG 下所有语义边的 rel 都是 "relates_to"，
          src 和 dst 都映射到同一个 entity 索引空间。
        """
        # ── entity 节点特征矩阵 ──
        if self._entity_nodes:
            features = [n["features"] for n in self._entity_nodes]
            data["entity"].x = torch.tensor(features, dtype=torch.float32)
        else:
            # 无实体节点时的占位（理论上不应发生，但确保不崩溃）
            data["entity"].x = torch.zeros(0, GLOBAL_ENT_DIM, dtype=torch.float32)

        if self.verbose:
            print(f"  [semantic_nodes] entity: {data['entity'].x.shape}")

        # ── relates_to 语义边 ──
        src_list: List[int] = []
        dst_list: List[int] = []

        for edge in self.raw.get("edges", []):
            src_id = edge["src"]
            dst_id = edge["dst"]

            if src_id not in self._id_to_idx or dst_id not in self._id_to_idx:
                raise KeyError(
                    f"语义边引用了未注册的节点 ID: src='{src_id}', dst='{dst_id}'。"
                    f"已注册 ID: {list(self._id_to_idx.keys())[:10]}..."
                )

            src_list.append(self._id_to_idx[src_id])
            dst_list.append(self._id_to_idx[dst_id])

        if src_list:
            data["entity", "relates_to", "entity"].edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
        else:
            # 无语义边时的占位（空 edge_index，保证图结构完整）
            data["entity", "relates_to", "entity"].edge_index = torch.zeros(
                2, 0, dtype=torch.long
            )

        if self.verbose:
            ei = data["entity", "relates_to", "entity"].edge_index
            print(f"  [semantic_edges] ('entity','relates_to','entity'): {ei.shape[1]} edges")

    # ─────────────────────────────────────────────────────────────────
    #  桥接层：variable → entity 的 mapped_to 边
    # ─────────────────────────────────────────────────────────────────

    def _build_bridge_layer(self, data: HeteroData):
        """
        遍历 variable_map，建立 variable → entity 的桥接边。

        USG 下所有 mappings 的 type 都是 "entity"，因此只需要一组
        ("variable", "mapped_to", "entity") 边，不再按实体类型分桶。
        """
        var_src_list: List[int] = []
        ent_dst_list: List[int] = []

        for entry in self.raw.get("variable_map", []):
            var_idx = entry["var_index"]
            for mapping in entry["mappings"]:
                eid = mapping["id"]
                if eid not in self._id_to_idx:
                    raise KeyError(
                        f"variable_map 引用了未注册的实体 ID: '{eid}'。"
                    )
                ent_local = self._id_to_idx[eid]
                var_src_list.append(var_idx)
                ent_dst_list.append(ent_local)

        if var_src_list:
            data["variable", "mapped_to", "entity"].edge_index = torch.tensor(
                [var_src_list, ent_dst_list], dtype=torch.long
            )
        else:
            data["variable", "mapped_to", "entity"].edge_index = torch.zeros(
                2, 0, dtype=torch.long
            )

        if self.verbose:
            ei = data["variable", "mapped_to", "entity"].edge_index
            print(f"  [bridge] ('variable','mapped_to','entity'): {ei.shape[1]} edges")

    # ─────────────────────────────────────────────────────────────────
    #  数学层：variable / constraint 节点 + constrains 边
    # ─────────────────────────────────────────────────────────────────

    def _build_math_layer(self, data: HeteroData, ecole_obs=None):
        """
        从 Ecole NodeBipartite 观测中提取数学层特征。

        Ecole 观测结构：
          obs.variable_features  : ndarray [N_var, 19]
          obs.row_features       : ndarray [N_con, 5]
          obs.edge_features.indices : ndarray [2, E]
            indices[0] = constraint 行号
            indices[1] = variable 列号

        安全处理：
          ① NaN → 0.0（np.nan_to_num）
          ② None 检查使用 `if x is None`（numpy array 兼容）
          ③ ecole_obs=None 时退化为全零占位

        边方向：("variable", "constrains", "constraint")
          src = variable = indices[1], dst = constraint = indices[0]
        """
        num_vars = len(self.raw.get("variable_map", []))

        if ecole_obs is None:
            # 无 Ecole 观测：全零占位（维度依然是 19 / 5）
            data["variable"].x   = torch.zeros(num_vars, 19, dtype=torch.float32)
            data["constraint"].x = torch.zeros(1, 5, dtype=torch.float32)
            # 占位边：对角线连接
            k = min(num_vars, 1)
            data["variable", "constrains", "constraint"].edge_index = torch.stack(
                [torch.arange(k, dtype=torch.long),
                 torch.arange(k, dtype=torch.long)]
            )
            return

        # ── variable 特征 [N_var, 19] ──
        _vf = getattr(ecole_obs, "variable_features", None)
        if _vf is None:
            _vf = getattr(ecole_obs, "column_features", None)

        if _vf is not None:
            _vf = np.nan_to_num(np.asarray(_vf, dtype=np.float32), nan=0.0)
            data["variable"].x = torch.from_numpy(_vf)
        else:
            data["variable"].x = torch.zeros(num_vars, 19, dtype=torch.float32)

        # ── constraint 特征 [N_con, 5] ──
        _cf = getattr(ecole_obs, "row_features", None)
        if _cf is None:
            _cf = getattr(ecole_obs, "constraint_features", None)

        if _cf is not None:
            _cf = np.nan_to_num(np.asarray(_cf, dtype=np.float32), nan=0.0)
            data["constraint"].x = torch.from_numpy(_cf)
        else:
            data["constraint"].x = torch.zeros(1, 5, dtype=torch.float32)

        # ── variable ↔ constraint 二分图边 ──
        edge_feat = getattr(ecole_obs, "edge_features", None)
        if edge_feat is not None:
            indices = getattr(edge_feat, "indices", None)
            if indices is not None:
                idx = np.asarray(indices, dtype=np.int64)
                # 交换行：src=variable(idx[1]), dst=constraint(idx[0])
                ei = torch.tensor(np.stack([idx[1], idx[0]]), dtype=torch.long)
                data["variable", "constrains", "constraint"].edge_index = ei
                if self.verbose:
                    print(
                        f"  [math] var:{data['variable'].x.shape} "
                        f"con:{data['constraint'].x.shape} edges:{ei.shape[1]}"
                    )
                return

        # 无法获取真实边：退化为对角线占位
        nv = data["variable"].x.shape[0]
        nc = data["constraint"].x.shape[0]
        k  = min(nv, nc)
        data["variable", "constrains", "constraint"].edge_index = torch.stack(
            [torch.arange(k, dtype=torch.long),
             torch.arange(k, dtype=torch.long)]
        )

    # ─────────────────────────────────────────────────────────────────
    #  主构建入口
    # ─────────────────────────────────────────────────────────────────

    def build(self, ecole_obs=None) -> HeteroData:
        """
        按顺序构建三层，返回结构固定的 HeteroData。

        输出保证恰好包含：
          节点：variable, constraint, entity
          边  ：("variable", "constrains", "constraint")
               ("entity", "relates_to", "entity")
               ("variable", "mapped_to", "entity")
        """
        data = HeteroData()
        self._build_math_layer(data, ecole_obs)    # ① variable / constraint
        self._build_semantic_layer(data)            # ② entity + relates_to
        self._build_bridge_layer(data)              # ③ variable → entity
        return data


# ═════════════════════════════════════════════════════════════════════
#  便捷接口
# ═════════════════════════════════════════════════════════════════════

def load_and_build(
    json_path: str,
    ecole_obs=None,
    verbose: bool = False,
) -> HeteroData:
    """
    从 JSON 文件路径加载实例，构建并返回结构固定的 HeteroData。

    参数
    ----
    json_path : Generator 输出的 .json 文件路径
    ecole_obs : Ecole NodeBipartite 观测（可选）
    verbose   : 是否打印各层统计

    返回
    ----
    HeteroData，拓扑结构固定为 3 节点 + 3 边

    示例
    ----
    >>> data = load_and_build("data/raw/employee_scheduling/es_001.json")
    >>> data.node_types   # ['variable', 'constraint', 'entity']
    >>> data.edge_types   # [('variable','constrains','constraint'),
    ...                   #  ('entity','relates_to','entity'),
    ...                   #  ('variable','mapped_to','entity')]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    builder = UniversalGraphBuilder(json_data, verbose=verbose)
    return builder.build(ecole_obs=ecole_obs)
