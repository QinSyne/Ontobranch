#!/usr/bin/env python3
"""
OntologyGraphBuilder —— 异构图构建器

将 Generator 输出的 (.json, Ecole obs) 对转换为 PyG HeteroData，
供 OntoGNN 消费。

构建的图包含三层：
  ① 语义层   : 来自 JSON nodes（如 Employee, Shift 节点）
               + 来自 JSON edges（如 same_day 边）
  ② 桥接层   : variable → 语义实体 的边
               （来自 JSON variable_map）
  ③ 数学层   : variable / constraint 节点 + variable↔constraint 二分图边
               （来自 Ecole NodeBipartite 观测）

核心设计：字符串 ID → PyG 局部整数索引 的转换
  PyG 的 edge_index 只认 0, 1, 2... 这样的局部索引。
  JSON 里的节点 ID 是字符串（"emp_0", "shift_3"）。
  解析时我们维护一个两层字典：
    id_to_idx["Employee"]["emp_0"] = 0
    id_to_idx["Shift"]["shift_3"] = 3
  所有涉及节点引用的地方，都通过这个字典做一次查表转换。

Author: OntoBranch-2026 Team
"""

import json
import numpy as np
import torch
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from torch_geometric.data import HeteroData


class OntologyGraphBuilder:
    """
    读取 Generator 输出的 JSON，构建三层 PyG HeteroData。

    直接使用 JSON nodes 里的 features 列表作为节点特征，
    不再需要额外的特征编码器（Generator 已经把特征算好了）。
    """

    def __init__(self, json_data: Dict[str, Any], verbose: bool = False):
        self.raw     = json_data
        self.verbose = verbose

        # 按类型分组的节点列表
        self._nodes_by_type: Dict[str, List[Dict]] = {}

        # 字符串 ID → 局部整数索引的两层查找表（最重要！）
        # {"Employee": {"emp_0": 0, "emp_1": 1, ...},
        #  "Shift":    {"shift_0": 0, "shift_1": 1, ...}}
        self._id_to_idx: Dict[str, Dict[str, int]] = {}

        self._parse_nodes()

    def _parse_nodes(self):
        """
        遍历 JSON 的 nodes 列表，按 type 分组，并建立 id_to_idx 映射。

        JSON 格式保证每个 node 有三个字段：
          {"id": "emp_0", "type": "Employee", "features": [1.13, 12.5, 0.0, 1.0, 0.0]}

        关键操作：
          local_idx = 该类型节点的顺序编号（从 0 开始）
          id_to_idx["Employee"]["emp_0"] = local_idx

        为什么不用全局 ID？
          PyG 的 HeteroData 为每种节点类型维护独立的特征矩阵。
          "Employee" 的矩阵下标从 0 开始，"Shift" 也从 0 开始，互相独立。
          所以 emp_0 的局部索引是 0，shift_0 的局部索引也是 0，并不冲突。
        """
        for node in self.raw.get("nodes", []):
            ntype   = node["type"]
            node_id = node["id"]

            if ntype not in self._nodes_by_type:
                self._nodes_by_type[ntype] = []
                self._id_to_idx[ntype]     = {}

            # local_idx = 当前该类型已有节点的数量（0-indexed）
            local_idx = len(self._nodes_by_type[ntype])
            self._nodes_by_type[ntype].append(node)
            self._id_to_idx[ntype][node_id] = local_idx

        if self.verbose:
            for ntype, nlist in self._nodes_by_type.items():
                print(f"  [parse_nodes] {ntype}: {len(nlist)} nodes")

    def _find_type(self, node_id: str) -> str:
        """
        在 _id_to_idx 里逐类型查找 node_id，返回它的类型名。

        例：_find_type("shift_3") → "Shift"

        如果找不到，直接 raise KeyError（遵守 No Guessing 原则）。
        """
        for ntype, id_map in self._id_to_idx.items():
            if node_id in id_map:
                return ntype
        raise KeyError(
            f"节点 id '{node_id}' 在所有类型中均未找到。"
            f"已注册类型: {list(self._id_to_idx.keys())}"
        )

    def _build_semantic_layer(self, data: HeteroData):
        """
        构建语义节点特征矩阵，并将语义边写入 HeteroData。

        节点特征：
          直接把 JSON node["features"] 列表堆叠成矩阵。
          例：Employee 有 15 个节点，每个 features 长度 5
           → data["Employee"].x = Tensor[15, 5]

        语义边（来自 JSON edges）：
          每条边 {"src": "shift_0", "dst": "shift_4", "rel": "same_day"}
          → src_type = _find_type("shift_0") = "Shift"
          → src_local = _id_to_idx["Shift"]["shift_0"] = 0
          → dst_local = _id_to_idx["Shift"]["shift_4"] = 4
          → 写入 data["Shift", "same_day", "Shift"].edge_index

        边特征（可选）：
          若边携带 "features" 字段（如 VRP 的距离/成本）：
            {"src":"loc_0","dst":"loc_1","rel":"road","features":[12.5,0.8]}
          则同类型的所有边若均有 features，
          → data["Loc","road","Loc"].edge_attr = Tensor[num_edges, feat_dim]
          若只有部分边携带 features（数据不一致），则跳过 edge_attr，不猜测补零。
        """
        # 节点特征矩阵
        for ntype, node_list in self._nodes_by_type.items():
            features = [n["features"] for n in node_list]
            data[ntype].x = torch.tensor(features, dtype=torch.float32)
            if self.verbose:
                print(f"  [semantic_nodes] {ntype}: {data[ntype].x.shape}")

        # 语义边：先分桶，再转 tensor
        # key = (src_type, rel, dst_type)
        # 每个桶是三元组：(src 列表, dst 列表, 边特征向量列表)
        # 边特征向量列表里存 List[float] 或 None（无 features 字段时）
        edge_bucket: Dict[Tuple, Tuple[List, List, List]] = defaultdict(
            lambda: ([], [], [])
        )

        for edge in self.raw.get("edges", []):
            src_id = edge["src"]
            dst_id = edge["dst"]
            rel    = edge["rel"]

            src_type  = self._find_type(src_id)
            dst_type  = self._find_type(dst_id)
            src_local = self._id_to_idx[src_type][src_id]
            dst_local = self._id_to_idx[dst_type][dst_id]

            bucket = edge_bucket[(src_type, rel, dst_type)]
            bucket[0].append(src_local)
            bucket[1].append(dst_local)
            # features 字段可选：无则追加 None，有则追加特征向量
            bucket[2].append(edge.get("features", None))

        for (src_type, rel, dst_type), (srcs, dsts, feats) in edge_bucket.items():
            data[src_type, rel, dst_type].edge_index = torch.tensor(
                [srcs, dsts], dtype=torch.long
            )

            # 只有当 **所有** 边都携带特征时，才写出 edge_attr。
            # 部分缺失视为数据不一致，跳过而非靠猜测补零。
            if feats and all(f is not None for f in feats):
                data[src_type, rel, dst_type].edge_attr = torch.tensor(
                    feats, dtype=torch.float32
                )
                if self.verbose:
                    ea = data[src_type, rel, dst_type].edge_attr
                    print(
                        f"  [semantic_edges] ('{src_type}','{rel}','{dst_type}'): "
                        f"{len(srcs)} edges, edge_attr {ea.shape}"
                    )
            else:
                if self.verbose:
                    print(
                        f"  [semantic_edges] ('{src_type}','{rel}','{dst_type}'): "
                        f"{len(srcs)} edges (no edge_attr)"
                    )

    def _build_bridge_layer(self, data: HeteroData):
        """
        遍历 variable_map，建立 variable → 语义实体 的桥接边。

        variable_map 中每条记录：
          {
            "var_index": 5,
            "mappings": [
              {"type": "Employee", "id": "emp_1"},
              {"type": "Shift",    "id": "shift_5"},
            ]
          }

        转换逻辑：
          var_idx = 5（SCIP 变量列号 = data["variable"] 的局部索引）

          对 {"type": "Employee", "id": "emp_1"}：
            ent_local = _id_to_idx["Employee"]["emp_1"] = 1
            → 写入 data["variable", "mapped_to", "Employee"].edge_index
              src=5, dst=1

          对 {"type": "Shift", "id": "shift_5"}：
            ent_local = _id_to_idx["Shift"]["shift_5"] = 5
            → 写入 data["variable", "mapped_to", "Shift"].edge_index
              src=5, dst=5

        关系名统一用 "mapped_to"——不同目标类型的 edge_type 三元组不同，
        不需要在关系名上再区分。
        """
        # key = entity_type 字符串（src 固定是 "variable"，rel 固定是 "mapped_to"）
        bridge_bucket: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))

        for entry in self.raw.get("variable_map", []):
            var_idx  = entry["var_index"]
            for mapping in entry["mappings"]:
                etype     = mapping["type"]
                eid       = mapping["id"]
                ent_local = self._id_to_idx[etype][eid]  # 查表
                bridge_bucket[etype][0].append(var_idx)
                bridge_bucket[etype][1].append(ent_local)

        for etype, (srcs, dsts) in bridge_bucket.items():
            data["variable", "mapped_to", etype].edge_index = torch.tensor(
                [srcs, dsts], dtype=torch.long
            )
            if self.verbose:
                print(f"  [bridge] ('variable','mapped_to','{etype}'): {len(srcs)}")

    def _build_math_layer(self, data: HeteroData, ecole_obs=None):
        """
        从 Ecole NodeBipartite 观测中提取数学层特征。

        Ecole NodeBipartite 观测结构：
          obs.variable_features  : np.ndarray [num_vars, 19]
          obs.row_features        : np.ndarray [num_constraints, 5]
          obs.edge_features.indices : np.ndarray [2, num_edges]
            indices[0] = constraint 行号
            indices[1] = variable 列号

        ⚠ 两个安全处理：
          1. NaN 清洗：Ecole 对不适用特征填 NaN → 替换为 0
          2. None 检查：用 `if x is None`，禁止 `if not x`（numpy array 歧义）

        边方向：edge type 为 ('variable', 'constrains', 'constraint')
          src = variable = indices[1]
          dst = constraint = indices[0]
          → 需要把 indices 的两行交换
        """
        num_vars = len(self.raw.get("variable_map", []))

        if ecole_obs is None:
            data["variable"].x   = torch.zeros(num_vars, 1)
            data["constraint"].x = torch.zeros(1, 1)
            return

        # variable 特征（兼容不同 Ecole 版本的属性名）
        _vf = getattr(ecole_obs, "variable_features", None)
        if _vf is None:
            _vf = getattr(ecole_obs, "column_features", None)

        if _vf is not None:
            _vf = np.nan_to_num(np.asarray(_vf, dtype=np.float32), nan=0.0)
            data["variable"].x = torch.from_numpy(_vf)
        else:
            data["variable"].x = torch.zeros(num_vars, 1)

        # constraint 特征
        _cf = getattr(ecole_obs, "row_features", None)
        if _cf is None:
            _cf = getattr(ecole_obs, "constraint_features", None)

        if _cf is not None:
            _cf = np.nan_to_num(np.asarray(_cf, dtype=np.float32), nan=0.0)
            data["constraint"].x = torch.from_numpy(_cf)
        else:
            data["constraint"].x = torch.zeros(1, 1)

        # variable ↔ constraint 二分图边
        edge_feat = getattr(ecole_obs, "edge_features", None)
        if edge_feat is not None:
            indices = getattr(edge_feat, "indices", None)
            if indices is not None:
                idx = np.asarray(indices, dtype=np.int64)
                # 交换两行：src=variable(idx[1]), dst=constraint(idx[0])
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
            [torch.arange(k), torch.arange(k)]
        )

    def build(self, ecole_obs=None) -> HeteroData:
        """按顺序构建三层，返回完整 HeteroData。"""
        data = HeteroData()
        self._build_math_layer(data, ecole_obs)   # ① variable / constraint
        self._build_semantic_layer(data)           # ② Employee / Shift / ...
        self._build_bridge_layer(data)             # ③ variable → entity
        return data


def load_and_build(
    json_path: str,
    ecole_obs=None,
    verbose: bool = False,
) -> HeteroData:
    """
    从 JSON 文件路径加载实例，构建并返回 HeteroData。

    参数
    ----
    json_path : Generator 输出的 .json 文件路径
    ecole_obs : Ecole NodeBipartite 观测（可选）
    verbose   : 是否打印各层统计

    返回
    ----
    HeteroData，节点类型动态取决于 JSON 中的 type 字段
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    builder = OntologyGraphBuilder(json_data, verbose=verbose)
    return builder.build(ecole_obs=ecole_obs)
