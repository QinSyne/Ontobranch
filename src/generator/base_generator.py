#!/usr/bin/env python3
"""
BaseGenerator v3.0 —— 通用语义图（USG）范式抽象基类

========================== 架构核心思想 ==========================
  为了实现"统一大模型（One-Model-Fits-All）"，我们采用 LLM 的
  "最大上下文（Max Context）+ Zero-Padding" 思想：
    - 所有业务实体统一为 "entity" 类型
    - 所有语义边统一为 "relates_to" 关系
    - 原始类型信息通过 entity_type_idx 编码为 One-hot 嵌入前缀

  全局输出特征严格对齐到 GLOBAL_ENT_DIM = 128 维：
    [0  : 16 ]  → 16 维 entity type One-hot 编码
    [16 : 128]  → 112 维原始特征槽（不足补零）

设计模式：Template Method（模板方法）
  - generate() 是固定的"总流程"
  - 具体步骤由子类通过覆写抽象方法来填充

输出格式（"数据协议 v3.0"）：
  每次调用 generate() 产出一对文件：
    ① <name>.lp    供 SCIP / Ecole 加载
    ② <name>.json  供图构建器使用（USG 格式）

Author: OntoBranch-2026 Team
"""

import os
import json
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

# ═════════════════════════════════════════════════════════════════════
#  全局常量 —— 整个系统的特征维度契约
# ═════════════════════════════════════════════════════════════════════

GLOBAL_ENT_DIM = 128   # 实体特征总维度（type one-hot + 原始特征 + zero-padding）
TYPE_DIM = 16           # 实体类型 One-hot 编码维度（支持最多 16 种实体类型）


class BaseGenerator(ABC):
    """
    USG 范式抽象基类。

    ==================== 关键契约 ====================
    ① JSON 中所有业务节点的 "type" 必须是 "entity"
    ② JSON 中所有语义边的 "rel" 必须是 "relates_to"
    ③ 所有 entity 节点的 "features" 维度 = GLOBAL_ENT_DIM (128)
    ④ 原始业务类型信息仅通过 entity_type_idx 体现在
       features[0:TYPE_DIM] 的 One-hot 编码中
    ⑤ variable_map 中 mappings 的 type 统一为 "entity"

    子类必须实现以下 6 个抽象方法：
      - _generate_entities()
      - _generate_variables()
      - _build_model()
      - _build_json_nodes()
      - _build_json_edges()
      - _get_problem_type()
    """

    def __init__(self, seed: int = 42):
        """
        参数
        ----
        seed : 全局随机种子，保证用相同参数可复现同一个实例。
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.entities: Dict[str, Any] = {}
        self.var_index: Dict[Any, int] = {}
        self.var_list: List[Dict] = []
        self.model = None

    # ─────────────────────────────────────────────────────────────────
    #  通用特征对齐方法（USG 核心）
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _harmonize_features(entity_type_idx: int, raw_features: List[float]) -> List[float]:
        """
        将任意原始特征向量对齐到 GLOBAL_ENT_DIM (128) 维。

        编码规则（数据协议 v3.0 强制要求）：
          [0     : TYPE_DIM]              → entity type One-hot 编码
          [TYPE_DIM : TYPE_DIM + raw_len] → 原始数值特征（保持原始顺序）
          [TYPE_DIM + raw_len : 128]      → 全零填充（Zero-padding）

        参数
        ----
        entity_type_idx : int
            该实体的类型编号（0-indexed），如 Employee=0, Shift=1。
            必须满足 0 <= entity_type_idx < TYPE_DIM。
        raw_features : List[float]
            子类计算好的原始数值特征（任意长度，但不能超过 112 维）。

        返回
        ----
        harmonized : List[float]
            长度恒为 GLOBAL_ENT_DIM (128) 的特征向量。

        异常
        ----
        ValueError : entity_type_idx 越界或 raw_features 过长。
        """
        if not (0 <= entity_type_idx < TYPE_DIM):
            raise ValueError(
                f"entity_type_idx={entity_type_idx} 越界，"
                f"合法范围 [0, {TYPE_DIM})。"
            )
        raw_len = len(raw_features)
        max_raw = GLOBAL_ENT_DIM - TYPE_DIM   # 112
        if raw_len > max_raw:
            raise ValueError(
                f"原始特征维度 {raw_len} 超过可用槽位 {max_raw}。"
            )

        # ── 构建 128 维向量 ──
        vec = [0.0] * GLOBAL_ENT_DIM

        # ① One-hot 前缀
        vec[entity_type_idx] = 1.0

        # ② 原始特征填入 [TYPE_DIM : TYPE_DIM + raw_len]
        for i, v in enumerate(raw_features):
            vec[TYPE_DIM + i] = float(v)

        # ③ 剩余位 [TYPE_DIM + raw_len : 128] 已经是 0.0，无需操作

        return vec

    # ─────────────────────────────────────────────────────────────────
    #  抽象方法（子类必须实现）
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def _generate_entities(self, **kwargs) -> Dict[str, Any]:
        """
        生成业务实体（如员工、班次、城市、车辆…）。

        返回
        ----
        entities : Dict[str, Any]
            键 = 实体类型名，如 "employees"、"shifts"
            值 = 该类型的实体字典（id → 属性字典）
        """
        ...

    @abstractmethod
    def _generate_variables(self) -> Tuple[Dict[Any, int], List[Dict]]:
        """
        枚举所有决策变量，建立业务键 ↔ 变量列号的双向索引。

        ★ USG 契约：mappings 中的 type 必须统一为 "entity" ★

        返回
        ----
        var_index : Dict[Any, int]
            正向映射：业务键 → 列号
        var_list : List[Dict]
            反向映射：每个元素格式：
            {
              "business_key": (emp_id, shift_id),
              "var_name":  "x_0_0",              # 可选，人类可读
              "mappings": [
                {"id": "emp_0",   "semantic_type": "Employee"},
                {"id": "shift_0", "semantic_type": "Shift"},
                # ★ 不再需要 "type": "entity" 字段，
                #   builder 固定构建为 entity 节点， JSON 不重复存储。
              ]
            }
        """
        ...

    @abstractmethod
    def _build_model(self) -> Any:
        """
        构建并返回 pyscipopt.Model 对象。

        ┌──────────────────────────────────────────────────────────┐
        │  ★★★ 强制对齐契约 ★★★                                    │
        │                                                          │
        │  添加变量时，必须严格按照 self.var_list 的顺序遍历：      │
        │      for k, entry in enumerate(self.var_list):           │
        │          vars[k] = model.addVar(...)                     │
        │                                                          │
        │  绝对禁止使用其他循环顺序来添加变量！                     │
        └──────────────────────────────────────────────────────────┘
        """
        ...

    @abstractmethod
    def _build_json_nodes(self) -> List[Dict]:
        """
        将业务实体转成 JSON 的 nodes 列表。

        ★ USG 契约（子类必须遵守）：
          - 每个节点的 "type" 必须写死为 "entity"
          - "features" 必须经过 _harmonize_features() 对齐到 128 维
          - "id" 保持全局唯一字符串

        返回
        ----
        nodes : List[Dict]
            [{"id": "emp_0", "type": "entity", "features": [128 维...]}, ...]
        """
        ...

    @abstractmethod
    def _build_json_edges(self) -> List[Dict]:
        """
        将业务实体间的关系转成 JSON 的 edges 列表。

        ★ USG 契约（子类必须遵守）：
          - 每条边的 "rel" 必须写死为 "relates_to"
          - "src" / "dst" 必须与 nodes 中的 id 对应

        返回
        ----
        edges : List[Dict]
            [
              {"src": "shift_0", "dst": "shift_1", "rel": "relates_to",
               "semantic_rel": "same_day"},   # semantic_rel 可选，仅供人类阅读
              ...
            ]
        """
        ...

    @abstractmethod
    def _get_problem_type(self) -> str:
        """返回问题类型标识，如 "employee_scheduling"。"""
        ...

    # ─────────────────────────────────────────────────────────────────
    #  模板方法（总流程，子类不需要覆写）
    # ─────────────────────────────────────────────────────────────────

    def generate(self, output_dir: str, instance_name: str, **kwargs) -> Dict[str, str]:
        """
        生成一个 MILP 实例，输出 .lp 和 .json 两个文件。

        流程固定：
          1. 生成实体   → self.entities
          2. 生成变量   → self.var_index / self.var_list
          3. 构建模型   → self.model
          4. 写 .lp 文件
          5. 写 .json 文件（USG 格式）
        """
        os.makedirs(output_dir, exist_ok=True)

        lp_path   = os.path.join(output_dir, f"{instance_name}.lp")
        json_path = os.path.join(output_dir, f"{instance_name}.json")

        # Step 1：生成业务实体
        self.entities = self._generate_entities(**kwargs)

        # Step 2：生成变量映射
        self.var_index, self.var_list = self._generate_variables()

        # Step 3：构建 SCIP 模型
        self.model = self._build_model()

        # Step 4：写 .lp 文件
        self.model.writeProblem(lp_path)

        # Step 5：写 .json 文件（USG 格式）
        self._write_json(json_path, instance_name, **kwargs)

        return {"lp": lp_path, "json": json_path}

    # ─────────────────────────────────────────────────────────────────
    #  私有辅助方法
    # ─────────────────────────────────────────────────────────────────

    def _build_annotation(self) -> Dict[str, Any]:
        """
        返回人类可读的注解字典（以 _ 前缀发布到 JSON，流水线不读取）。

        子类可覆写此方法，加入特征维度说明、边类型说明等。
        基类默认返回空字典。
        """
        return {}

    def _write_json(self, json_path: str, instance_name: str, **kwargs):
        """
        将 USG 格式的语义信息写入 .json 文件。

        ★ 数据协议 v3.0 输出格式：
          - nodes[].type 全部为 "entity"
          - nodes[].features 全部为 128 维
          - edges[] 只含 src / dst / semantic_rel（rel 字段已省略，
            builder 固定构建为 relates_to，无需存储）
          - variable_map[].mappings[].type 全部为 "entity"

        顶部 _annotation 字段（人类只读，流水线跳过）：
          - feature_schema  : 各实体类型的特征维度含义
          - semantic_rel_schema : 边类型含义
          - edge_stats      : 各类型边计数汇总
        """
        nodes = self._build_json_nodes()
        edges = self._build_json_edges()

        # ── 安全校验：确保 USG 契约被遵守 ──
        for node in nodes:
            assert node["type"] == "entity", (
                f"USG 违规：节点 {node['id']} 的 type 应为 'entity'，"
                f"实际为 '{node['type']}'。"
            )
            assert len(node["features"]) == GLOBAL_ENT_DIM, (
                f"USG 违规：节点 {node['id']} 的 features 维度应为 "
                f"{GLOBAL_ENT_DIM}，实际为 {len(node['features'])}。"
            )
        # 注：edges 不再要求 rel 字段；builder 固定以 relates_to 构建语义边

        # ── 边统计（人类可读，写入 _annotation）──
        rel_counts: Dict[str, int] = {}
        for edge in edges:
            k = edge.get("semantic_rel", "unknown")
            rel_counts[k] = rel_counts.get(k, 0) + 1

        annotation = self._build_annotation()
        annotation["edge_stats"] = {
            "total": len(edges),
            "by_semantic_rel": rel_counts,
        }

        # 构建 variable_map
        variable_map = []
        for idx, entry in enumerate(self.var_list):
            item: Dict[str, Any] = {"var_index": idx}
            if "var_name" in entry:              # 人类可读，流水线不依赖
                item["var_name"] = entry["var_name"]
            item["mappings"] = entry["mappings"]
            variable_map.append(item)

        payload = {
            "_annotation":  annotation,          # 人类只读，流水线跳过
            "metadata": {
                "problem_type":   self._get_problem_type(),
                "instance_name":  instance_name,
                "num_variables":  len(self.var_list),
                "num_edges":      len(edges),
                "edge_type_counts": rel_counts,   # 各语义边类型的数量
                "seed":           self.seed,
                **kwargs,
            },
            "nodes":        nodes,
            "edges":        edges,
            "variable_map": variable_map,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False, ensure_ascii=False)
