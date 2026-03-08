#!/usr/bin/env python3
"""
BaseGenerator —— 所有 MILP 实例生成器的抽象基类

设计模式：Template Method（模板方法）
  - generate() 是固定的"总流程"，定义了"先生成实体 → 再建变量 → 再加约束
    → 最后写文件"这个不变的骨架
  - 具体步骤（实体长什么样、约束怎么写）由子类通过覆写抽象方法来填充

输出格式（"数据宪法"）：
  每次调用 generate() 产出一对文件：
    ① <name>.lp    供 SCIP / Ecole 加载，包含变量声明、约束、目标函数
    ② <name>.json  供图构建器使用，包含业务实体节点、边、变量映射

Author: OntoBranch-2026 Team
"""

import os
import json
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseGenerator(ABC):
    """
    所有生成器的抽象基类。

    子类必须实现以下 6 个抽象方法（详见各方法的 docstring）：
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
               在构造函数里统一设置，子类不需要再操心。
        """
        self.seed = seed
        # 同时固定 Python 内置 random 和 numpy 的随机状态
        random.seed(seed)
        np.random.seed(seed)

        # 以下属性在 generate() 调用后会被赋值，
        # 供子类的各个方法互相访问（避免在方法间反复传参）
        self.entities: Dict[str, Any] = {}   # 业务实体，键是实体类型名
        self.var_index: Dict[Any, int] = {}  # 正向映射：业务键 → 变量列号
        self.var_list: List[Dict] = []           # 反向映射：变量列号 → 带类型的映射描述
        self.model = None                    # pyscipopt Model 对象

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
            键   = 实体类型名，如 "employees"、"shifts"、"cities"
            值   = 该类型的实体字典（id → 属性字典）

        示例（员工排班）：
            {
              "employees": {0: {"name": "emp_0", "skill_weight": 2.0, ...}},
              "shifts":    {0: {"name": "shift_0", "min_coverage": 2, ...}},
            }
        """
        ...

    @abstractmethod
    def _generate_variables(self) -> Tuple[Dict[Any, int], List[Dict]]:
        """
        枚举所有决策变量，建立业务键 ↔ 变量列号的双向索引。

        前提：self.entities 已由 _generate_entities() 赋值。

        返回
        ----
        var_index : Dict[Any, int]
            正向映射：业务键（如 (emp_id, shift_id) tuple）→ 列号

        var_list : List[Dict]
            反向映射：列号 → 带类型的映射描述。var_list[k] 对应第 k 列。

            ★ 每个元素必须是如下格式的字典：
            {
              "business_key": (emp_id, shift_id),     # 方便程序内部查询
              "mappings": [
                {"type": "Employee", "id": "emp_0"},
                {"type": "Shift",    "id": "shift_0"},
              ]
            }

            其中 mappings 列表描述了该变量关联的所有业务实体，
            包含 type（节点类型，与 JSON nodes 中的 type 一致）
            和 id（节点 ID，与 JSON nodes 中的 id 一致）。

            图构建器依赖 type+id 来建立 variable → entity 的桥接边，
            仅靠整数 ID 无法区分"employee_0"和"shift_0"。
        """
        ...

    @abstractmethod
    def _build_model(self) -> Any:
        """
        构建并返回 pyscipopt.Model 对象。

        职责：
          1. 创建 Model()
          2. 添加决策变量（addVar）
          3. 添加约束（addCons）
          4. 设置目标函数（setObjective）

        前提：self.entities、self.var_index、self.var_list 已赋值。

        ┌──────────────────────────────────────────────────────────┐
        │  ★★★ 强制对齐契约 ★★★                                    │
        │                                                          │
        │  添加变量时，必须严格按照 self.var_list 的顺序遍历：      │
        │                                                          │
        │      vars = {}                                           │
        │      for k, entry in enumerate(self.var_list):           │
        │          vars[k] = model.addVar(...)                     │
        │                                                          │
        │  绝对禁止使用其他循环顺序（如 for emp in employees       │
        │  嵌套 for shift in shifts）来添加变量！                   │
        │                                                          │
        │  原因：SCIP 内部的变量列号 = addVar() 的调用次序。       │
        │  Ecole 读到的 variable_features[k] 就是第 k 次添加的     │
        │  变量。JSON variable_map[k] 也按 var_list[k] 写出。      │
        │  三者的 k 必须对齐，否则语义标签会错位。                  │
        └──────────────────────────────────────────────────────────┘

        返回
        ----
        model : pyscipopt.Model
            已完全构建好、可直接调用 writeProblem() 的模型
        """
        ...

    @abstractmethod
    def _build_json_nodes(self) -> List[Dict]:
        """
        将业务实体转成 JSON 的 nodes 列表。

        JSON 格式规定（数据宪法）：
          每个节点是一个字典，必须包含：
            - "id"       : 全局唯一字符串，如 "employee_0"、"shift_3"
            - "type"     : 节点类型，如 "Employee"、"Shift"
            - "features" : 数值特征列表（float），供 GNN 使用

        返回
        ----
        nodes : List[Dict]
            所有业务实体节点的列表
        """
        ...

    @abstractmethod
    def _build_json_edges(self) -> List[Dict]:
        """
        将业务实体间的关系转成 JSON 的 edges 列表。

        JSON 格式规定（数据宪法）：
          每条边是一个字典，必须包含：
            - "src"      : 源节点 id（与 nodes 里的 id 对应）
            - "dst"      : 目标节点 id
            - "rel"      : 关系类型，如 "assigned_to"、"connects"

        返回
        ----
        edges : List[Dict]
        """
        ...

    @abstractmethod
    def _get_problem_type(self) -> str:
        """
        返回问题类型的短字符串标识，如 "employee_scheduling"。
        用于 JSON metadata 字段。
        """
        ...

    # ─────────────────────────────────────────────────────────────────
    #  模板方法（总流程，子类不需要覆写）
    # ─────────────────────────────────────────────────────────────────

    def generate(self, output_dir: str, instance_name: str, **kwargs) -> Dict[str, str]:
        """
        生成一个 MILP 实例，输出 .lp 和 .json 两个文件。

        这是"总指挥"方法，流程固定：
          1. 生成实体   → 存入 self.entities
          2. 生成变量   → 存入 self.var_index / self.var_list
          3. 构建模型   → 存入 self.model
          4. 写 .lp 文件（调用 SCIP 的 writeProblem）
          5. 写 .json 文件（调用 _write_json）

        参数
        ----
        output_dir    : 输出目录路径（不存在则自动创建）
        instance_name : 实例文件名（不含扩展名），如 "emp_001"
        **kwargs      : 传递给 _generate_entities() 的领域参数，
                        如 num_employees=20, num_shifts=30

        返回
        ----
        paths : Dict[str, str]  {"lp": "...", "json": "..."}
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        lp_path   = os.path.join(output_dir, f"{instance_name}.lp")
        json_path = os.path.join(output_dir, f"{instance_name}.json")

        # ── Step 1：生成业务实体，存到 self.entities ──
        self.entities = self._generate_entities(**kwargs)

        # ── Step 2：生成变量映射，存到 self.var_index / self.var_list ──
        self.var_index, self.var_list = self._generate_variables()

        # ── Step 3：构建 SCIP 模型（变量 + 约束 + 目标函数）──
        self.model = self._build_model()

        # ── Step 4：写 .lp 文件（SCIP 原生方法，格式固定）──
        self.model.writeProblem(lp_path)

        # ── Step 5：写 .json 文件（语义信息）──
        self._write_json(json_path, instance_name, **kwargs)

        return {"lp": lp_path, "json": json_path}

    # ─────────────────────────────────────────────────────────────────
    #  私有辅助方法（子类通常不需要覆写）
    # ─────────────────────────────────────────────────────────────────

    def _write_json(self, json_path: str, instance_name: str, **kwargs):
        """
        将语义信息写入 .json 文件。

        文件结构（数据宪法规定，不可随意更改）：
          {
            "metadata": {
              "problem_type": "employee_scheduling",
              "instance_name": "emp_001",
              "num_variables": 600,
              "seed": 42,
              ...kwargs（如 num_employees, num_shifts）
            },
            "nodes": [ {"id": "employee_0", "type": "Employee", "features": [...]}, ... ],
            "edges": [ {"src": "employee_0", "dst": "shift_0", "rel": "assigned_to"}, ... ],
            "variable_map": [
              {"var_index": 0, "mappings": [{"type": "Employee", "id": "emp_0"}, {"type": "Shift", "id": "shift_0"}]},
              ...
            ]
          }

        variable_map 是连接语义层和数学层的核心桥梁：
          - var_index : 该变量在 SCIP 模型中的列号（0-indexed）
          - mappings  : 该变量关联的业务实体列表，每项包含：
              - type : 节点类型（与 nodes[].type 一致）
              - id   : 节点 ID（与 nodes[].id 一致）
            图构建器据此建立 variable_node → entity_node 的桥接边。
            类型信息是必需的——仅靠整数 ID 无法区分不同类型的实体。
        """
        # 调用子类实现，获取节点和边
        nodes = self._build_json_nodes()
        edges = self._build_json_edges()

        # 构建 variable_map：直接从 var_list 中提取 mappings
        # var_list[k] 已经是 {"business_key": ..., "mappings": [...]} 格式
        variable_map = []
        for idx, entry in enumerate(self.var_list):
            variable_map.append({
                "var_index": idx,
                "mappings":  entry["mappings"],
            })

        # 组装完整 JSON 结构
        payload = {
            "metadata": {
                "problem_type":   self._get_problem_type(),
                "instance_name":  instance_name,
                "num_variables":  len(self.var_list),
                "seed":           self.seed,
                **kwargs,   # 透传领域参数，如 num_employees=20
            },
            "nodes":        nodes,
            "edges":        edges,
            "variable_map": variable_map,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            # indent=2 保证可读性，sort_keys 保证顺序稳定（方便 diff）
            json.dump(payload, f, indent=2, sort_keys=False, ensure_ascii=False)
