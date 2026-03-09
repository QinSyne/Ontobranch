#!/usr/bin/env python3
"""
EmployeeSchedulingGenerator v3.0 —— 员工排班 MILP 实例生成器（USG 范式）
继承 BaseGenerator v3.0，实现模板方法的 6 个抽象接口。
========================== USG 关键变化 ==========================
  - 所有 Employee / Shift 节点在 JSON 中统一输出为 type="entity"
  - 原始类型信息通过 entity_type_idx 编码到 128 维特征的 One-hot 前缀
  - 所有语义边在 JSON 中统一输出为 rel="relates_to"
  - variable_map 中的 mappings.type 统一为 "entity"
  entity_type_idx 分配：
    Employee = 0
    Shift    = 1

核心数学建模（与 v2 完全一致）
================================
决策变量：x[i][j] ∈ {0, 1}，表示"员工 i 是否被分配到班次 j"

约束类型：
  ① 覆盖约束（每个班次至少 min_coverage 个人）
  ② 冲突约束（同一员工在同一天最多上 1 个班次）
  ③ 浮点加权技能约束（打破 TU 的关键！）
  ④ 全局紧致预算约束

目标函数：minimize 总排班成本

Author: OntoBranch-2026 Team
"""

import random
import numpy as np
from typing import Any, Dict, List, Tuple
from pyscipopt import Model, quicksum

from .base_generator import BaseGenerator


# ─────────────────────────────────────────────────────────────────────
#  常量定义
# ─────────────────────────────────────────────────────────────────────

# 技能等级 → 浮点加权系数（非整数，破坏 TU）
SKILL_WEIGHTS = {
    "junior": 1.13,
    "senior": 2.47,
    "expert": 3.89,
}

SKILL_LEVELS = list(SKILL_WEIGHTS.keys())

# 班次时段名称
PERIODS = ["morning", "afternoon", "evening", "night"]

# ─── USG 实体类型编号（在 TYPE_DIM=16 的 One-hot 空间中的位置）───
EMPLOYEE_TYPE_IDX = 0
SHIFT_TYPE_IDX    = 1


class EmployeeSchedulingGenerator(BaseGenerator):
    """
    员工排班问题的 MILP 实例生成器（USG 范式）。

    用法示例：
        gen = EmployeeSchedulingGenerator(seed=42)
        paths = gen.generate(
            output_dir="data/raw/employee_scheduling",
            instance_name="es_001",
            num_employees=20,
            num_shifts=30,
        )
    """

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ①：生成业务实体
    # ─────────────────────────────────────────────────────────────────

    def _generate_entities(
        self,
        num_employees: int = 20,
        num_shifts: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        随机生成员工和班次实体。

        Employee 属性：name, skill_level, skill_weight, hourly_rate
        Shift 属性：name, day, period, period_index, min_coverage, skill_threshold
        """
        employees = {}
        for i in range(num_employees):
            skill = random.choice(SKILL_LEVELS)
            employees[i] = {
                "name":         f"emp_{i}",
                "skill_level":  skill,
                "skill_weight": SKILL_WEIGHTS[skill],
                "hourly_rate":  round(random.uniform(5.0, 20.0), 2),
            }

        shifts = {}
        for j in range(num_shifts):
            mc = random.randint(1, 2)
            shifts[j] = {
                "name":            f"shift_{j}",
                "day":             j // len(PERIODS),
                "period":          PERIODS[j % len(PERIODS)],
                "period_index":    j % len(PERIODS),
                "min_coverage":    mc,
                "skill_threshold": round(mc * 2.15, 2),
            }

        return {"employees": employees, "shifts": shifts}

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ②：建立变量映射（USG 格式）
    # ─────────────────────────────────────────────────────────────────

    def _generate_variables(self) -> Tuple[Dict[Tuple[int, int], int], List[Dict]]:
        """
        枚举所有 (员工, 班次) 组合作为 0/1 决策变量。

        ★ USG 变化：mappings 中的 type 统一为 "entity" ★
        原始类型信息已编码到 features 的 One-hot 前缀中，无需在此区分。
        """
        employees = self.entities["employees"]
        shifts    = self.entities["shifts"]

        var_index: Dict[Tuple[int, int], int] = {}
        var_list:  List[Dict] = []

        idx = 0
        for emp_id in sorted(employees.keys()):
            for shift_id in sorted(shifts.keys()):
                var_index[(emp_id, shift_id)] = idx
                var_list.append({
                    "business_key": (emp_id, shift_id),
                    "var_name":     f"x_{emp_id}_{shift_id}",   # 人类可读，流水线不读
                    "mappings": [
                        {"id": f"emp_{emp_id}",    "semantic_type": "Employee"},
                        {"id": f"shift_{shift_id}", "semantic_type": "Shift"},
                        # ★ type 字段已省略；builder 固定构建为 entity 节点
                    ],
                })
                idx += 1

        return var_index, var_list

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ③：构建 SCIP 模型（与 v2 逻辑完全一致）
    # ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> Model:
        """
        构建完整的 SCIP MILP 模型。

        ★ 遵守强制对齐契约：按 self.var_list 顺序添加变量 ★
        数学建模部分与 v2 完全一致，USG 不影响数学层。
        """
        employees = self.entities["employees"]
        shifts    = self.entities["shifts"]

        model = Model("employee_scheduling")
        model.setMinimize()

        # ── 添加决策变量（严格按 var_list 顺序）──
        scip_vars = {}
        for k, entry in enumerate(self.var_list):
            emp_id, shift_id = entry["business_key"]
            scip_vars[k] = model.addVar(
                name=f"x_{emp_id}_{shift_id}",
                vtype="B",
                lb=0.0,
                ub=1.0,
            )

        def x(emp_id, shift_id):
            """快捷查找：(emp_id, shift_id) → SCIP 变量对象"""
            return scip_vars[self.var_index[(emp_id, shift_id)]]

        # ── 约束 ①：覆盖约束 ──
        for j, shift in shifts.items():
            model.addCons(
                quicksum(x(i, j) for i in employees) >= shift["min_coverage"],
                name=f"coverage_{j}",
            )

        # ── 约束 ②：冲突约束（同一员工同一天 ≤1 班）──
        day_to_shifts: Dict[int, List[int]] = {}
        for j, shift in shifts.items():
            day_to_shifts.setdefault(shift["day"], []).append(j)

        for i in employees:
            for day, shift_ids in day_to_shifts.items():
                if len(shift_ids) > 1:
                    model.addCons(
                        quicksum(x(i, j) for j in shift_ids) <= 1,
                        name=f"conflict_{i}_{day}",
                    )

        # ── 约束 ③：浮点加权技能约束（破坏 TU）──
        for j, shift in shifts.items():
            model.addCons(
                quicksum(
                    employees[i]["skill_weight"] * x(i, j)
                    for i in employees
                ) >= shift["skill_threshold"],
                name=f"skill_{j}",
            )

        # ── 约束 ④：全局紧致预算约束（破坏 TU）──
        total_min_coverage = sum(s["min_coverage"] for s in shifts.values())
        avg_rate = np.mean([e["hourly_rate"] for e in employees.values()])
        budget = round(total_min_coverage * avg_rate * 1.3, 2)

        model.addCons(
            quicksum(
                employees[entry["business_key"][0]]["hourly_rate"] * scip_vars[k]
                for k, entry in enumerate(self.var_list)
            ) <= budget,
            name="budget",
        )

        # ── 目标函数：最小化总排班成本 ──
        model.setObjective(
            quicksum(
                employees[entry["business_key"][0]]["hourly_rate"] * scip_vars[k]
                for k, entry in enumerate(self.var_list)
            ),
            sense="minimize",
        )

        return model

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ④：构建 JSON 节点（USG 格式）
    # ─────────────────────────────────────────────────────────────────

    def _build_json_nodes(self) -> List[Dict]:
        """
        将员工和班次实体转成 USG 格式的 nodes 列表。

        ★ USG 关键变化：
          - type 统一为 "entity"
          - features 经过 _harmonize_features() 对齐到 128 维
          - 原始类型通过 entity_type_idx 编码为 One-hot 前缀

        Employee 原始特征（5 维）：
          [skill_weight, hourly_rate, is_junior, is_senior, is_expert]
          → _harmonize_features(0, raw) → 128 维

        Shift 原始特征（5 维）：
          [min_coverage, skill_threshold, period_index, day_normalized, 1.0]
          → _harmonize_features(1, raw) → 128 维
        """
        employees = self.entities["employees"]
        shifts    = self.entities["shifts"]
        nodes     = []

        # ── Employee 节点 → entity (type_idx=0) ──
        for emp_id, emp in employees.items():
            is_junior = 1.0 if emp["skill_level"] == "junior" else 0.0
            is_senior = 1.0 if emp["skill_level"] == "senior" else 0.0
            is_expert = 1.0 if emp["skill_level"] == "expert" else 0.0

            raw_features = [
                emp["skill_weight"],
                emp["hourly_rate"],
                is_junior,
                is_senior,
                is_expert,
            ]

            nodes.append({
                "id":       f"emp_{emp_id}",
                "type":     "entity",                                         # ★ USG：统一为 entity
                "features": self._harmonize_features(EMPLOYEE_TYPE_IDX, raw_features),  # ★ 128 维
            })

        # ── Shift 节点 → entity (type_idx=1) ──
        total_days = max(1, len(shifts) // len(PERIODS))
        for shift_id, shift in shifts.items():
            raw_features = [
                float(shift["min_coverage"]),
                shift["skill_threshold"],
                float(shift["period_index"]),
                shift["day"] / max(1, total_days - 1) if total_days > 1 else 0.0,
                1.0,   # 偏置项
            ]

            nodes.append({
                "id":       f"shift_{shift_id}",
                "type":     "entity",                                      # ★ USG：统一为 entity
                "features": self._harmonize_features(SHIFT_TYPE_IDX, raw_features),  # ★ 128 维
            })

        return nodes

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ⑤：构建 JSON 边（USG 格式）
    # ─────────────────────────────────────────────────────────────────

    def _build_json_edges(self) -> List[Dict]:
        """
        构建业务实体之间的语义关系边。

        ★ USG 契约："rel" 统一为 "relates_to"，流水线只读此字段。
        ★ 人类可读："semantic_rel" 标注具体业务语义，流水线忽略。

        边类型（双向）：
          same_day  : 同一天的两个班次之间，对应冲突约束来源
          can_cover : 员工技能满足班次单人要求时（skill_weight >=
                      skill_threshold / min_coverage），员工↔班次互连
        """
        shifts    = self.entities["shifts"]
        employees = self.entities["employees"]
        edges: List[Dict] = []

        # ── ① same_day：同一天班次互连（双向）──
        day_to_shifts: Dict[int, List[int]] = {}
        for j, shift in shifts.items():
            day_to_shifts.setdefault(shift["day"], []).append(j)

        for shift_ids in day_to_shifts.values():
            for a_idx in range(len(shift_ids)):
                for b_idx in range(a_idx + 1, len(shift_ids)):
                    a, b = shift_ids[a_idx], shift_ids[b_idx]
                    for src, dst in ((a, b), (b, a)):
                        edges.append({
                            "src":          f"shift_{src}",
                            "dst":          f"shift_{dst}",
                            "semantic_rel": "same_day",
                        })

        # ── ② can_cover：技能满足单人要求的员工↔班次（双向）──
        for i, emp in employees.items():
            for j, shift in shifts.items():
                per_person_req = shift["skill_threshold"] / max(1, shift["min_coverage"])
                if emp["skill_weight"] >= per_person_req:
                    for src_id, dst_id in (
                        (f"emp_{i}", f"shift_{j}"),
                        (f"shift_{j}", f"emp_{i}"),
                    ):
                        edges.append({
                            "src":          src_id,
                            "dst":          dst_id,
                            "semantic_rel": "can_cover",
                        })

        return edges

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ⑥：问题类型标识
    # ─────────────────────────────────────────────────────────────────

    def _build_annotation(self) -> Dict:
        """
        人类可读注解：特征维度说明 + 边类型说明。
        流水线不读取此字段（_ 前缀约定）。
        """
        return {
            "_note": "此字段仅供人类阅读，流水线忽略所有 _annotation 内容",
            "feature_schema": {
                "dims_0_15":  "One-hot 实体类型前缀（TYPE_DIM=16）：Employee=dim0，Shift=dim1",
                "Employee": {
                    "dim_16": "skill_weight（junior=1.13 / senior=2.47 / expert=3.89）",
                    "dim_17": "hourly_rate（随机 5~20 的时薪）",
                    "dim_18": "is_junior（0/1）",
                    "dim_19": "is_senior（0/1）",
                    "dim_20": "is_expert（0/1）",
                    "dims_21_127": "Zero-padding",
                },
                "Shift": {
                    "dim_16": "min_coverage（最少需要几人覆盖）",
                    "dim_17": "skill_threshold（技能加权最低总量要求）",
                    "dim_18": "period_index（morning=0 / afternoon=1 / evening=2 / night=3）",
                    "dim_19": "day_normalized（班次所在天数归一化到 [0,1]）",
                    "dim_20": "bias=1.0（常数偏置）",
                    "dims_21_127": "Zero-padding",
                },
            },
            "semantic_rel_schema": {
                "same_day":  "同一天的两个班次，对应冲突约束（Shift↔Shift，双向）",
                "can_cover": "员工技能满足班次单人门槛（skill_weight >= skill_threshold/min_coverage），Emp↔Shift，双向",
            },
        }

    def _get_problem_type(self) -> str:
        """返回问题类型标识。"""
        return "employee_scheduling"
