#!/usr/bin/env python3
"""
EmployeeSchedulingGenerator —— 员工排班 MILP 实例生成器

继承 BaseGenerator，实现模板方法的 6 个抽象接口。

核心数学建模
============
决策变量：x[i][j] ∈ {0, 1}，表示"员工 i 是否被分配到班次 j"

约束类型：
  ① 覆盖约束（每个班次至少 min_coverage 个人）
  ② 冲突约束（同一员工在同一天最多上 1 个班次）
  ③ 浮点加权技能约束（打破 TU 的关键！）
     每个班次的"加权技能总分"≥ 阈值
     权重使用带小数扰动的浮点数（1.13 / 2.47 / 3.89），而非整数
  ④ 全局紧致预算约束（全部排班总成本 ≤ 理论最小成本 × 1.3）

目标函数：minimize 总排班成本

关于全幺模性（TU）
==================
如果只有覆盖约束（系数全 1）和冲突约束（系数全 1），约束矩阵是全幺模的，
LP 松弛直接得到整数最优解，SCIP 根本不需要分支。
浮点技能约束（系数 1.13, 2.47, 3.89）和预算约束（系数为 hourly_rate）
引入非整数系数，彻底破坏 TU 性质。

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

# 技能等级 → 浮点加权系数
# 必须是"不整的"小数，目的就是让约束矩阵不再是 0/1 矩阵
# 选值原则：彼此不成整数倍关系，且与 1/2/3 明显不同
SKILL_WEIGHTS = {
    "junior": 1.13,
    "senior": 2.47,
    "expert": 3.89,
}

# 技能等级列表（按从低到高排列，用于随机选择）
SKILL_LEVELS = list(SKILL_WEIGHTS.keys())

# 班次时段名称（纯语义信息，会写进 JSON 供 GNN 作为节点特征参考）
PERIODS = ["morning", "afternoon", "evening", "night"]


class EmployeeSchedulingGenerator(BaseGenerator):
    """
    员工排班问题的 MILP 实例生成器。

    用法示例：
        gen = EmployeeSchedulingGenerator(seed=42)
        paths = gen.generate(
            output_dir="data/raw/employee_scheduling",
            instance_name="es_001",
            num_employees=20,
            num_shifts=30,
        )
        # paths = {"lp": "...es_001.lp", "json": "...es_001.json"}
    """

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ①：生成业务实体
    # ─────────────────────────────────────────────────────────────────

    def _generate_entities(
        self,
        num_employees: int = 20,
        num_shifts: int = 30,
        **kwargs,                # 接收并忽略 Base 透传的其他参数
    ) -> Dict[str, Any]:
        """
        随机捏造员工和班次。

        员工属性：
          - name         : 便于人读，如 "emp_0"
          - skill_level  : "junior" / "senior" / "expert"（随机分配）
          - skill_weight : 对应的浮点权重（1.13 / 2.47 / 3.89）
                           ★ 存进实体，后面建约束时直接用，不用每次反查
          - hourly_rate  : 时薪（5.0~20.0 随机浮点），用于成本目标和预算约束

        班次属性：
          - name         : 如 "shift_0"
          - day          : 属于第几天（0-indexed，每天有 len(PERIODS) 个班次）
          - period       : 时段字符串，如 "morning"
          - period_index : 时段的数值编码（0/1/2/3），给 GNN 当特征
          - min_coverage : 该班次最低需要多少人（1 或 2，不设太大防止不可行）
          - skill_threshold : 该班次"加权技能点总和"的最低要求
                              = min_coverage × 2.15（一个不整的系数）
                              这是打破 TU 的核心约束之一

        返回
        ----
        {"employees": {...}, "shifts": {...}}
        """
        employees = {}
        for i in range(num_employees):
            skill = random.choice(SKILL_LEVELS)
            employees[i] = {
                "name":         f"emp_{i}",
                "skill_level":  skill,
                "skill_weight": SKILL_WEIGHTS[skill],
                # 时薪在 [5.0, 20.0] 区间随机，保留 2 位小数
                "hourly_rate":  round(random.uniform(5.0, 20.0), 2),
            }

        shifts = {}
        # 根据班次总数和每天的时段数，算出总天数
        num_days = max(1, num_shifts // len(PERIODS))

        for j in range(num_shifts):
            mc = random.randint(1, 2)   # 最低人数覆盖需求
            shifts[j] = {
                "name":            f"shift_{j}",
                "day":             j // len(PERIODS),   # 整除 → 所属天
                "period":          PERIODS[j % len(PERIODS)],
                "period_index":    j % len(PERIODS),    # 数值编码
                "min_coverage":    mc,
                # 技能阈值 = 最低人数 × 2.15（不整的浮点数！）
                # 含义：如果需要 2 人，那要求加权技能总和 ≥ 4.30
                # junior (1.13) + junior (1.13) = 2.26 < 4.30 → 不够！
                # 必须至少有一个 senior 或 expert，这就是"技能门槛"
                "skill_threshold": round(mc * 2.15, 2),
            }

        return {"employees": employees, "shifts": shifts}

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ②：建立变量映射（遵守对齐契约）
    # ─────────────────────────────────────────────────────────────────

    def _generate_variables(self) -> Tuple[Dict[Tuple[int, int], int], List[Dict]]:
        """
        枚举所有 (员工, 班次) 组合作为 0/1 决策变量。

        ★ 遍历顺序固定为：外层 emp_id 升序，内层 shift_id 升序
        ★ 这个顺序就是 SCIP 的变量列号：
            var_list[0] = emp_0 × shift_0
            var_list[1] = emp_0 × shift_1
            ...

        var_list 中每个元素是一个结构化字典，包含：
          - business_key : (emp_id, shift_id) 元组，方便程序内部查询
          - mappings     : 带类型的实体引用列表（数据宪法要求的格式）

        var_index 中键是 (emp_id, shift_id) 元组，值是列号。
        """
        employees = self.entities["employees"]
        shifts    = self.entities["shifts"]

        var_index: Dict[Tuple[int, int], int] = {}
        var_list:  List[Dict] = []

        idx = 0
        # ★ 固定遍历顺序：emp_id 升序 → shift_id 升序
        for emp_id in sorted(employees.keys()):
            for shift_id in sorted(shifts.keys()):
                # 正向索引
                var_index[(emp_id, shift_id)] = idx

                # 反向索引：带类型的结构化描述（给 JSON variable_map 用）
                var_list.append({
                    "business_key": (emp_id, shift_id),
                    "mappings": [
                        # 第一个关联实体：Employee
                        {"type": "Employee", "id": f"emp_{emp_id}"},
                        # 第二个关联实体：Shift
                        {"type": "Shift",    "id": f"shift_{shift_id}"},
                    ],
                })

                idx += 1

        return var_index, var_list

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ③：构建 SCIP 模型
    # ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> Model:
        """
        构建完整的 SCIP MILP 模型。

        ★ 遵守强制对齐契约 ★
        添加变量的循环严格遍历 self.var_list（而非另起循环遍历 employees×shifts），
        保证 addVar() 的调用次序与 var_list 完全一致。

        约束结构（4 类）：
          ① coverage_j   : 每个班次 j 至少 min_coverage[j] 人
          ② conflict_i_d : 员工 i 在第 d 天最多上 1 班
          ③ skill_j      : 每个班次 j 的加权技能总和 ≥ skill_threshold[j]
          ④ budget        : 全局总成本 ≤ 理论最低成本 × 1.3

        目标函数：
          minimize Σ hourly_rate[i] × x[i][j]
        """
        employees = self.entities["employees"]
        shifts    = self.entities["shifts"]

        model = Model("employee_scheduling")
        model.setMinimize()   # 目标方向：最小化

        # ── 添加决策变量 ──────────────────────────────────────────
        # ★★★ 严格按 var_list 顺序 ★★★
        # scip_vars[k] 对应第 k 号变量，即 var_list[k] 描述的 (emp, shift) 分配
        scip_vars = {}
        for k, entry in enumerate(self.var_list):
            emp_id, shift_id = entry["business_key"]
            var_name = f"x_{emp_id}_{shift_id}"
            # 0/1 二元变量：vtype="B"（Binary）
            scip_vars[k] = model.addVar(
                name=var_name,
                vtype="B",      # Binary
                lb=0.0,
                ub=1.0,
            )

        # 为了后面写约束方便，也建一个 (emp_id, shift_id) → SCIP 变量 的快速查找表
        # 可以直接用 var_index 做键的反查
        def x(emp_id, shift_id):
            """快捷函数：给定 (emp_id, shift_id)，返回对应的 SCIP 变量对象"""
            k = self.var_index[(emp_id, shift_id)]
            return scip_vars[k]

        # ── 约束 ①：覆盖约束（每个班次至少 min_coverage 人）────────
        # Σ_i x[i][j] ≥ min_coverage[j]，∀ j
        # 系数全是 1（0/1 矩阵的一部分，本身不破坏 TU）
        for j, shift in shifts.items():
            model.addCons(
                quicksum(x(i, j) for i in employees) >= shift["min_coverage"],
                name=f"coverage_{j}",
            )

        # ── 约束 ②：冲突约束（同一员工同一天最多上 1 关班次）───────
        # Σ_{j ∈ day_d} x[i][j] ≤ 1，∀ i, d
        # 先按天数对班次分组
        day_to_shifts: Dict[int, List[int]] = {}
        for j, shift in shifts.items():
            day_to_shifts.setdefault(shift["day"], []).append(j)

        for i in employees:
            for day, shift_ids in day_to_shifts.items():
                if len(shift_ids) > 1:   # 当天只有 1 个班次时约束无意义
                    model.addCons(
                        quicksum(x(i, j) for j in shift_ids) <= 1,
                        name=f"conflict_{i}_{day}",
                    )

        # ── 约束 ③：浮点加权技能约束（★ 打破 TU 的核心 ★）─────────
        # Σ_i (skill_weight[i] × x[i][j]) ≥ skill_threshold[j]，∀ j
        #
        # skill_weight 是 1.13 / 2.47 / 3.89 这样的浮点数
        # skill_threshold 是 min_coverage × 2.15
        # 这些非整数系数让约束矩阵不再是 0/1 矩阵，彻底破坏 TU
        #
        # 直觉解释：不光要够人数，还要够"技能水平"。
        # 两个 junior (1.13+1.13=2.26) 凑不够一个 2 人班次的技能要求 (4.30)
        for j, shift in shifts.items():
            model.addCons(
                quicksum(
                    employees[i]["skill_weight"] * x(i, j)
                    for i in employees
                ) >= shift["skill_threshold"],
                name=f"skill_{j}",
            )

        # ── 约束 ④：全局紧致预算约束（★ 另一个破 TU 手段 ★）──────
        # Σ_{i,j} hourly_rate[i] × x[i][j] ≤ budget
        #
        # budget = 理论最低总成本 × 1.3
        # 理论最低总成本 = Σ_j min_coverage[j] × 全员平均时薪
        # 乘 1.3 留出余量，避免不可行，但又足够"紧"
        #
        # hourly_rate 是 [5.0, 20.0] 区间的浮点数，
        # 作为系数进一步向约束矩阵注入非整数值
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

        # ── 目标函数：最小化总排班成本 ────────────────────────────
        # minimize Σ_{i,j} hourly_rate[i] × x[i][j]
        model.setObjective(
            quicksum(
                employees[entry["business_key"][0]]["hourly_rate"] * scip_vars[k]
                for k, entry in enumerate(self.var_list)
            ),
            sense="minimize",
        )

        return model

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ④：构建 JSON 节点
    # ─────────────────────────────────────────────────────────────────

    def _build_json_nodes(self) -> List[Dict]:
        """
        将员工实体和班次实体转成 JSON nodes 列表。

        每个节点包含：
          - id       : 全局唯一字符串，格式与 variable_map.mappings.id 一致
          - type     : "Employee" 或 "Shift"
          - features : 数值特征列表（GNN 用这些数字作为节点初始特征）

        Employee 特征向量（5 维）：
          [skill_weight, hourly_rate, is_junior, is_senior, is_expert]
          其中 is_xxx 是 one-hot 编码（0.0 或 1.0）

        Shift 特征向量（5 维）：
          [min_coverage, skill_threshold, period_index, day_normalized, 1.0]
          最后一位是偏置项（常数 1），保证维度对齐
        """
        employees = self.entities["employees"]
        shifts    = self.entities["shifts"]
        nodes     = []

        # ── Employee 节点 ──
        for emp_id, emp in employees.items():
            # one-hot 编码技能等级
            is_junior = 1.0 if emp["skill_level"] == "junior" else 0.0
            is_senior = 1.0 if emp["skill_level"] == "senior" else 0.0
            is_expert = 1.0 if emp["skill_level"] == "expert" else 0.0

            nodes.append({
                "id":       f"emp_{emp_id}",     # 与 variable_map 中的 id 一致！
                "type":     "Employee",
                "features": [
                    emp["skill_weight"],   # 连续值特征
                    emp["hourly_rate"],    # 连续值特征
                    is_junior,             # one-hot
                    is_senior,             # one-hot
                    is_expert,             # one-hot
                ],
            })

        # ── Shift 节点 ──
        total_days = max(1, len(shifts) // len(PERIODS))
        for shift_id, shift in shifts.items():
            nodes.append({
                "id":       f"shift_{shift_id}",  # 与 variable_map 中的 id 一致！
                "type":     "Shift",
                "features": [
                    float(shift["min_coverage"]),
                    shift["skill_threshold"],
                    float(shift["period_index"]),
                    # 天数归一化到 [0, 1]，避免值域差异过大
                    shift["day"] / max(1, total_days - 1) if total_days > 1 else 0.0,
                    1.0,   # 偏置项（常数特征）
                ],
            })

        return nodes

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ⑤：构建 JSON 边
    # ─────────────────────────────────────────────────────────────────

    def _build_json_edges(self) -> List[Dict]:
        """
        构建业务实体之间的语义关系边。

        边类型 1："same_day"
          连接属于同一天的两个 Shift 节点。
          含义：这两个班次存在时间冲突（同一个员工不能同时上）。
          方向：双向（shift_a → shift_b 且 shift_b → shift_a）

        边类型 2："scheduled_on_day"
          连接每个 Shift 和一个虚拟的 Day 节点？
          —— 不需要，day 信息已编码进 Shift 的 features 里。

        目前只建 same_day 边，保持简洁。
        更复杂的边（如 "skill_compatible"）可以按需添加。
        """
        shifts = self.entities["shifts"]
        edges  = []

        # ── 按天分组 ──
        day_to_shifts: Dict[int, List[int]] = {}
        for j, shift in shifts.items():
            day_to_shifts.setdefault(shift["day"], []).append(j)

        # ── 同天的班次之间建 same_day 边 ──
        for day, shift_ids in day_to_shifts.items():
            # 两两组合（无序对，但建双向边）
            for a_idx in range(len(shift_ids)):
                for b_idx in range(a_idx + 1, len(shift_ids)):
                    a, b = shift_ids[a_idx], shift_ids[b_idx]
                    # 双向边
                    edges.append({
                        "src": f"shift_{a}",
                        "dst": f"shift_{b}",
                        "rel": "same_day",
                    })
                    edges.append({
                        "src": f"shift_{b}",
                        "dst": f"shift_{a}",
                        "rel": "same_day",
                    })

        return edges

    # ─────────────────────────────────────────────────────────────────
    #  实现抽象方法 ⑥：问题类型标识
    # ─────────────────────────────────────────────────────────────────

    def _get_problem_type(self) -> str:
        """返回问题类型标识，用于 JSON metadata。"""
        return "employee_scheduling"
