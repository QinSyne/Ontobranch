#!/usr/bin/env python3
"""
generate_instances.py —— USG 范式实例批量生成脚本

功能：
  调用 EmployeeSchedulingGenerator（v3.0 USG 版本），
  批量生成多个 MILP 实例，每个实例输出：
    ① <name>.lp    供 SCIP / Ecole 加载的数学模型
    ② <name>.json  符合 USG 协议的本体语义图（entity 节点 + relates_to 边）

生成后自动执行快速验证：
  - JSON 结构完整性（metadata / nodes / edges / variable_map）
  - entity 特征维度 == 128
  - type 全部为 "entity"，rel 全部为 "relates_to"
  - variable_map 无跳号、无乱序

用法：
  python scripts/generate_instances.py                    # 使用默认配置
  python scripts/generate_instances.py --dry-run          # 仅打印配置，不生成
  python scripts/generate_instances.py --num 10           # 生成 10 个实例

Author: OntoBranch-2026 Team
Date: 2026-03
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any

# ── 路径设置：脚本可从任意位置执行 ──
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.employee_scheduling_generator import EmployeeSchedulingGenerator
from src.generator.base_generator import GLOBAL_ENT_DIM

# ═══════════════════════════════════════════════════════════════════
#  生成配置表
#
#  每条记录对应一个待生成的实例：
#    name          : 文件名（不含扩展名），例如 "employee_scheduling_001"
#    seed          : 随机种子（保证可复现）
#    num_employees : 员工数
#    num_shifts    : 班次数
# ═══════════════════════════════════════════════════════════════════

DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    # ── 小规模实例（快速验证） ──
    {"name": "employee_scheduling_001", "seed": 100, "num_employees": 10, "num_shifts": 12},
    {"name": "employee_scheduling_002", "seed": 101, "num_employees": 15, "num_shifts": 20},
    {"name": "employee_scheduling_003", "seed": 102, "num_employees": 20, "num_shifts": 24},
    # ── 中规模实例（训练用） ──
    {"name": "employee_scheduling_004", "seed": 200, "num_employees": 30, "num_shifts": 40},
    {"name": "employee_scheduling_005", "seed": 201, "num_employees": 40, "num_shifts": 48},
]

OUTPUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "employee_scheduling")


# ───────────────────────────────────────────────────────────────────
#  验证单个实例 JSON 是否符合 USG 协议（data_protocol v3.0）
# ───────────────────────────────────────────────────────────────────

class USGProtocolValidator:
    """
    对生成的 JSON 文件执行协议合规检查。

    验证项：
      ① metadata 字段完整
      ② 所有 node.type == "entity"
      ③ 所有 node.features 维度 == GLOBAL_ENT_DIM (128)
      ④ 所有 edge.rel == "relates_to"（如有边）
      ⑤ variable_map 无跳号，var_index 连续且从 0 开始
      ⑥ variable_map 中所有 mappings.type == "entity"
    """

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.json_path = json_path
        self.errors: List[str] = []

    def validate(self) -> bool:
        """执行全部验证，返回是否通过。"""
        self._check_metadata()
        self._check_nodes()
        self._check_edges()
        self._check_variable_map()
        return len(self.errors) == 0

    def _check_metadata(self):
        meta = self.data.get("metadata", {})
        for field in ["problem_type", "instance_name", "num_variables"]:
            if field not in meta:
                self.errors.append(f"metadata 缺少字段: {field}")

    def _check_nodes(self):
        nodes = self.data.get("nodes", [])
        if not nodes:
            self.errors.append("nodes 列表为空")
            return
        for i, node in enumerate(nodes):
            if node.get("type") != "entity":
                self.errors.append(
                    f"nodes[{i}] (id={node.get('id')}) type={node.get('type')} != 'entity'"
                )
            feat = node.get("features", [])
            if len(feat) != GLOBAL_ENT_DIM:
                self.errors.append(
                    f"nodes[{i}] (id={node.get('id')}) features 维度={len(feat)} != {GLOBAL_ENT_DIM}"
                )

    def _check_edges(self):
        # v3.1：rel 字段已从 JSON 中省略（builder 固定构建为 relates_to）
        # 此处只检查必要字段 src / dst 存在，并确认 semantic_rel 合法
        valid_rels = {"same_day", "can_cover"}
        for i, edge in enumerate(self.data.get("edges", [])):
            if "src" not in edge or "dst" not in edge:
                self.errors.append(f"edges[{i}] 缺少 src 或 dst 字段")
            sr = edge.get("semantic_rel")
            if sr is not None and sr not in valid_rels:
                self.errors.append(
                    f"edges[{i}] semantic_rel='{sr}' 不在已知集合 {valid_rels}"
                )

    def _check_variable_map(self):
        var_map = self.data.get("variable_map", [])
        if not var_map:
            self.errors.append("variable_map 为空")
            return
        # 检查连续性
        for expected_idx, entry in enumerate(var_map):
            if entry.get("var_index") != expected_idx:
                self.errors.append(
                    f"variable_map[{expected_idx}].var_index={entry.get('var_index')} 不连续"
                )
        # v3.1：mappings 不再含 type 字段（builder 固定为 entity）
        # 只检查每条 mapping 必须有 id
        for i, entry in enumerate(var_map):
            for j, mapping in enumerate(entry.get("mappings", [])):
                if "id" not in mapping:
                    self.errors.append(
                        f"variable_map[{i}].mappings[{j}] 缺少 id 字段"
                    )


# ───────────────────────────────────────────────────────────────────
#  生成单个实例
# ───────────────────────────────────────────────────────────────────

def generate_one(cfg: Dict[str, Any], output_dir: str, verbose: bool = True) -> Dict[str, Any]:
    """
    生成一个 MILP 实例，返回结果摘要字典。

    参数
    ----
    cfg : 生成配置，包含 name, seed, num_employees, num_shifts
    output_dir : 输出目录
    verbose    : 是否打印详细信息

    返回
    ----
    result : 包含路径、统计、验证状态的摘要字典
    """
    name          = cfg["name"]
    seed          = cfg["seed"]
    num_employees = cfg["num_employees"]
    num_shifts    = cfg["num_shifts"]

    gen = EmployeeSchedulingGenerator(seed=seed)

    t0 = time.perf_counter()
    paths = gen.generate(
        output_dir=output_dir,
        instance_name=name,
        num_employees=num_employees,
        num_shifts=num_shifts,
    )
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    # ── 统计数据 ──
    json_path = paths["json"]
    lp_path   = paths["lp"]

    with open(json_path, "r", encoding="utf-8") as f:
        jdata = json.load(f)

    num_entity_nodes = len(jdata.get("nodes", []))
    num_edges        = len(jdata.get("edges", []))
    num_variables    = jdata["metadata"]["num_variables"]

    # 计算 entity 特征的非零率（One-hot 前缀）
    nodes = jdata.get("nodes", [])
    if nodes:
        # 验证 One-hot 前缀正确性（每个 entity 特征向量前 16 维应恰好有一个 1）
        onehot_ok = all(
            sum(n["features"][:16]) == 1.0 for n in nodes
        )
    else:
        onehot_ok = False

    # LP 文件大小
    lp_size = os.path.getsize(lp_path)

    # ── USG 协议验证 ──
    validator = USGProtocolValidator(json_path)
    valid     = validator.validate()

    result = {
        "name":          name,
        "seed":          seed,
        "employees":     num_employees,
        "shifts":        num_shifts,
        "variables":     num_variables,
        "entities":      num_entity_nodes,
        "edges":         num_edges,
        "onehot_ok":     onehot_ok,
        "lp_kb":         round(lp_size / 1024, 1),
        "elapsed_ms":    round(elapsed, 1),
        "usg_valid":     valid,
        "errors":        validator.errors,
        "json_path":     json_path,
        "lp_path":       lp_path,
    }

    if verbose:
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"  [{status}] {name}")
        print(f"           employees={num_employees}, shifts={num_shifts}, "
              f"vars={num_variables}")
        print(f"           entities={num_entity_nodes}(128-dim), edges={num_edges}, "
              f"one-hot={'OK' if onehot_ok else 'FAIL'}")
        print(f"           LP={result['lp_kb']}KB  |  生成耗时={result['elapsed_ms']}ms")
        if not valid:
            for err in validator.errors:
                print(f"           [ERROR] {err}")

    return result


# ───────────────────────────────────────────────────────────────────
#  批量生成主函数
# ───────────────────────────────────────────────────────────────────

def generate_all(
    configs: List[Dict[str, Any]],
    output_dir: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    批量生成所有实例，打印完整汇总报告。
    """
    print("=" * 65)
    print("  OntoBranch-2026 | USG 实例批量生成器 (data_protocol v3.0)")
    print("=" * 65)
    print(f"  输出目录  : {output_dir}")
    print(f"  实例数量  : {len(configs)}")
    print(f"  GLOBAL_ENT_DIM : {GLOBAL_ENT_DIM}")
    print()

    if dry_run:
        print("[DRY RUN] 以下配置将被生成：")
        for cfg in configs:
            print(f"  {cfg['name']}  seed={cfg['seed']}  "
                  f"employees={cfg['num_employees']}  shifts={cfg['num_shifts']}")
        return

    os.makedirs(output_dir, exist_ok=True)

    results   = []
    total_t0  = time.perf_counter()

    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] 生成中...")
        result = generate_one(cfg, output_dir, verbose=verbose)
        results.append(result)
        print()

    total_elapsed = (time.perf_counter() - total_t0) * 1000

    # ── 汇总报告 ──
    n_pass = sum(1 for r in results if r["usg_valid"])
    n_fail = len(results) - n_pass

    print("=" * 65)
    print("  汇总报告")
    print("=" * 65)
    print(f"  总实例数   : {len(results)}")
    print(f"  USG 验证   : {n_pass} PASS  /  {n_fail} FAIL")
    print(f"  总耗时     : {total_elapsed:.1f}ms")
    print()
    print(f"  {'实例名':<36} {'vars':>6} {'ent':>4} {'edges':>5} {'LP(KB)':>7} {'USG':>5}")
    print(f"  {'-'*36} {'-'*6} {'-'*4} {'-'*5} {'-'*7} {'-'*5}")
    for r in results:
        usg_str = "✓" if r["usg_valid"] else "✗"
        print(f"  {r['name']:<36} {r['variables']:>6} {r['entities']:>4} "
              f"{r['edges']:>5} {r['lp_kb']:>7.1f} {usg_str:>5}")
    print()

    if n_fail == 0:
        print("  ✓ 所有实例均符合 USG 数据协议 v3.0")
    else:
        print(f"  ✗ {n_fail} 个实例存在协议违规，请检查上方 [ERROR] 信息")

    print(f"\n  文件路径：{output_dir}/")
    print("=" * 65)


# ───────────────────────────────────────────────────────────────────
#  CLI 入口
# ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="OntoBranch-2026 USG 实例批量生成脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="限制生成数量（默认：DEFAULT_CONFIGS 全部）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印配置，不实际生成文件",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"输出目录（默认：{OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="减少输出详细度",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    configs = DEFAULT_CONFIGS
    if args.num is not None:
        configs = configs[: args.num]

    generate_all(
        configs=configs,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
