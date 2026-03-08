#!/usr/bin/env python3
"""
OntoGNN 模型冒烟测试 (Smoke Test)

验证内容:
  1. load_and_build 加载 JSON → HeteroData（无 ecole_obs，使用占位特征）
  2. create_ontognn 工厂函数正确实例化模型
  3. Forward Pass 运行无报错
  4. 输出形状 == data['variable'].num_nodes
  5. 输出无 NaN
  6. Ablation: use_ontology=False 也能跑通，且参数量明显少于完整模型
  7. 覆盖所有 6 种问题类型

Author: OntoBranch-2026 Team
Date: February 2026
"""

import os
import sys
import time
import glob
import torch

# ── 路径设置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from graph.builder import load_and_build
from model.ontognn import OntoGNN, create_ontognn


# ─────────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> int:
    """统计模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt_params(n: int) -> str:
    """格式化参数量显示。"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def find_test_files(data_dir: str) -> dict:
    """
    在 data_dir 下查找每种问题类型的第一个 JSON 文件。
    返回 {problem_type: json_path}。
    """
    result = {}
    if not os.path.isdir(data_dir):
        return result
    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        jsons = sorted(glob.glob(os.path.join(subdir_path, "*.json")))
        if jsons:
            result[subdir] = jsons[0]
    return result


# ─────────────────────────────────────────────────────────────────────
#  单问题类型测试
# ─────────────────────────────────────────────────────────────────────

def test_single_problem(problem_type: str, json_path: str, hidden_dim: int = 64):
    """
    对单个问题类型执行完整冒烟测试。

    Returns:
        dict: 测试结果摘要
    """
    print(f"\n{'─' * 60}")
    print(f"  Testing: {problem_type}")
    print(f"  File:    {os.path.basename(json_path)}")
    print(f"{'─' * 60}")

    # ── Step 1: 加载数据（无 ecole_obs → 占位特征） ──
    data, stats = load_and_build(json_path, ecole_obs=None, verbose=False)

    num_vars = data["variable"].x.shape[0]
    num_cons = data["constraint"].x.shape[0]
    num_node_types = len(data.node_types)
    num_edge_types = len(data.edge_types)

    print(f"  Graph:   {num_vars} vars, {num_cons} cons, "
          f"{num_node_types} node types, {num_edge_types} edge types")
    print(f"  Semantic entities: {stats['entity_counts']}")

    # ── Step 2: 实例化完整模型 (use_ontology=True) ──
    model_full = create_ontognn(data, hidden_dim=hidden_dim, use_ontology=True)
    params_full = count_params(model_full)

    # ── Step 3: Forward Pass ──
    model_full.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        scores_full = model_full(data)
    t_full = (time.perf_counter() - t0) * 1000  # ms

    # ── Step 4: 断言检查 ──
    # 4a. 形状
    assert scores_full.shape == (num_vars,), (
        f"[FAIL] 输出形状 {scores_full.shape} != 期望 ({num_vars},)"
    )
    # 4b. 无 NaN
    assert not torch.isnan(scores_full).any(), (
        f"[FAIL] 输出包含 NaN!"
    )
    # 4c. 无 Inf
    assert not torch.isinf(scores_full).any(), (
        f"[FAIL] 输出包含 Inf!"
    )

    print(f"  [Full]   params={fmt_params(params_full)}, "
          f"forward={t_full:.2f}ms, "
          f"output={scores_full.shape}, "
          f"range=[{scores_full.min():.4f}, {scores_full.max():.4f}]")

    # ── Step 5: Ablation (use_ontology=False) ──
    model_base = create_ontognn(data, hidden_dim=hidden_dim, use_ontology=False)
    params_base = count_params(model_base)

    model_base.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        scores_base = model_base(data)
    t_base = (time.perf_counter() - t0) * 1000

    # 断言: 形状 + NaN
    assert scores_base.shape == (num_vars,), (
        f"[FAIL] Baseline 输出形状 {scores_base.shape} != ({num_vars},)"
    )
    assert not torch.isnan(scores_base).any(), "[FAIL] Baseline 输出包含 NaN!"

    # 断言: Baseline 参数量应严格少于完整模型
    assert params_base < params_full, (
        f"[FAIL] Baseline ({params_base}) 参数量应少于 Full ({params_full})"
    )

    reduction = (1 - params_base / params_full) * 100

    print(f"  [Base]   params={fmt_params(params_base)}, "
          f"forward={t_base:.2f}ms, "
          f"output={scores_base.shape}, "
          f"range=[{scores_base.min():.4f}, {scores_base.max():.4f}]")
    print(f"  [Δ]      Full vs Base: {fmt_params(params_full)} vs {fmt_params(params_base)} "
          f"(-{reduction:.1f}% params)")

    return {
        "problem_type": problem_type,
        "num_variables": num_vars,
        "num_constraints": num_cons,
        "params_full": params_full,
        "params_base": params_base,
        "time_full_ms": round(t_full, 2),
        "time_base_ms": round(t_base, 2),
        "status": "PASS",
    }


# ─────────────────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  OntoGNN Smoke Test — Forward Pass Verification")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    test_files = find_test_files(data_dir)

    if not test_files:
        print(f"\n[ERROR] 在 {data_dir} 下未找到测试数据。")
        print("请先运行: python scripts/generate_training_data.py --output-dir data/raw --instances-per-type 3")
        sys.exit(1)

    print(f"\n发现 {len(test_files)} 种问题类型的测试数据")

    results = []
    passed = 0
    failed = 0

    for problem_type, json_path in test_files.items():
        try:
            result = test_single_problem(problem_type, json_path)
            results.append(result)
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {problem_type}: {e}")
            results.append({"problem_type": problem_type, "status": "FAIL", "error": str(e)})
            failed += 1

    # ── 汇总报告 ──
    print(f"\n{'=' * 60}")
    print("  Summary Report")
    print(f"{'=' * 60}")

    # 打印完整模型结构（仅第一个）
    if results and results[0]["status"] == "PASS":
        first_json = list(test_files.values())[0]
        data, _ = load_and_build(first_json)
        model_demo = create_ontognn(data, hidden_dim=64, use_ontology=True)
        print("\n  Model Architecture (sample — employee_scheduling):")
        print("  " + "-" * 56)
        for line in str(model_demo).split("\n"):
            print(f"    {line}")
        print("  " + "-" * 56)

    # 参数量对比表
    print(f"\n  {'Problem Type':<25} {'Vars':>5} {'Full':>10} {'Base':>10} {'Δ%':>7} {'Fwd(ms)':>8}")
    print(f"  {'─' * 25} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 7} {'─' * 8}")
    for r in results:
        if r["status"] == "PASS":
            delta = (1 - r["params_base"] / r["params_full"]) * 100
            print(f"  {r['problem_type']:<25} {r['num_variables']:>5} "
                  f"{fmt_params(r['params_full']):>10} {fmt_params(r['params_base']):>10} "
                  f"{delta:>6.1f}% {r['time_full_ms']:>7.2f}")
        else:
            print(f"  {r['problem_type']:<25}  FAIL — {r.get('error', 'unknown')[:40]}")

    print(f"\n  Results: {passed} PASSED, {failed} FAILED, {passed + failed} TOTAL")

    if failed == 0:
        print("\n  ✅ All smoke tests passed!")
        print("  OntoGNN is ready to connect to Ecole for training.")
    else:
        print(f"\n  ❌ {failed} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
