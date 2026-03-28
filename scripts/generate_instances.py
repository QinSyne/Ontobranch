#!/usr/bin/env python3
"""
generate_instances.py —— USG 范式高强度预训练数据批量生成脚本

职责：
  严格基于高强度网格配置（Grid Configurations），批量生成：
   - 员工排班 (ES)
   - 设施选址 (FLP)
   两类多规模 NP-Hard MILP 问题的 .lp 和 .json 文件。

重点要求：
  - 严禁调用任何生态求解器（如 Ecole / SCIP solve），彻底杜绝在此图生成阶段的任何求解推断。
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# 解决路径依赖
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.employee_scheduling_generator import EmployeeSchedulingGenerator
from src.generator.facility_location_generator import FacilityLocationGenerator

# ==============================================================================
#  网格配置参数
# ==============================================================================

# ES (排班) 配置: (员工数, 班次数)
ES_GRID = {
    "Medium": {"num_employees": 50, "num_shifts": 150},
    "Large": {"num_employees": 100, "num_shifts": 300},
    "Hardcore": {"num_employees": 150, "num_shifts": 450},
}

# FLP (选址) 配置: (客户数, 备选设施数)
FLP_GRID = {
    "Medium": {"num_customers": 100, "num_facilities": 30},
    "Large": {"num_customers": 200, "num_facilities": 60},
    "Hardcore": {"num_customers": 300, "num_facilities": 100},
}

NUM_INSTANCES_PER_CONFIG = 20
BASE_SEED = 20260329

ES_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "employee_scheduling")
FLP_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "facility_location")

# ==============================================================================

def ensure_folders():
    for d in [ES_OUTPUT_DIR, FLP_OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

def update_progress_bar(current: int, total: int, prefix_text: str = "", bar_length: int = 40):
    progress = current / total
    block = int(round(bar_length * progress))
    current_str = f"{current:02d}"
    text = f"\r{prefix_text} [{'=' * block}{'.' * (bar_length - block)}] ({current_str}/{total})"
    sys.stdout.write(text)
    sys.stdout.flush()
    if current == total:
        print() # New line when finished

def main():
    print("=" * 70)
    print("🚀 开始高强度规模网格数据批量生成 (USG Graph Only) 🚀")
    print(f"每种配置生成 {NUM_INSTANCES_PER_CONFIG} 份实例...")
    print("=" * 70)
    
    ensure_folders()
    
    es_generator = EmployeeSchedulingGenerator()
    flp_generator = FacilityLocationGenerator()

    total_instances = 0
    start_time = time.time()

    # --------------------------------------------------------------------------
    # 1. 批量生成 ES 实例
    # --------------------------------------------------------------------------
    print("\n--- 任务阶段 1: Employee Scheduling (ES) ---")
    es_seed_offset = 0
    for level, config in ES_GRID.items():
        prefix = f"[ES-{level}] "
        # 固定前缀长度对齐打印
        prefix = f"{prefix:<15}"
        for i in range(1, NUM_INSTANCES_PER_CONFIG + 1):
            seed = BASE_SEED + es_seed_offset
            instance_name = f"es_{level.lower()}_{i:03d}"
            
            es_generator.generate(
                output_dir=ES_OUTPUT_DIR,
                instance_name=instance_name,
                num_employees=config["num_employees"],
                num_shifts=config["num_shifts"],
                random_seed=seed
            )
            es_seed_offset += 1
            total_instances += 1
            
            update_progress_bar(i, NUM_INSTANCES_PER_CONFIG, prefix)

    # --------------------------------------------------------------------------
    # 2. 批量生成 FLP 实例
    # --------------------------------------------------------------------------
    print("\n--- 任务阶段 2: Facility Location Problem (FLP) ---")
    flp_seed_offset = 5000 # 避开 ES 的种子段
    for level, config in FLP_GRID.items():
        prefix = f"[FLP-{level}]"
        prefix = f"{prefix:<15}"
        for i in range(1, NUM_INSTANCES_PER_CONFIG + 1):
            seed = BASE_SEED + flp_seed_offset
            instance_name = f"flp_{level.lower()}_{i:03d}"
            
            flp_generator.generate(
                output_dir=FLP_OUTPUT_DIR,
                instance_name=instance_name,
                num_customers=config["num_customers"],
                num_facilities=config["num_facilities"],
                random_seed=seed
            )
            flp_seed_offset += 1
            total_instances += 1
            
            update_progress_bar(i, NUM_INSTANCES_PER_CONFIG, prefix)

    # --------------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 成功完成！共计生成 {total_instances} 个复合配置实例。")
    print(f"⏱️ 耗时：{elapsed_time:.2f} 秒。")
    print("======================================================================")

if __name__ == '__main__':
    main()
