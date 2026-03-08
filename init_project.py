#!/usr/bin/env python3
"""
OntoBranch-2026 项目初始化脚本

创建异构图神经网络 Learning-to-Branch 项目的目录结构。
将业务本体语义融入分支决策的突破性研究项目。

Author: AI Research Team
Date: February 2026
"""

import os
from pathlib import Path
import json
from typing import List, Dict, Any


class ProjectInitializer:
    """OntoBranch-2026 项目初始化器"""
    
    def __init__(self, root_path: str = "."):
        self.root = Path(root_path).resolve()
        self.project_name = "OntoBranch-2026"
        
    def create_directories(self) -> None:
        """创建项目目录结构"""
        directories = [
            # 源代码目录
            "src/generator",           # 数据生成（LP文件 + Ontology JSON）
            "src/graph",              # 异构图构建（Ontology-Variable-Constraint）
            "src/model",              # GNN模型（基于PyG HeteroData）
            "src/trainer",            # 训练循环
            "src/utils",              # 工具函数
            
            # 数据目录
            "data/raw",               # 原始数据（LP文件 + JSON本体）
            "data/processed",         # 预处理后的异构图数据
            "data/splits",            # 训练/验证/测试划分
            
            # 实验相关
            "experiments/configs",    # 实验配置文件
            "experiments/logs",       # 训练日志
            "experiments/checkpoints", # 模型检查点
            "experiments/results",    # 实验结果
            
            # 文档和脚本
            "scripts",                # 运行脚本
            "notebooks",              # Jupyter notebooks
            "docs",                   # 文档
            
            # 测试
            "tests/unit",             # 单元测试
            "tests/integration",      # 集成测试
        ]
        
        print(f"🚀 初始化 {self.project_name} 项目...")
        
        for directory in directories:
            dir_path = self.root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
            
            # 创建 __init__.py 文件（对于 Python 包）
            if directory.startswith("src/") or directory.startswith("tests/"):
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text(f'"""{"子包模块"}"""n')
                    
    def create_config_files(self) -> None:
        """创建配置文件"""
        
        # 项目配置
        project_config = {
            "project": {
                "name": "OntoBranch-2026",
                "description": "Ontology-Enhanced Learning-to-Branch via Heterogeneous GNNs",
                "version": "0.1.0",
                "authors": ["AI Research Team"],
                "keywords": ["Learning-to-Branch", "GNN", "Ontology", "Heterogeneous Graph"]
            },
            "data": {
                "problem_types": ["MILP", "Knapsack", "SetCover", "VehicleRouting"],
                "ontology_types": ["BusinessEntity", "Resource", "Constraint", "Objective"],
                "graph_types": ["Variable", "Constraint", "Entity"]
            },
            "model": {
                "backbone": "HeteroGNN",
                "node_types": ["variable", "constraint", "entity"],
                "edge_types": ["var_constraint", "entity_var", "entity_constraint"],
                "hidden_dim": 64,
                "num_layers": 3
            }
        }
        
        config_path = self.root / "experiments/configs/project_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(project_config, f, indent=2, ensure_ascii=False)
        print(f"✅ 创建配置: experiments/configs/project_config.json")
        
        # Hydra 配置模板
        hydra_config = """# @package _global_

# OntoBranch-2026 默认配置
defaults:
  - data: default
  - model: hetero_gnn
  - trainer: default
  - _self_

# 实验设置
experiment:
  name: ontobranch_baseline
  seed: 42
  device: auto  # cuda, cpu, auto
  
# 数据设置
data:
  problem_size: [50, 100, 200]  # 变量数量
  ontology_density: 0.3         # 本体连接密度
  batch_size: 32
  
# 训练设置  
trainer:
  max_epochs: 100
  learning_rate: 0.001
  patience: 10
  
# 日志设置
logging:
  wandb:
    project: ontobranch-2026
    entity: null
    mode: online  # online, offline, disabled
"""
        
        hydra_path = self.root / "experiments/configs/config.yaml"
        hydra_path.write_text(hydra_config)
        print(f"✅ 创建配置: experiments/configs/config.yaml")
        
    def create_readme(self) -> None:
        """创建 README 文件"""
        readme_content = f"""# {self.project_name}

**Ontology-Enhanced Learning-to-Branch via Heterogeneous Graph Neural Networks**

🏆 **目标**: NeurIPS/ICML 2026 顶会论文

## 🚀 核心创新

将业务本体（Ontology）融入 Learning-to-Branch，构建 **Ontology-Variable-Constraint** 三层异构图：

```
Business Entities (本体层)     ←→ Variables (变量层)     ←→ Constraints (约束层)
    员工、车辆、任务                x₁, x₂, ..., xₙ           c₁, c₂, ..., cₘ
```

## 📁 项目结构

```
{self.project_name}/
├── src/
│   ├── generator/         # 数据生成（LP + Ontology JSON）
│   ├── graph/            # 异构图构建 (PyG HeteroData)
│   ├── model/            # GNN 模型
│   └── trainer/          # 训练流程
├── data/
│   ├── raw/             # 原始 LP 文件 & JSON 本体
│   └── processed/       # 预处理异构图数据
├── experiments/         # 实验配置 & 结果
└── scripts/            # 运行脚本
```

## 🛠️ 环境设置

```bash
# 1. 创建 Conda 环境
conda env create -f environment.yml
conda activate ontobranch-2026

# 2. 验证关键依赖
python -c "import ecole; print('✅ Ecole OK')"
python -c "import torch_geometric; print('✅ PyG OK')"

# 3. 初始化项目（如果还未运行）
python init_project.py
```

## 🔬 技术栈

- **🔗 Ecole**: 与 SCIP 求解器交互的 SOTA 库
- **🧠 PyTorch Geometric**: 异构图神经网络 (`HeteroData`)
- **⚡ SCIP/Gurobi**: 线性规划求解器
- **📊 Weights & Biases**: 实验跟踪

## 📊 实验计划

1. **基线对比**: 与 Gasse et al. (2019) 对比
2. **消融实验**: 本体信息的贡献度
3. **可解释性**: 异构图中的注意力可视化
4. **泛化性**: 跨领域问题的迁移能力

## 🎯 预期贡献

1. **理论**: 异构图在组合优化中的应用
2. **方法**: Ontology-GNN 架构设计
3. **实证**: 多个工业场景的验证
4. **开源**: 完整的研究代码库

---

*让我们一起推进 Learning-to-Branch 的边界！* 🚀
"""
        
        readme_path = self.root / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"✅ 创建文档: README.md")
        
    def create_gitignore(self) -> None:
        """创建 .gitignore 文件"""
        gitignore_content = """# OntoBranch-2026 .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Experiments & Logs
experiments/logs/
experiments/checkpoints/*.pth
experiments/results/*.json
*.log

# Data (保护隐私)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv
env/
venv/

# Weights & Biases
wandb/

# Gurobi
gurobi.log
*.lp
*.mps
*.sol

# SCIP
*.scip
"""
        
        gitignore_path = self.root / ".gitignore"
        gitignore_path.write_text(gitignore_content)
        print(f"✅ 创建文档: .gitignore")
        
    def create_placeholder_files(self) -> None:
        """创建占位文件保持目录结构"""
        placeholders = [
            ("data/raw/.gitkeep", "# 保持目录结构"),
            ("data/processed/.gitkeep", "# 保持目录结构"), 
            ("experiments/logs/.gitkeep", "# 保持目录结构"),
            ("experiments/checkpoints/.gitkeep", "# 保持目录结构"),
            ("experiments/results/.gitkeep", "# 保持目录结构"),
        ]
        
        for file_path, content in placeholders:
            full_path = self.root / file_path
            full_path.write_text(content)
            print(f"✅ 创建占位: {file_path}")
            
    def run(self) -> None:
        """执行完整的项目初始化"""
        print(f"🎯 初始化路径: {self.root}")
        print("=" * 60)
        
        self.create_directories()
        print()
        
        self.create_config_files()
        print()
        
        self.create_readme()
        self.create_gitignore()
        self.create_placeholder_files()
        
        print("\n" + "=" * 60)
        print(f"🎉 {self.project_name} 项目初始化完成！")
        print("\n📋 后续步骤:")
        print("1️⃣  conda env create -f environment.yml")
        print("2️⃣  conda activate ontobranch-2026") 
        print("3️⃣  验证环境: python -c 'import ecole; print(\"Ecole OK\")'")
        print("4️⃣  开始编码: 从 src/generator/ 开始！")
        print("\n🚀 祝你研究顺利，冲击顶会！")


if __name__ == "__main__":
    initializer = ProjectInitializer()
    initializer.run()