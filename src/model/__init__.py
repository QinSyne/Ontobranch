"""
OntoBranch-2026 模型模块

核心模型:
  - OntoGNN: 三明治架构 GNN（语义编码 → 语义注入 → 数学推理）
"""

from .ontognn import OntoGNN

__all__ = ["OntoGNN"]