# OntoBranch-2026 数据宪法 (Data Protocol v2.1)

> **版本历史**
> - v1.0：初版，旧格式（`ontology.entities` + `mathematical_model.variables`）
> - v2.0：全面重构，引入 `nodes / edges / variable_map` 三段式结构
> - v2.1（当前）：补充边特征（`edge.features`）规范；明确 Builder 的处理策略；补全各模块的接口摘要

本宪法定义 **Generator → Builder → Model** 全链路的数据格式契约。  
Generator 负责生成，Builder 绝对信任并依赖此格式，Model 通过 PyG HeteroData 消费。

---

## 一、核心原则

### 1. 领域无关性 (Domain-Agnostic)
禁止在架构层（Builder、Model）硬编码任何业务名词（如 `Employee`、`Shift`）。  
节点类型、关系类型完全由 JSON 数据动态驱动，Builder 一律按字符串处理。

### 2. 绝对对齐契约 (Strict Alignment Contract)
**这是本宪法最重要的约束，违反将导致训练数据静默出错。**

```
var_list 枚举顺序
  = model.addVar() 调用顺序
  = variable_map[k]["var_index"] == k
  = Ecole variable_features[k] 对应的变量
```

三者共享同一个 `self.var_list` 的遍历顺序，**绝对禁止另起独立循环**。

### 3. 特征完备性 (Feature Completeness)
- **同类型节点**：同一 `type` 的所有节点，`features` 列表长度必须完全相等，缺项补 `0.0`。  
- **同类型边**：同一 `rel` 的所有边，要么全部携带 `features`（且长度相等），要么全部不携带。  
  部分有、部分无 → Builder 跳过 `edge_attr`（不补零，视为上游 bug）。

---

## 二、JSON 文件格式（完整规范）

每个实例输出一对文件：`<name>.json` + `<name>.lp`，存放于 `data/raw/<problem_type>/`。

```json
{
  "metadata": {
    "problem_type": "employee_scheduling",
    "instance_name": "employee_scheduling_001",
    "num_variables": 300,
    "seed": 100
  },

  "nodes": [
    {
      "id": "emp_0",
      "type": "Employee",
      "features": [1.13, 11.89, 1.0, 0.0, 0.0]
    },
    {
      "id": "shift_0",
      "type": "Shift",
      "features": [1.0, 2.15, 0, 0.0, 1.0]
    }
  ],

  "edges": [
    {
      "src": "shift_0",
      "dst": "shift_1",
      "rel": "same_day"
    },
    {
      "src": "loc_0",
      "dst": "loc_1",
      "rel": "road",
      "features": [12.5, 0.8]
    }
  ],

  "variable_map": [
    {
      "var_index": 0,
      "mappings": [
        {"type": "Employee", "id": "emp_0"},
        {"type": "Shift",    "id": "shift_0"}
      ]
    }
  ]
}
```

### 字段说明

#### `metadata`（必填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `problem_type` | string | 问题类型，与目录名一致，全小写下划线 |
| `instance_name` | string | 实例名，全局唯一，与文件名去掉扩展名一致 |
| `num_variables` | int | 决策变量总数，必须与 `.lp` 文件的列数严格一致 |
| `seed` | int | 可选，生成随机种子，透传存档即可 |

#### `nodes`（必填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | **全局**唯一字符串 ID，建议加类型前缀，如 `emp_0`、`shift_3` |
| `type` | string | 节点类型，首字母大写，如 `Employee`、`Location` |
| `features` | float[] | 数值特征向量，**同类型节点维度必须相等** |

#### `edges`（可选，无则省略整个字段）

| 字段 | 类型 | 说明 |
|---|---|---|
| `src` | string | 源节点的全局唯一 ID |
| `dst` | string | 目标节点的全局唯一 ID |
| `rel` | string | 关系类型字符串，可自定义，如 `same_day`、`road` |
| `features` | float[] | **可选**。边特征（如距离、成本）。**同一 rel 要么全有，要么全无** |

#### `variable_map`（必填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `var_index` | int | 从 0 开始的连续整数，= SCIP 变量列号，= Ecole obs 特征行号 |
| `mappings` | list | 该变量在业务上关联的实体列表，每项含 `type` + `id` |

`variable_map` 必须按 `var_index` 升序排列，且无跳号（0, 1, 2, …, N-1）。

---

## 三、Builder 行为规范（`src/graph/builder.py`）

Builder 将 `(JSON, Ecole obs)` → PyG `HeteroData`，构建三层图：

### 构建顺序

```
① 数学层  _build_math_layer()   → variable[N,19] + constraint[C,5]
                                  + ('variable','constrains','constraint')
② 语义层  _build_semantic_layer()→ Employee[E,5] + Shift[S,5] + ...
                                  + ('Shift','same_day','Shift') + ...
③ 桥接层  _build_bridge_layer() → ('variable','mapped_to','Employee') + ...
```

### ID 转换机制

Builder 内部维护两层查找表：

```python
_id_to_idx["Employee"]["emp_0"] = 0   # 局部索引，从 0 开始
_id_to_idx["Shift"]["shift_3"]  = 3
```

不同类型的局部索引相互独立，与 PyG HeteroData 的节点矩阵行号严格对应。

### 边特征策略

- 若同一 `rel` 的所有边均携带 `features` → 写出 `edge_attr: Tensor[num_edges, feat_dim]`  
- 若存在任意一条边缺失 `features` → **跳过** `edge_attr`，不补零，日志打印 `(no edge_attr)`

### 数学层安全处理

- 所有 Ecole 特征用 `np.nan_to_num(..., nan=0.0)` 清洗 NaN  
- 空值检查用 `if x is None`，**禁止** `if not x`（numpy array 布尔歧义）  
- `ecole_obs=None` 时，`variable.x` 退化为全零占位 `[N, 1]`

### 便捷接口

```python
from src.graph.builder import load_and_build

data = load_and_build(
    json_path="data/raw/employee_scheduling/employee_scheduling_001.json",
    ecole_obs=None,   # 可传入 Ecole NodeBipartite 观测
    verbose=True,
)
# data.node_types  → ['variable', 'constraint', 'Employee', 'Shift']
# data.edge_types  → [('variable','constrains','constraint'),
#                     ('Shift','same_day','Shift'),
#                     ('variable','mapped_to','Employee'),
#                     ('variable','mapped_to','Shift')]
```

---

## 四、Generator 实现规范（`src/generator/`）

### BaseGenerator 模板方法调用顺序

```
generate()
  ├─ _generate_entities()   填充 self.employees / self.shifts / ...
  ├─ _generate_variables()  填充 self.var_list（每项含 business_key + mappings）
  ├─ _build_model()         按 var_list 顺序 addVar()，⚠️ 禁止另起循环
  ├─ _write_lp()            写出 .lp 文件（pyscipopt model.writeProblem）
  └─ _write_json()          写出 .json，variable_map 直接从 var_list 生成
```

### `var_list` 元素结构

```python
{
    "business_key": (emp_id, shift_id),   # 调试用，不写入 JSON
    "mappings": [
        {"type": "Employee", "id": "emp_0"},
        {"type": "Shift",    "id": "shift_0"},
    ]
}
```

### 节点特征设计规则

各场景的节点特征由 `_build_json_nodes()` 负责计算并直接嵌入 JSON，  
Builder 无需再做任何归一化或编码——**特征工程的职责在 Generator，不在 Builder**。

### TU-Breaking 策略（员工排班示例）

员工排班约束矩阵原本是全幺模（TU）的，整数松弛等价于 LP 松弛，训练数据退化。  
通过以下两处打破：

1. **浮点技能权重**：`junior=1.13, senior=2.47, expert=3.89`（非整数）  
2. **全局预算约束**：总成本 ≤ min_feasible_cost × 1.3

---

## 五、各场景节点特征摘要

### Employee Scheduling

| 节点类型 | 特征维度 | 特征含义（按列顺序） |
|---|---|---|
| `Employee` | 5 | skill_weight, hourly_rate, is_junior, is_senior, is_expert |
| `Shift` | 5 | min_coverage, skill_threshold, period_index, day_normalized, 1.0（常数偏置） |

| 边类型 | `rel` | `features` | 说明 |
|---|---|---|---|
| `('Shift','same_day','Shift')` | `same_day` | 无 | 同天两两相连（双向） |

---

## 六、禁止事项（红线清单）

1. **禁止**在 `_build_model()` 内另起独立 `for emp in ...` 循环添加变量——必须用 `for k, entry in enumerate(self.var_list)`  
2. **禁止**同类型节点的 `features` 长度不一致  
3. **禁止**同一 `rel` 的边部分有 `features`、部分无 `features`  
4. **禁止**`variable_map` 的 `var_index` 跳号或乱序  
5. **禁止**在 Builder 中用 `if not x` 判断 numpy array（用 `if x is None`）  
6. **禁止**在 Builder 或 Model 中硬编码节点类型名称（应动态读取 `data.node_types`）