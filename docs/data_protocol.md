# OntoBranch-2026 数据协议 (Data Protocol v3.0)

> **版本历史**
> - v1.0：初版，旧格式（`ontology.entities` + `mathematical_model.variables`）
> - v2.0：全面重构，引入 `nodes / edges / variable_map` 三段式结构
> - v2.1：补充边特征规范；明确 Builder 处理策略
> - **v3.0（当前）：实施 USG（通用语义图）范式。所有业务节点统一为 entity，所有语义边统一为 relates_to。模型参数结构完全静态化，彻底消灭领域灾难（Domain Catastrophe）。**

本协议定义 **Generator → Builder → Model** 全链路的数据格式契约。  
Generator 负责生成，Builder 绝对信任并依赖此格式，Model 通过 PyG HeteroData 消费。

---

## 一、核心原则

### 1. 通用语义图范式 (Universal Semantic Graph, USG)
**这是 v3.0 最核心的变化。**

整个系统中只允许存在以下固定的图拓扑结构：

| 层次 | 节点类型 | 边类型 | 说明 |
|---|---|---|---|
| 数学层 | `variable` | `constrains` | SCIP 变量 ↔ 约束 |
| 数学层 | `constraint` | | |
| 语义层 | `entity` | `relates_to` | **所有业务节点统称**（不再区分 Employee/Shift 等） |
| 桥接层 | | `mapped_to` | variable → entity 的桥接|

**绝对禁止**在 Builder 或 Model 中出现任何领域特化的节点/边类型名称。

### 2. 特征统一化与补齐 (Feature Harmonization & Zero-Padding)

引入全局配置常量（定义在 `src/generator/base_generator.py`）：
```python
GLOBAL_ENT_DIM = 128   # entity 特征总维度
TYPE_DIM = 16           # 实体类型 One-hot 编码维度
```

所有 entity 节点的 `features` 严格对齐到 128 维：

| 区间 | 维度 | 内容 |
|---|---|---|
| `[0 : 16]` | 16 | 实体类型 One-hot 编码（如 Employee → 第 0 位，Shift → 第 1 位） |
| `[16 : 128]` | 112 | 原始数值特征 + Zero-padding |

### 3. 绝对对齐契约 (Strict Alignment Contract)
**违反将导致训练数据静默出错。**

```
var_list 枚举顺序
  = model.addVar() 调用顺序
  = variable_map[k]["var_index"] == k
  = Ecole variable_features[k] 对应的变量
```

三者共享同一个 `self.var_list` 的遍历顺序，**绝对禁止另起独立循环**。

### 4. 特征完备性 (Feature Completeness)
- **entity 节点**：所有节点 `features` 长度恒为 `GLOBAL_ENT_DIM` (128)。  
- **语义边**：同一 `rel`（统一为 `relates_to`）的所有边，要么全部携带 `features`（且长度相等），要么全部不携带。

---

## 二、JSON 文件格式（v3.0 完整规范）

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
      "type": "entity",
      "features": [1.0, 0.0, 0.0, ..., 1.13, 11.89, 1.0, 0.0, 0.0, ..., 0.0]
    },
    {
      "id": "shift_0",
      "type": "entity",
      "features": [0.0, 1.0, 0.0, ..., 1.0, 2.15, 0.0, 0.0, 1.0, ..., 0.0]
    }
  ],

  "edges": [
    {
      "src": "shift_0",
      "dst": "shift_1",
      "rel": "relates_to"
    }
  ],

  "variable_map": [
    {
      "var_index": 0,
      "mappings": [
        {"type": "entity", "id": "emp_0"},
        {"type": "entity", "id": "shift_0"}
      ]
    }
  ]
}
```

### 字段说明

#### `metadata`（必填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `problem_type` | string | 问题类型，全小写下划线 |
| `instance_name` | string | 实例名，全局唯一 |
| `num_variables` | int | 决策变量总数 |
| `seed` | int | 可选，生成随机种子 |

#### `nodes`（必填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | **全局**唯一字符串 ID，如 `emp_0`、`shift_3` |
| `type` | string | **固定为 `"entity"`**（USG 契约） |
| `features` | float[128] | **恒定 128 维**（[0:16] One-hot + [16:128] 原始特征 + zero-padding） |

#### `edges`（可选）

| 字段 | 类型 | 说明 |
|---|---|---|
| `src` | string | 源节点的全局唯一 ID |
| `dst` | string | 目标节点的全局唯一 ID |
| `rel` | string | **固定为 `"relates_to"`**（USG 契约） |
| `features` | float[] | **可选**。边特征。同一 rel 要么全有，要么全无 |

#### `variable_map`（必填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `var_index` | int | 从 0 开始的连续整数 |
| `mappings` | list | 每项含 `type`（固定为 `"entity"`） + `id` |

---

## 三、Builder 行为规范（`src/graph/builder.py`）

Builder 将 `(JSON, Ecole obs)` → PyG `HeteroData`，输出 **结构绝对固定** 的三层图。

### 输出拓扑（恒定不变）

```
节点：variable[N, 19]  constraint[C, 5]  entity[E, 128]
边  ：("variable", "constrains", "constraint")
      ("entity", "relates_to", "entity")
      ("variable", "mapped_to", "entity")
```

### 构建顺序

```
① 数学层  _build_math_layer()    → variable + constraint + constrains 边
② 语义层  _build_semantic_layer() → entity + relates_to 边
③ 桥接层  _build_bridge_layer()  → mapped_to 边
```

### ID 转换机制

Builder 维护单一 flat 映射表（不再按类型分桶）：

```python
_id_to_idx["emp_0"]   = 0
_id_to_idx["shift_0"] = 20
```

所有 entity 共享同一个局部索引空间。

### 数学层安全处理

- Ecole 特征用 `np.nan_to_num(..., nan=0.0)` 清洗 NaN  
- 空值检查用 `if x is None`，**禁止** `if not x`  
- `ecole_obs=None` 时退化为全零占位：`variable[N, 19]`, `constraint[1, 5]`

### 便捷接口

```python
from src.graph.builder import load_and_build

data = load_and_build(
    json_path="data/raw/employee_scheduling/employee_scheduling_001.json",
    ecole_obs=None,
    verbose=True,
)
# data.node_types  → ['variable', 'constraint', 'entity']
# data.edge_types  → [('variable','constrains','constraint'),
#                     ('entity','relates_to','entity'),
#                     ('variable','mapped_to','entity')]
```

---

## 四、Generator 实现规范（`src/generator/`）

### BaseGenerator 模板方法调用顺序

```
generate()
  ├─ _generate_entities()    填充 self.entities
  ├─ _generate_variables()   填充 self.var_list（mappings.type 统一为 "entity"）
  ├─ _build_model()          按 var_list 顺序 addVar()，⚠️ 禁止另起循环
  ├─ _write_lp()             写出 .lp 文件
  └─ _write_json()           写出 USG 格式 .json（含安全校验）
```

### `var_list` 元素结构（v3.0）

```python
{
    "business_key": (emp_id, shift_id),
    "mappings": [
        {"type": "entity", "id": "emp_0"},
        {"type": "entity", "id": "shift_0"},
    ]
}
```

### 特征对齐方法

`BaseGenerator._harmonize_features(entity_type_idx, raw_features)` → `List[float]`

```
输入：entity_type_idx=0, raw_features=[1.13, 11.89, 1.0, 0.0, 0.0]
输出：[1.0, 0.0, 0.0, ...(16维), 1.13, 11.89, 1.0, 0.0, 0.0, 0.0, ...(共128维)]
```

### 节点输出规则

| 字段 | v2.x | v3.0 |
|---|---|---|
| `type` | 业务名称如 `"Employee"` | **固定 `"entity"`** |
| `features` 维度 | 各类型自定（如 5 维） | **恒定 128 维** |
| `rel` | 业务名称如 `"same_day"` | **固定 `"relates_to"`** |

---

## 五、Model 架构规范（`src/model/ontognn.py`）

### OntoGNN v3.0 —— 终极静态模型

**不再接受 metadata 参数。不再使用 LazyLinear。参数结构完全固定。**

| Stage | 名称 | 输入维度 | 输出维度 | 实现 |
|---|---|---|---|---|
| 0 | 特征投影 | var:19, con:5, ent:128 | 全部 H | 3 个静态 `nn.Linear` |
| 1 | 语义编码 | entity [N_ent, H] | entity [N_ent, H] | `GATConv` + 残差 |
| 2 | 语义注入 | entity → variable | variable [N_var, H] | 手工 Q-K-V 稀疏注意力 |
| 3 | 数学推理 | var ↔ con bipartite | var [N_var, H] | `GATConv` 二分图 + 残差 |
| 4 | 决策头 | variable [N_var, H] | scores [N_var] | MLP: H → H → 1 |

### 使用示例

```python
from src.model.ontognn import OntoGNN

model = OntoGNN(hidden_dim=64)   # 不需要 metadata！
scores = model(data)             # data 来自 UniversalGraphBuilder
```

---

## 六、各场景 entity_type_idx 分配

### Employee Scheduling

| 原始类型 | entity_type_idx | 原始特征维度 | 原始特征含义 |
|---|---|---|---|
| Employee | 0 | 5 | skill_weight, hourly_rate, is_junior, is_senior, is_expert |
| Shift | 1 | 5 | min_coverage, skill_threshold, period_index, day_normalized, 1.0 |

| 语义边 | JSON `rel` | 说明 |
|---|---|---|
| 同天班次 | `relates_to` | 同天两两相连（双向） |

---

## 七、禁止事项（红线清单）

1. **禁止**在 JSON 中使用 `"entity"` 以外的节点 `type`  
2. **禁止**在 JSON 中使用 `"relates_to"` 以外的边 `rel`  
3. **禁止** entity 节点的 `features` 长度不等于 128  
4. **禁止**在 `_build_model()` 内另起独立循环添加变量  
5. **禁止**`variable_map` 的 `var_index` 跳号或乱序  
6. **禁止**`variable_map` 中 `mappings.type` 使用 `"entity"` 以外的值  
7. **禁止**在 Builder 中用 `if not x` 判断 numpy array  
8. **禁止**在 Model 中使用 `LazyLinear` 或接受 `metadata` 参数  
9. **禁止**在 Model 或 Builder 中硬编码业务名词（如 `Employee`、`Shift`）