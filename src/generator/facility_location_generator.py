#!/usr/bin/env python3
"""
FacilityLocationGenerator —— USG 范式设施选址问题生成器

生成单源带容量限制的设施选址问题 (Capacitated Facility Location Problem, CFLP)
变量:
  - y[i]: 设施 i 是否开放 (Binary)
  - x[i, j]: 客户 j 是否由设施 i 供货 (Binary, 单源)

USG 映射:
  - 设施 (Facility) 和 客户 (Customer) 均作为 "entity" 节点。
  - facility (type_idx = 0)，customer (type_idx = 1)
  - 包含距离相近的相互排斥或同一区域的边缘 (competing_facility, same_region)
"""

import math
from typing import Any, Dict, List, Tuple
from pyscipopt import Model

from src.generator.base_generator import BaseGenerator, GLOBAL_ENT_DIM

class FacilityLocationGenerator(BaseGenerator):

    def _get_problem_type(self) -> str:
        return "facility_location"

    def _build_annotation(self) -> Dict[str, Any]:
        return {
            "feature_schema": {
                " facility (type_idx=0)": "[one_hot] + [fixed_cost, capacity, x, y] + [0]*112",
                " customer (type_idx=1)": "[one_hot] + [demand, x, y] + [0]*112"
            },
            "semantic_rel_schema": {
                "competing_facility": "两个设施之间空间距离极近",
                "same_region": "两个客户之间空间距离极近"
            }
        }

    def _generate_entities(self, num_facilities: int, num_customers: int, **kwargs) -> Dict[str, Any]:
        """生成设施和客户特征"""
        import random
        facilities = {}
        # 为了保证问题有解，总容量需要大于总需求
        # 先生成客户，再生成设施
        
        customers = {}
        total_demand = 0
        for j in range(num_customers):
            cx = random.uniform(0, 100)
            cy = random.uniform(0, 100)
            demand = random.uniform(10, 50)
            total_demand += demand
            customers[f"customer_{j}"] = {
                "id": f"customer_{j}",
                "demand": demand,
                "x": cx,
                "y": cy
            }
            
        avg_capacity_needed = (total_demand / num_facilities) * 1.5 # 适度拥挤
        
        for i in range(num_facilities):
            fx = random.uniform(0, 100)
            fy = random.uniform(0, 100)
            fc = random.uniform(1000, 5000)
            cap = random.uniform(avg_capacity_needed * 0.8, avg_capacity_needed * 1.2)
            facilities[f"facility_{i}"] = {
                "id": f"facility_{i}",
                "fixed_cost": fc,
                "capacity": cap,
                "x": fx,
                "y": fy
            }
            
        return {
            "facilities": facilities,
            "customers": customers
        }

    def _generate_variables(self) -> Tuple[Dict[Any, int], List[Dict]]:
        var_index = {}
        var_list = []
        idx = 0
        
        facilities = self.entities["facilities"]
        customers = self.entities["customers"]
        
        # 1. 设施开放变量 y[i]
        for f_id in facilities.keys():
            var_index[("y", f_id)] = idx
            var_list.append({
                "business_key": ("y", f_id),
                "var_name": f"y_{f_id}",
                "mappings": [
                    {"id": f_id, "type": "entity"}
                ]
            })
            idx += 1
            
        # 2. 供货变量 x[i, j]
        for f_id in facilities.keys():
            for c_id in customers.keys():
                var_index[("x", f_id, c_id)] = idx
                var_list.append({
                    "business_key": ("x", f_id, c_id),
                    "var_name": f"x_{f_id}_{c_id}",
                    "mappings": [
                        {"id": f_id, "type": "entity"},
                        {"id": c_id, "type": "entity"}
                    ]
                })
                idx += 1
                
        return var_index, var_list

    def _build_model(self) -> Any:
        model = Model(self._get_problem_type())
        model.hideOutput()

        facilities = self.entities["facilities"]
        customers = self.entities["customers"]

        # 创建变量，严格按照 var_list 顺序
        scip_vars = {}
        for k, entry in enumerate(self.var_list):
            v_name = entry["var_name"]
            scip_vars[k] = model.addVar(vtype="B", name=v_name)
            
        y_vars = {}
        x_vars = {}
        
        for f_id in facilities.keys():
            y_i = scip_vars[self.var_index[("y", f_id)]]
            y_vars[f_id] = y_i
            
        for f_id in facilities.keys():
            for c_id in customers.keys():
                x_ij = scip_vars[self.var_index[("x", f_id, c_id)]]
                x_vars[(f_id, c_id)] = x_ij

        # 目标函数：最小化建造成本 + 运输成本
        obj_expr = 0
        for f_id, f_data in facilities.items():
            obj_expr += f_data["fixed_cost"] * y_vars[f_id]
            
        for f_id, f_data in facilities.items():
            for c_id, c_data in customers.items():
                dist = math.hypot(f_data["x"] - c_data["x"], f_data["y"] - c_data["y"])
                transport_cost = dist * c_data["demand"] * 0.1 # 单位距离单位需求成本
                obj_expr += transport_cost * x_vars[(f_id, c_id)]
                
        model.setObjective(obj_expr, "minimize")

        # 约束1：每个客户必须被且仅被一个设施服务
        for c_id in customers.keys():
            model.addCons(
                sum(x_vars[(f_id, c_id)] for f_id in facilities.keys()) == 1,
                name=f"Demand_{c_id}"
            )

        # 约束2：设施容量限制
        for f_id, f_data in facilities.items():
            model.addCons(
                sum(customers[c_id]["demand"] * x_vars[(f_id, c_id)] for c_id in customers.keys()) <= f_data["capacity"] * y_vars[f_id],
                name=f"Capacity_{f_id}"
            )
            
        return model

    def _build_json_nodes(self) -> List[Dict]:
        nodes = []
        
        # 设施 type_idx = 0
        for f_id, f_data in self.entities["facilities"].items():
            raw_feat = [f_data["fixed_cost"], f_data["capacity"], f_data["x"], f_data["y"]]
            nodes.append({
                "id": f_id,
                "type": "entity",
                "features": self._harmonize_features(0, raw_feat)
            })
            
        # 客户 type_idx = 1
        for c_id, c_data in self.entities["customers"].items():
            raw_feat = [c_data["demand"], c_data["x"], c_data["y"]]
            nodes.append({
                "id": c_id,
                "type": "entity",
                "features": self._harmonize_features(1, raw_feat)
            })
            
        return nodes

    def _build_json_edges(self) -> List[Dict]:
        edges = []
        facilities = list(self.entities["facilities"].values())
        customers = list(self.entities["customers"].values())

        # competing_facility: 设施间距离 < 20
        for i in range(len(facilities)):
            for j in range(i + 1, len(facilities)):
                dist = math.hypot(facilities[i]["x"] - facilities[j]["x"], facilities[i]["y"] - facilities[j]["y"])
                if dist < 20:
                    edges.append({
                        "src": facilities[i]["id"],
                        "dst": facilities[j]["id"],
                        "rel": "relates_to",
                        "semantic_rel": "competing_facility"
                    })
                    edges.append({
                        "src": facilities[j]["id"],
                        "dst": facilities[i]["id"],
                        "rel": "relates_to",
                        "semantic_rel": "competing_facility"
                    })
                    
        # same_region: 客户间距离 < 15
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                dist = math.hypot(customers[i]["x"] - customers[j]["x"], customers[i]["y"] - customers[j]["y"])
                if dist < 15:
                    edges.append({
                        "src": customers[i]["id"],
                        "dst": customers[j]["id"],
                        "rel": "relates_to",
                        "semantic_rel": "same_region"
                    })
                    edges.append({
                        "src": customers[j]["id"],
                        "dst": customers[i]["id"],
                        "rel": "relates_to",
                        "semantic_rel": "same_region"
                    })
                    
        return edges
