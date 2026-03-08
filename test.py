import json
import re

def check_data_integrity(json_path, lp_path):
    print(f"🕵️ 正在审计: {json_path}")
    
    # 1. 读取 JSON 中的变量名
    with open(json_path, 'r') as f:
        data = json.load(f)
    json_vars = set(data['mathematical_model']['variables'].keys())
    print(f"   ✅ JSON 包含 {len(json_vars)} 个变量定义")

    # 2. 读取 LP 文件中的变量名 (简单的正则提取)
    lp_vars = set()
    with open(lp_path, 'r') as f:
        content = f.read()
        # 寻找形如 "x_1_2" 或 "assign_emp1_shift1" 的变量
        # SCIP LP 格式通常在 Bounds 或 Binary 部分列出变量
        # 这里做一个简单的全文本搜索匹配 JSON 里的名字
        for var in json_vars:
            if var in content:
                lp_vars.add(var)
            else:
                print(f"   ❌ 警告: 变量 {var} 在 JSON 中存在，但在 LP 文件中未找到！")
    
    # 3. 验证一致性
    missing_in_lp = json_vars - lp_vars
    if not missing_in_lp:
        print("   🎉 验证通过：JSON 与 LP 变量名完全一致！")
        return True
    else:
        print(f"   😱 验证失败：有 {len(missing_in_lp)} 个变量在 LP 中丢失。")
        return False

# 运行测试
if __name__ == "__main__":
    # 替换为你实际生成的文件路径
    check_data_integrity(
        "data/raw/employee_scheduling/employee_scheduling_001.json",
        "data/raw/employee_scheduling/employee_scheduling_001.lp"
    )