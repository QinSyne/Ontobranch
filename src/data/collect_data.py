import os
import glob
import json
import time
import torch
import ecole

def collect_expert_labels(data_dir: str):
    """
    通过 Ecole 和 SCIP 后端，提取数据集全集中所有 LP 实例在干预阶段 (Root Node) 
    的 Strong Branching Scores 作为真正的专家打分替代品，以供监督学习模仿。
    """
    print("=" * 60)
    print(f"🚀 开始通过 Ecole 提取专家强分支 (Strong Branching) 标签 🚀")
    print(f"📂 扫描目录: {data_dir}")
    print("=" * 60)

    # 1. 初始化 Ecole 强化学习收集环境 
    # 关闭多余切平面与重启，以便纯净快速获取根节点信息
    scip_params = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 300
    }

    env = ecole.environment.Branching(
        observation_function=ecole.observation.StrongBranchingScores(),
        scip_params=scip_params
    )

    # 2. 匹配并遍历所有的 .lp 计算实例
    lp_files = sorted(glob.glob(os.path.join(data_dir, "*.lp")))
    if not lp_files:
        print("❌ 未在目标路径下找到任何 .lp 文件！")
        return

    success_count = 0

    for lp_file in lp_files:
        base_name = os.path.basename(lp_file)
        json_file = lp_file.replace(".lp", ".json")
        save_path = lp_file.replace(".lp", "_label.pt")
        
        start_time = time.time()
        
        # 判断配对的 JSON 描述图结构是否存在，保障图数据与标签数据的双向奔赴
        if not os.path.exists(json_file):
            print(f"⚠️ 跳过 {base_name}，找不到对应的 .json 文件。")
            continue

        try:
            # 读取 JSON 元数据，解析真正的全局变量总数
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f).get("metadata", {})
                num_vars = metadata.get("num_variables", None)
            
            if num_vars is None:
                print(f"⚠️ 跳过 {base_name}，JSON 缺失 metadata.num_variables 约束！")
                continue

            # Step in: 利用 Ecole 重置到根节点并拿出来专家的观察打分
            obs, action_set, reward, done, info = env.reset(lp_file)

            # SCIP 如果直接把问题做无缝消除了，将没有节点可供打分支并处于 done=True
            if done:
                print(f"⚠️跳过 {base_name} - 已经被在 Presolve 前处理阶段直接秒解，无根节点分支需求。")
                continue

            # 组装 0-Padding 的满长张量，这为后续直接与图特征堆叠提供严密对接
            labels = torch.zeros(num_vars, dtype=torch.float32)
            
            # Ecole 只会返回允许被打分支的变量（即未定整数），需要从稀疏数组退回至长列中
            # Note: 之前代码中，torch.tensor(obs) 的维度可能是传入的 obs 整块，
            # 若 obs 的大小不是 len(action_set)，强行赋值会报错。
            # 这里为了对接 Ecole (它可能返回了各种格式的 obs，StrongBranching 返回一个 len(action_set) 大小的 array)，
            # 我们直接选取对于 action 集合的分数即可。
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Sometimes Ecole returns a larger obs array than action_set. Let's make sure shapes match.
            if obs_tensor.shape[0] != action_set.shape[0]:
                obs_tensor = obs_tensor[action_set]

            labels[action_set] = obs_tensor

            torch.save(labels, save_path)
            
            cost_time = time.time() - start_time
            print(f"✔️ [成功] {base_name} | 耗时: {cost_time:.2f}s | "
                  f"可选分支行为 action_set: {len(action_set)}/{num_vars}")
            
            success_count += 1

        except Exception as e:
            # 捕获因 SCIP 内部断言错误等引起的奇葩环境抛错
            print(f"❌ 解析 LP 实例出错 {base_name}: {e}")

    print("=" * 60)
    print(f"🎉 全部结束！共成功采集并固化了 {success_count} 个实例的独立强分支伪标签。")
    print("=" * 60)


if __name__ == '__main__':
    # 遍历 raw 下所有的业务子目录并执行采集
    base_dir = "data/raw"
    if os.path.exists(base_dir):
        for subdir in os.listdir(base_dir):
            target_dir = os.path.join(base_dir, subdir)
            if os.path.isdir(target_dir):
                collect_expert_labels(target_dir)