import os
import sys

# 【兼容性修复】启用 MPS(Mac GPU) 不支持的 scatter 算子的 CPU 回退
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

# 注册项目根目录以应对独立运行时的路径解析
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.train.dataset import OntoDataset
from src.model.ontognn import OntoGNN
from src.train.loss import bipartite_infonce_loss
from src.train.metrics import evaluate_batch

def main():
    print("=" * 60)
    print("🚀 OntoGNN MVP Training Pipeline Starting... 🚀")
    print("=" * 60)

    # 1. 设置训练设备 (兼容 Apple Metal / CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡ Use Device: MPS (Mac GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("⚡ Use Device: CUDA")
    else:
        device = torch.device("cpu")
        print("⚡ Use Device: CPU")

    # 2. 初始化核心组件
    data_dir = "data/raw" # 直指混合根目录
    if not os.path.exists(data_dir):
        print(f"❌ Error: 数据集目录 '{data_dir}' 不存在，请先确认路径！")
        return

    dataset = OntoDataset(data_dir)
    # 把所有的 5 个图凑在一个 Batch 里面进行史诗级泛化融合测试
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    print(f"📦 Dataset loaded: {len(dataset)} graphs found.")

    model = OntoGNN(hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # 3. 开启训练循环
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        total_ndcg = 0.0
        total_acc = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            # 严格对齐数据设备
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # 拿到预测分数
            scores = model(batch)
            
            # 抽取真实打分分布与 Batching 索引映射
            labels = batch['variable'].y
            batch_idx = batch['variable'].batch
            
            # 计算排名/对比损失
            loss = bipartite_infonce_loss(
                scores=scores,
                labels=labels,
                batch_idx=batch_idx,
                pos_ratio=0.1,    # 取排名前 10% 做为正样本
                tau=0.1
            )
            
            batch_ndcg, batch_acc = evaluate_batch(scores.detach(), labels, batch_idx, k=5)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_ndcg += batch_ndcg
            total_acc += batch_acc
            num_batches += 1
            
        # 计算该 Epoch 所有 batch 的平摊代价
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_ndcg = total_ndcg / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        print(f"📅 Epoch [{epoch + 1:02d}/{num_epochs}] | Loss: {avg_loss:.4f} | NDCG@5: {avg_ndcg:.4f} | Acc@5: {avg_acc:.4f}")

    print("=" * 60)
    print("🎉 MVP Training pipeline completed successfully! ")

if __name__ == '__main__':
    main()