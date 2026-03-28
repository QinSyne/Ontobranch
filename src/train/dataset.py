import os
import sys
import glob
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

# Add project root to sys.path to resolve absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.graph.builder import load_and_build

class OntoDataset(Dataset):
    """
    USG 万能本体数据集：读取全部子目录，选取带有真实专家强分支标签的实例，
    构建包含 3种节点+3种边 的跨领域大图。
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.valid_files = []
        
        # 递归扫描所有子目录下的 .json 文件
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    # 寻找同名但后缀为 _label.pt 的真实标签文件
                    label_path = json_path.replace(".json", "_label.pt")
                    
                    # 只有那些成功逃过被“秒解”并收集到真实强分支分数的实例入选：
                    if os.path.exists(label_path):
                        self.valid_files.append((json_path, label_path))
        
    def __len__(self):
        return len(self.valid_files)
        
    def __getitem__(self, idx: int):
        json_path, label_path = self.valid_files[idx]
        
        # 1. 加载图拓扑和基础特征（128维 entity 等）
        data = load_and_build(json_path, verbose=False)
        
        # 2. 注入强分支真实标签 (Real Expert Score)
        # 取代之前 torch.rand 产生的伪造数
        # 由于我们当时存的就是和变量数对齐的一维张量，直接挂载即可
        data['variable'].y = torch.load(label_path, weights_only=True)
        return data

if __name__ == '__main__':
    # ── 测试数据集和 DataLoader 的图拼接行为 ──
    print("=" * 50)
    print("🚀 开始测试 OntoDataset 批处理机制 🚀")
    print("=" * 50)
    
    # 指向大目录，由程序自己递归寻找可用打分
    data_dir = "data/raw"
    dataset = OntoDataset(data_dir)
    print(f"✅ 找到 {len(dataset)} 个带有真实专家标签的 Json/LP 子图。")
    
    # 使用 PyG 专用的 DataLoader 进行图 Batch 拼接
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    
    # 抽取第一个批次
    batch = next(iter(loader))
    
    print("\n📦 抽取的超级图 Batch (batch_size=2) :")
    print("-" * 50)
    print(batch)
    
    print("\n🏷️  验证变量和标签在拼接后的汇总情况:")
    print(f"  - batch['variable'].x shape : {batch['variable'].x.shape}")
    print(f"  - batch['variable'].y shape : {batch['variable'].y.shape}")
    print("=" * 50)
    print("🎉 批处理测试完成，超级大的连通分量即为 batch 切片。")
