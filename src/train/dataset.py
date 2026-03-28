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
    USG 万能本体数据集：加载所有 JSON，构建带假标签的 PyG HeteroData 以供批处理训练。
    """
    def __init__(self, data_dir: str):
        super().__init__()
        # 扫描目录下所有的 .json 文件
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx: int):
        json_path = self.file_paths[idx]
        
        # 1. 使用构建器加载图（此时 ecole_obs 默认为 None，退化为全零数学占位）
        data = load_and_build(json_path, verbose=False)
        
        # 2. 注入伪造的专家标签 (Expert Score) 以供监督学习测试
        num_vars = data['variable'].x.shape[0]
        data['variable'].y = torch.rand(num_vars, dtype=torch.float32)
        
        return data

if __name__ == '__main__':
    # ── 测试数据集和 DataLoader 的图拼接行为 ──
    print("=" * 50)
    print("🚀 开始测试 OntoDataset 批处理机制 🚀")
    print("=" * 50)
    
    # 指向本地的 ES 数据集目录
    data_dir = "data/raw/employee_scheduling"
    dataset = OntoDataset(data_dir)
    print(f"✅ 找到 {len(dataset)} 个 Json 实例文件。")
    
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
