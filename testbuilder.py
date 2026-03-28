from src.graph.builder import load_and_build

def test_graph_builder():
    print("=" * 50)
    print("🚀 开始测试 UniversalGraphBuilder 🚀")
    print("=" * 50)
    
    # 指向你刚刚上传的这个 json 实例
    json_path = "data/raw/employee_scheduling/employee_scheduling_001.json"
    
    try:
        # 开启 verbose=True 会打印各层的节点和边数量
        data = load_and_build(json_path, verbose=True)
        
        print("\n✅ 图构建成功！以下是 PyG 接收到的最终拓扑结构：")
        print("-" * 50)
        
        print("🟢 节点特征矩阵 (Node Features):")
        for nt in data.node_types:
            print(f"  - {nt:10s} : shape {data[nt].x.shape}")
            
        print("\n🔗 边连接索引 (Edge Indices):")
        for et in data.edge_types:
            # et 是一个 tuple, 比如 ('variable', 'mapped_to', 'entity')
            print(f"  - {et[0]} -> {et[1]} -> {et[2]} : shape {data[et].edge_index.shape}")
            
        print("\n🏷️  边属性 (Edge Attributes):")
        for et in data.edge_types:
            if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None:
                print(f"  - {et[0]} -> {et[1]} -> {et[2]} : shape {data[et].edge_attr.shape}")
                unique_vals = data[et].edge_attr.unique().tolist()
                print(f"    └─ 包含的 Unique Hash 索引: {unique_vals}")
            
        print("=" * 50)
        print("🎉 测试完成！图结构是否完全符合 3节点+3边 的极简设计？")
        
    except Exception as e:
        print(f"\n❌ 图构建失败，捕获到异常：\n{e}")

from src.graph.builder import load_and_build
from src.model.ontognn import OntoGNN

def test_graph_builder_and_model():
    print("=" * 50)
    print("🚀 测试流水线: JSON -> GraphBuilder -> OntoGNN 🚀")
    print("=" * 50)
    
    json_path = "data/raw/employee_scheduling/employee_scheduling_001.json"
    
    # 1. 构建图数据
    print("\n[1] 正在构建 HeteroData...")
    data = load_and_build(json_path, verbose=False)
    print("✅ 图构建完成！")
    
    # 2. 初始化模型
    print("\n[2] 正在初始化 OntoGNN (三明治架构)...")
    # 启用 MPS 后端（如果可用）来测试 M5 Pro 的硬件加速
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = OntoGNN(hidden_dim=64).to(device)
    data = data.to(device)
    print(f"✅ 模型已加载至设备: {device}")
    
    # 3. 前向传播
    print("\n[3] 正在执行前向传播 (Forward Pass)...")
    try:
        scores = model(data)
        print("✅ 前向传播成功！")
        print("-" * 50)
        print(f"🎯 最终输出 Scores 形状: {scores.shape}")
        print(f"🎯 预期形状应为: torch.Size([{data['variable'].x.shape[0]}])")
        print(f"📊 前 5 个变量的打分预览: {scores[:5].tolist()}")
    except Exception as e:
        print(f"\n❌ 前向传播失败，捕获到异常：\n{e}")


if __name__ == "__main__":
    test_graph_builder_and_model()