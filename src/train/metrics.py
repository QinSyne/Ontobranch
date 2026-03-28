import torch

def calculate_ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """
    针对单张图计算 NDCG@K
    :param scores: 模型的预测打分 [N]
    :param labels: 专家（SCIP）结点的 Branching 真实打分 [N]
    :param k: 评估前 K 个
    :return: NDCG@K (float)
    """
    valid_mask = labels > 1e-6
    valid_scores = scores[valid_mask]
    valid_labels = labels[valid_mask]
    
    if len(valid_labels) < 2:
        return 0.0
    
    k = min(k, len(valid_labels))
    
    # 获取真实的 Top-K (Ideal)
    ideal_sorted_labels, _ = torch.sort(valid_labels, descending=True)
    ideal_top_k = ideal_sorted_labels[:k]
    
    # 按照模型预测的得分排序，取排名前 K 的索引
    _, pred_indices = torch.sort(valid_scores, descending=True)
    # 拿到这批索引对应的【真实标签分数】
    pred_top_k_labels = valid_labels[pred_indices[:k]]
    
    # 计算 DCG
    # log2(i + 1), index i 顺位从 1 开始 -> i+1 从 2 开始
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=scores.device))
    dcg = (pred_top_k_labels / discounts).sum().item()
    
    # 计算 IDCG (Ideal DCG)
    idcg = (ideal_top_k / discounts).sum().item()
    
    if idcg <= 0.0 or idcg != idcg:  # 分母为0或NaN
        return 0.0
    
    return dcg / idcg

def calculate_acc_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """
    针对单张图计算 Accuracy@K：
    模型预测的 Top-1 变量，是否落在了真实性能的前 K 之中
    """
    valid_mask = labels > 1e-6
    valid_scores = scores[valid_mask]
    valid_labels = labels[valid_mask]
    
    if len(valid_labels) < 2:
        return 0.0
        
    k = min(k, len(valid_labels))
    
    # 模型预测最高的 1 个变量
    top1_pred_idx = torch.argmax(valid_scores).item()
    
    # 真实情况中前 K 高的变量
    _, ideal_indices = torch.topk(valid_labels, k)
    
    if top1_pred_idx in ideal_indices.tolist():
        return 1.0
    return 0.0

def evaluate_batch(scores: torch.Tensor, labels: torch.Tensor, batch_idx: torch.Tensor, k: int = 5) -> tuple[float, float]:
    """
    批处理级别的 NDCG@K 和 Acc@K
    :return: (avg_ndcg_at_k, avg_acc_at_k)
    """
    unique_graphs = torch.unique(batch_idx)
    total_ndcg = 0.0
    total_acc = 0.0
    valid_graphs = 0
    
    for g_idx in unique_graphs:
        mask = (batch_idx == g_idx)
        g_scores = scores[mask]
        g_labels = labels[mask]
        
        # 图中需要有多于1个变量可供排序
        if len(g_labels) < 2:
            continue
            
        total_ndcg += calculate_ndcg_at_k(g_scores, g_labels, k)
        total_acc += calculate_acc_at_k(g_scores, g_labels, k)
        valid_graphs += 1
        
    if valid_graphs == 0:
        return 0.0, 0.0
        
    return total_ndcg / valid_graphs, total_acc / valid_graphs
