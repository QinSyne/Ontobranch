import torch

def bipartite_infonce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: torch.Tensor,
    pos_ratio: float = 0.05,
    tau: float = 0.1
) -> torch.Tensor:
    """
    针对二分图排序任务的 InfoNCE 对比学习损失 (Supervised Contrastive Loss)。
    在多图 Batching 场景下，按单张图独立计算排序约束，并在图级别求均值返回。

    参数:
        scores: [N_var] 模型预测的当前节点打分分数
        labels: [N_var] 专家给出的节点标签/得分（值越大表示越优越、越应被优先分支）
        batch_idx: [N_var] PyG 生成的图分配向量，指明了每个变量属于这个 Batch 中的哪张图
        pos_ratio: 比例，将排名前 %% 的视为正样本 (目标分支优先)
        tau: 温度系数 (Temperature)，用来控制对差距的敏感度。越小分布越陡峭。

    返回:
        标量张量 (Scalar Tensor) 代表整个 Batch 的 InfoNCE Loss。
    """
    # 确保图的有效性
    if batch_idx.numel() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    num_graphs = int(batch_idx.max().item()) + 1
    loss_list = []

    for i in range(num_graphs):
        # 提取第 i 张图的节点集
        mask = (batch_idx == i)
        s = scores[mask]
        y = labels[mask]

        num_vars = s.size(0)
        # 如果当前图内的节点数太少，根本无法提供充足的 对比池（正样本 + 负样本），强制抛弃
        if num_vars < 2:
            continue

        # 按目标标签值大小逆序排列，值越大的在越前面
        sorted_indices = torch.argsort(y, descending=True)

        # 取 Top XX% 为正样本池（最低需选取一个作为正样本）
        num_pos = max(1, int(num_vars * pos_ratio))
        # 取 Bottom 50% 的为负样本池（确保负样本量充足）
        num_neg = max(1, int(num_vars * 0.5))

        pos_indices = sorted_indices[:num_pos]
        neg_indices = sorted_indices[-num_neg:]

        # 获取对应的网络预测分数
        s_pos = s[pos_indices]
        s_neg = s[neg_indices]

        loss_p_list = []
        # 对每一个正例样本都拉取与全体负例对齐：
        for s_p in s_pos:
            # 缩放至 InfoNCE 中，以温度 tau 平滑分布（控制概率“锐度”）
            logits = torch.cat([s_p.view(1), s_neg]) / tau

            # -------------------------------------------------------------
            # LogSumExp 技巧防止溢出物理意义注释：
            # InfoNCE 的本质是最大化： exp(s_p) / (exp(s_p) + Σexp(s_neg))
            # 取 -log 后得到目标损失函数 loss = - s_p + log(exp(s_p) + Σexp(s_neg))
            # 直接算 exp() 极易导致浮点溢出变为 NaN。而 torch.logsumexp 先减去最大值
            # 来实现理论上对等、但数值边界绝对平滑稳定的计算效果。
            # -------------------------------------------------------------
            loss_p = torch.logsumexp(logits, dim=0) - logits[0]
            loss_p_list.append(loss_p)

        if loss_p_list:
            # 当前图由于是所有正样本平分的聚合损失，在此加和平均
            loss_list.append(torch.stack(loss_p_list).mean())

    if not loss_list:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # 汇总 Batch 内所有有效子图的对比损失，平均后进行逆传播更新
    return torch.stack(loss_list).mean()


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 开始测试 Bipartite InfoNCE Loss 🚀")
    print("=" * 50)

    # 1. 制造测试场景
    # 模拟 [10] 个节点打分，分成 2 张图（各含 5 节点）
    scores = torch.randn(10, requires_grad=True)
    labels = torch.rand(10)
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    print("📊 假造的模型打分 scores:", scores.data.tolist())
    print("🎯 假造的专家标签 labels:", labels.data.tolist())
    print("📦 所属图向量 batch_idx :", batch_idx.tolist())

    # 2. 计算损失
    loss = bipartite_infonce_loss(scores, labels, batch_idx, pos_ratio=0.2, tau=0.1)
    
    print("\n✅ 成功计算对比损失 (InfoNCE Loss):", loss.item())

    # 3. 考察梯度
    loss.backward()
    print("💥 是否成功回传梯度 (scores.grad):", scores.grad.tolist())
    print("=" * 50)