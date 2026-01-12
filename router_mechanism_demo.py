"""
Router机制可视化演示
演示为什么使用 representative_indices 的权重能帮助选出合适的 Expert
"""

import torch
import torch.nn.functional as F
import numpy as np

def demonstrate_router_mechanism():
    """
    演示 Router 机制的核心原理
    """
    print("=" * 80)
    print("Router 机制原理解析演示")
    print("=" * 80)
    
    # 1. 模拟场景设置
    hidden_size = 8
    intermediate_size = 12
    n_experts = 3
    
    print("\n【步骤1】模拟原始 MLP 的权重")
    print("-" * 80)
    
    # 模拟原始 MLP 的权重
    gate_proj_weight = torch.randn(intermediate_size, hidden_size)
    up_proj_weight = torch.randn(intermediate_size, hidden_size)
    
    print(f"gate_proj.weight 形状: {gate_proj_weight.shape}")
    print(f"up_proj.weight 形状: {up_proj_weight.shape}")
    
    # 2. 模拟输入（假设这是"数学问题"的特征）
    print("\n【步骤2】模拟输入特征")
    print("-" * 80)
    h = torch.tensor([[0.5, 0.8, 0.3, 0.1, 0.9, 0.2, 0.6, 0.4]], dtype=torch.float32)
    print(f"输入 h 形状: {h.shape}")
    print(f"输入 h: {h[0].numpy()}")
    
    # 3. 原始 MLP 的前向传播
    print("\n【步骤3】原始 MLP 的前向传播")
    print("-" * 80)
    gate = F.silu(h @ gate_proj_weight.T)
    up = h @ up_proj_weight.T
    intermediate = gate * up
    
    print(f"gate 形状: {gate.shape}")
    print(f"up 形状: {up.shape}")
    print(f"intermediate 形状: {intermediate.shape}")
    print(f"\nintermediate 值: {intermediate[0].numpy()}")
    
    # 找出激活最强的神经元（top-3）
    top_k = 3
    _, top_indices = torch.topk(intermediate[0].abs(), k=top_k)
    print(f"\n激活最强的 {top_k} 个神经元索引: {top_indices.tolist()}")
    print(f"它们的激活值: {intermediate[0][top_indices].numpy()}")
    
    # 4. 模拟 Expert 分组（简化版）
    print("\n【步骤4】模拟 Expert 分组")
    print("-" * 80)
    
    # 假设通过 k-means 聚类，得到以下分组：
    # Expert 0: 神经元 [0, 1, 2, 3]
    # Expert 1: 神经元 [4, 5, 6, 7]
    # Expert 2: 神经元 [8, 9, 10, 11]
    expert_groups = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ]
    
    # 假设代表神经元是每个 group 中激活率最高的
    # 这里简化：选择 group 中激活值最大的神经元作为代表
    representative_indices = []
    for i, group in enumerate(expert_groups):
        group_intermediate = intermediate[0][group]
        max_idx_in_group = torch.argmax(group_intermediate.abs())
        representative_idx = group[max_idx_in_group]
        representative_indices.append(representative_idx)
        print(f"Expert {i}: 神经元组 {group}, 代表神经元: {representative_idx}")
    
    print(f"\n所有代表神经元索引: {representative_indices}")
    
    # 5. Router 初始化
    print("\n【步骤5】Router 初始化（使用代表神经元的权重）")
    print("-" * 80)
    
    core_weights = up_proj_weight[representative_indices, :]  # [n_experts, hidden_size]
    core_gate_weights = gate_proj_weight[representative_indices, :]  # [n_experts, hidden_size]
    
    print(f"core_weights 形状: {core_weights.shape}")
    print(f"core_gate_weights 形状: {core_gate_weights.shape}")
    
    # 6. Router 的前向传播
    print("\n【步骤6】Router 的前向传播（计算相似度）")
    print("-" * 80)
    
    # 计算点积相似度
    classifier_scores = h @ core_weights.T  # [1, n_experts]
    gate_scores = F.silu(h @ core_gate_weights.T)  # [1, n_experts]
    scores = (classifier_scores * gate_scores).abs()
    
    print(f"classifier_scores (h @ core_weights^T): {classifier_scores[0].numpy()}")
    print(f"gate_scores (act_fn(h @ core_gate_weights^T)): {gate_scores[0].numpy()}")
    print(f"最终 scores: {scores[0].numpy()}")
    
    # Softmax 归一化
    scores_softmax = F.softmax(scores, dim=-1)
    print(f"Softmax 后的 scores: {scores_softmax[0].numpy()}")
    
    # 选择 top-1 expert
    selected_expert = torch.argmax(scores_softmax, dim=-1)[0].item()
    print(f"\n选择的 Expert: {selected_expert}")
    
    # 7. 验证逻辑
    print("\n【步骤7】验证逻辑")
    print("-" * 80)
    
    # 检查：如果输入激活了某个 expert 中的神经元，router 是否选择了该 expert？
    activated_neurons = top_indices.tolist()
    print(f"激活的神经元: {activated_neurons}")
    
    for i, group in enumerate(expert_groups):
        overlap = set(activated_neurons) & set(group)
        if overlap:
            print(f"Expert {i} 的神经元 {list(overlap)} 被激活")
            if i == selected_expert:
                print(f"  ✓ Router 正确选择了 Expert {i}")
            else:
                print(f"  ✗ Router 选择了 Expert {selected_expert}，但应该选择 Expert {i}")
    
    # 8. 解释 hW^T 的原理
    print("\n【步骤8】hW^T 的几何解释")
    print("-" * 80)
    
    expert_idx = selected_expert
    w = core_weights[expert_idx]  # 选择的 expert 的权重
    
    # 计算点积
    dot_product = (h[0] @ w).item()
    
    # 计算范数和夹角
    h_norm = torch.norm(h[0], p=2).item()
    w_norm = torch.norm(w, p=2).item()
    cos_theta = dot_product / (h_norm * w_norm + 1e-8)
    theta_deg = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
    
    print(f"输入 h 的 L2 范数: {h_norm:.4f}")
    print(f"Expert {expert_idx} 权重 W 的 L2 范数: {w_norm:.4f}")
    print(f"点积 h·W: {dot_product:.4f}")
    print(f"余弦相似度 cos(θ): {cos_theta:.4f}")
    print(f"夹角 θ: {theta_deg:.2f}°")
    print(f"\n几何意义: h·W = ||h|| × ||W|| × cos(θ)")
    print(f"          = {h_norm:.4f} × {w_norm:.4f} × {cos_theta:.4f} = {dot_product:.4f}")
    print(f"\n如果 h 和 W 方向相似（θ 小），点积大 → 高分 → 选择该 Expert")
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_router_mechanism()

