"""
验证聚类质量和权重相似性
检查同一 expert group 内的权重是否相似
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


def compute_cosine_similarity_matrix(weights: torch.Tensor) -> torch.Tensor:
    """
    计算权重矩阵中所有行之间的余弦相似度
    
    Args:
        weights: [n_neurons, hidden_size]
    
    Returns:
        similarity_matrix: [n_neurons, n_neurons]
    """
    # 归一化
    weights_norm = F.normalize(weights, p=2, dim=1)
    # 计算余弦相似度
    similarity = torch.mm(weights_norm, weights_norm.T)
    return similarity


def compute_pairwise_cosine_similarity(weights: torch.Tensor) -> torch.Tensor:
    """
    计算权重矩阵中所有行对之间的余弦相似度（返回上三角矩阵）
    
    Args:
        weights: [n_neurons, hidden_size]
    
    Returns:
        similarities: 上三角矩阵的值（不包括对角线）
    """
    similarity_matrix = compute_cosine_similarity_matrix(weights)
    n = similarity_matrix.shape[0]
    # 提取上三角矩阵（不包括对角线）
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    similarities = similarity_matrix[mask]
    return similarities


def verify_clustering_quality(
    expert_groups: List[List[int]],
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    verbose: bool = True
) -> dict:
    """
    验证聚类质量：检查同一 expert group 内的权重是否相似
    
    Args:
        expert_groups: 每个 expert 包含的神经元索引列表
        gate_proj_weight: [intermediate_size, hidden_size]
        up_proj_weight: [intermediate_size, hidden_size]
        verbose: 是否打印详细信息
    
    Returns:
        results: 包含相似度统计的字典
    """
    results = {
        'intra_similarities': {'gate': [], 'up': []},
        'inter_similarities': {'gate': [], 'up': []},
        'expert_stats': []
    }
    
    # 计算组内相似度
    for i, group in enumerate(expert_groups):
        if len(group) < 2:
            if verbose:
                print(f"Expert {i}: 神经元数量 < 2，跳过相似度计算")
            continue
        
        group_gate = gate_proj_weight[group, :]  # [n_neurons, hidden_size]
        group_up = up_proj_weight[group, :]
        
        # 计算组内权重相似度
        gate_sim = compute_pairwise_cosine_similarity(group_gate)
        up_sim = compute_pairwise_cosine_similarity(group_up)
        
        gate_mean = gate_sim.mean().item()
        up_mean = up_sim.mean().item()
        
        results['intra_similarities']['gate'].append(gate_mean)
        results['intra_similarities']['up'].append(up_mean)
        
        # 计算组间相似度（与所有其他 groups 比较）
        inter_gate_sims = []
        inter_up_sims = []
        
        for j, other_group in enumerate(expert_groups):
            if i == j or len(other_group) == 0:
                continue
            
            # 计算两个 group 的 centroid
            group_gate_centroid = group_gate.mean(dim=0)
            other_gate_centroid = gate_proj_weight[other_group, :].mean(dim=0)
            
            group_up_centroid = group_up.mean(dim=0)
            other_up_centroid = up_proj_weight[other_group, :].mean(dim=0)
            
            # 计算 centroid 之间的相似度
            inter_gate_sim = F.cosine_similarity(
                group_gate_centroid.unsqueeze(0),
                other_gate_centroid.unsqueeze(0)
            ).item()
            
            inter_up_sim = F.cosine_similarity(
                group_up_centroid.unsqueeze(0),
                other_up_centroid.unsqueeze(0)
            ).item()
            
            inter_gate_sims.append(inter_gate_sim)
            inter_up_sims.append(inter_up_sim)
        
        inter_gate_mean = np.mean(inter_gate_sims) if inter_gate_sims else 0.0
        inter_up_mean = np.mean(inter_up_sims) if inter_up_sims else 0.0
        
        results['inter_similarities']['gate'].append(inter_gate_mean)
        results['inter_similarities']['up'].append(inter_up_mean)
        
        # 计算质量分数：组内相似度 - 组间相似度
        quality_gate = gate_mean - inter_gate_mean
        quality_up = up_mean - inter_up_mean
        quality_avg = (quality_gate + quality_up) / 2
        
        results['expert_stats'].append({
            'expert': i,
            'n_neurons': len(group),
            'intra_gate_sim': gate_mean,
            'intra_up_sim': up_mean,
            'inter_gate_sim': inter_gate_mean,
            'inter_up_sim': inter_up_mean,
            'quality_gate': quality_gate,
            'quality_up': quality_up,
            'quality_avg': quality_avg
        })
        
        if verbose:
            print(f"\nExpert {i} (神经元数: {len(group)}):")
            print(f"  组内相似度 - gate: {gate_mean:.4f}, up: {up_mean:.4f}")
            print(f"  组间相似度 - gate: {inter_gate_mean:.4f}, up: {inter_up_mean:.4f}")
            print(f"  质量分数 - gate: {quality_gate:.4f}, up: {quality_up:.4f}, 平均: {quality_avg:.4f}")
            
            if quality_avg < 0:
                print(f"  ⚠️  警告: 组间相似度 > 组内相似度，聚类质量可能不佳")
            elif quality_avg < 0.1:
                print(f"  ⚠️  警告: 质量分数较低，聚类可能不够好")
            else:
                print(f"  ✓ 聚类质量良好")
    
    # 整体统计
    if results['intra_similarities']['gate']:
        avg_intra_gate = np.mean(results['intra_similarities']['gate'])
        avg_intra_up = np.mean(results['intra_similarities']['up'])
        avg_inter_gate = np.mean(results['inter_similarities']['gate'])
        avg_inter_up = np.mean(results['inter_similarities']['up'])
        
        overall_quality = (avg_intra_gate + avg_intra_up) / 2 - (avg_inter_gate + avg_inter_up) / 2
        
        if verbose:
            print("\n" + "=" * 80)
            print("整体统计:")
            print(f"  平均组内相似度 - gate: {avg_intra_gate:.4f}, up: {avg_intra_up:.4f}")
            print(f"  平均组间相似度 - gate: {avg_inter_gate:.4f}, up: {avg_inter_up:.4f}")
            print(f"  整体质量分数: {overall_quality:.4f}")
            
            if overall_quality < 0:
                print("  ❌ 聚类质量差: 组间相似度 > 组内相似度")
            elif overall_quality < 0.1:
                print("  ⚠️  聚类质量一般: 组内和组间相似度差异较小")
            else:
                print("  ✓ 聚类质量良好: 组内相似度明显大于组间相似度")
        
        results['overall'] = {
            'avg_intra_gate': avg_intra_gate,
            'avg_intra_up': avg_intra_up,
            'avg_inter_gate': avg_inter_gate,
            'avg_inter_up': avg_inter_up,
            'overall_quality': overall_quality
        }
    
    return results


def verify_representative_neurons(
    expert_groups: List[List[int]],
    representative_indices: List[int],
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    verbose: bool = True
) -> dict:
    """
    验证代表神经元是否能代表整个 expert group
    
    Args:
        expert_groups: 每个 expert 包含的神经元索引列表
        representative_indices: 每个 expert 的代表神经元索引
        gate_proj_weight: [intermediate_size, hidden_size]
        up_proj_weight: [intermediate_size, hidden_size]
        verbose: 是否打印详细信息
    
    Returns:
        results: 包含代表神经元相似度的字典
    """
    results = {
        'representative_similarities': {'gate': [], 'up': []},
        'expert_stats': []
    }
    
    for i, (group, rep_idx) in enumerate(zip(expert_groups, representative_indices)):
        if len(group) < 2:
            continue
        
        # 代表神经元的权重
        rep_gate = gate_proj_weight[rep_idx, :]  # [hidden_size]
        rep_up = up_proj_weight[rep_idx, :]
        
        # Group 内所有神经元的权重
        group_gate = gate_proj_weight[group, :]  # [n_neurons, hidden_size]
        group_up = up_proj_weight[group, :]
        
        # 计算代表神经元与 group 内其他神经元的相似度
        rep_gate_expanded = rep_gate.unsqueeze(0)  # [1, hidden_size]
        rep_up_expanded = rep_up.unsqueeze(0)
        
        gate_sims = F.cosine_similarity(rep_gate_expanded, group_gate, dim=1)
        up_sims = F.cosine_similarity(rep_up_expanded, group_up, dim=1)
        
        gate_mean = gate_sims.mean().item()
        up_mean = up_sims.mean().item()
        
        results['representative_similarities']['gate'].append(gate_mean)
        results['representative_similarities']['up'].append(up_mean)
        
        results['expert_stats'].append({
            'expert': i,
            'representative_idx': rep_idx,
            'gate_similarity': gate_mean,
            'up_similarity': up_mean,
            'avg_similarity': (gate_mean + up_mean) / 2
        })
        
        if verbose:
            print(f"\nExpert {i} (代表神经元: {rep_idx}):")
            print(f"  与 group 内神经元的相似度 - gate: {gate_mean:.4f}, up: {up_mean:.4f}")
            
            if gate_mean < 0.3 or up_mean < 0.3:
                print(f"  ⚠️  警告: 代表神经元与 group 内其他神经元相似度较低")
            else:
                print(f"  ✓ 代表神经元能较好地代表整个 group")
    
    # 整体统计
    if results['representative_similarities']['gate']:
        avg_gate = np.mean(results['representative_similarities']['gate'])
        avg_up = np.mean(results['representative_similarities']['up'])
        
        if verbose:
            print("\n" + "=" * 80)
            print("代表神经元整体统计:")
            print(f"  平均相似度 - gate: {avg_gate:.4f}, up: {avg_up:.4f}")
            print(f"  平均相似度: {(avg_gate + avg_up) / 2:.4f}")
        
        results['overall'] = {
            'avg_gate_similarity': avg_gate,
            'avg_up_similarity': avg_up,
            'avg_similarity': (avg_gate + avg_up) / 2
        }
    
    return results


if __name__ == "__main__":
    # 示例用法
    print("聚类质量验证工具")
    print("=" * 80)
    print("\n使用方法:")
    print("1. 在 construct_experts_k_means 后调用 verify_clustering_quality")
    print("2. 检查组内相似度是否 > 组间相似度")
    print("3. 如果质量不佳，考虑使用 centroid 权重或改进聚类方法")
    print("\n示例代码:")
    print("""
    from verify_clustering_quality import verify_clustering_quality, verify_representative_neurons
    
    # 在 construct_moe 函数中，聚类后添加验证
    expert_groups, experts, representative_indices = construct_experts_k_means(...)
    
    # 验证聚类质量
    clustering_results = verify_clustering_quality(
        expert_groups,
        layer.mlp.gate_proj.weight.data,
        layer.mlp.up_proj.weight.data,
        verbose=True
    )
    
    # 验证代表神经元
    rep_results = verify_representative_neurons(
        expert_groups,
        representative_indices,
        layer.mlp.gate_proj.weight.data,
        layer.mlp.up_proj.weight.data,
        verbose=True
    )
    """)

