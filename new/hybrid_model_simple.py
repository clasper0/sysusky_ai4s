"""
简化版混合模型：结合图神经网络和分子描述符进行多任务pIC50预测
===================================================================

该模块实现了结合图神经网络特征和传统分子描述符的简化版混合模型，
用于同时预测分子对多个靶点的pIC50值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_model import MolecularGCN
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class HybridMolecularPredictor(nn.Module):
    """
    混合分子预测器：结合图神经网络和分子描述符特征
    """

    def __init__(
        self,
        gcn_input_dim: int = 36,
        gcn_hidden_dims: List[int] = [128, 256, 512],
        descriptor_dim: int = 0,  # 分子描述符特征维度
        num_targets: int = 5,     # 靶点数量
        gcn_dropout_rate: float = 0.2,
        fusion_hidden_dims: List[int] = [512, 256],  # 融合层隐藏维度
        fusion_dropout_rate: float = 0.1
    ):
        """
        初始化混合分子预测器
        
        Args:
            gcn_input_dim: GCN输入特征维度
            gcn_hidden_dims: GCN隐藏层维度列表
            descriptor_dim: 分子描述符特征维度
            num_targets: 靶点数量（输出维度）
            gcn_dropout_rate: GCN Dropout率
            fusion_hidden_dims: 融合层隐藏维度列表
            fusion_dropout_rate: 融合层Dropout率
        """
        super(HybridMolecularPredictor, self).__init__()
        
        self.descriptor_dim = descriptor_dim
        self.num_targets = num_targets
        
        # 图神经网络分支 (使用完整的MolecularGCN)
        self.gcn_branch = MolecularGCN(
            input_dim=gcn_input_dim,
            hidden_dims=gcn_hidden_dims,
            output_dim=1,  # 设置为1，我们不使用它的预测头
            dropout_rate=gcn_dropout_rate,
            use_batch_norm=True,
            use_residual=True,
            attention_heads=4
        )
        
        # 替换MolecularGCN的预测头，我们自己实现
        self.gcn_branch.prediction_head = nn.Identity()
        
        # 计算融合层输入维度
        # 注意：我们需要直接使用池化后的特征
        gcn_output_dim = gcn_hidden_dims[-1] // 2  # MolecularGCN的输出维度
        fusion_input_dim = gcn_output_dim + descriptor_dim
        
        # 构建融合层
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        fusion_layers.append(nn.Linear(prev_dim, num_targets))
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_data, descriptor_features=None):
        """
        前向传播
        
        Args:
            graph_data: 图数据 (PyTorch Geometric Data对象)
            descriptor_features: 分子描述符特征 (torch.Tensor, 可选)
            
        Returns:
            torch.Tensor: 预测的多靶点pIC50值 [batch_size, num_targets]
        """
        # 临时修复：确保GCN层正确处理边权重
        original_edge_attr = graph_data.edge_attr
        graph_data.edge_attr = None
        
        # 通过GCN分支提取图特征（使用我们修改后的版本）
        graph_features = self.gcn_branch(graph_data)  # [batch_size, gcn_output_dim]
        
        # 恢复原始边属性
        graph_data.edge_attr = original_edge_attr
        
        # 如果提供了分子描述符特征，则与图特征拼接
        if descriptor_features is not None and self.descriptor_dim > 0:
            # 确保批次大小匹配
            if descriptor_features.size(0) != graph_features.size(0):
                # 通过ptr计算实际批次大小
                actual_batch_size = graph_features.size(0)
                descriptor_features = descriptor_features[:actual_batch_size]
            # 拼接图特征和描述符特征
            combined_features = torch.cat([graph_features, descriptor_features], dim=1)
        else:
            # 仅使用图特征
            combined_features = graph_features
            
        # 通过融合网络预测多靶点pIC50
        predictions = self.fusion_network(combined_features)
        
        return predictions


def load_descriptor_data(data_path: str, descriptor_cols: List[str]) -> np.ndarray:
    """
    加载分子描述符数据
    
    Args:
        data_path: 数据文件路径
        descriptor_cols: 描述符列名列表
        
    Returns:
        np.ndarray: 描述符特征矩阵
    """
    df = pd.read_csv(data_path)
    
    # 提取描述符特征
    descriptor_features = df[descriptor_cols].values.astype(np.float32)
    
    return descriptor_features


def create_hybrid_model(
    gcn_input_dim: int = 36,
    gcn_hidden_dims: List[int] = [128, 256, 512],
    descriptor_dim: int = 0,
    num_targets: int = 5,
    **kwargs
) -> HybridMolecularPredictor:
    """
    创建混合模型的工厂函数
    
    Args:
        gcn_input_dim: GCN输入特征维度
        gcn_hidden_dims: GCN隐藏层维度列表
        descriptor_dim: 分子描述符特征维度
        num_targets: 靶点数量
        **kwargs: 其他参数
        
    Returns:
        HybridMolecularPredictor: 初始化的混合模型
    """
    return HybridMolecularPredictor(
        gcn_input_dim=gcn_input_dim,
        gcn_hidden_dims=gcn_hidden_dims,
        descriptor_dim=descriptor_dim,
        num_targets=num_targets,
        **kwargs
    )


# 示例用法
if __name__ == "__main__":
    # 创建混合模型示例
    model = HybridMolecularPredictor(
        gcn_input_dim=36,
        gcn_hidden_dims=[128, 256, 512],
        descriptor_dim=10,  # 假设有10个描述符特征
        num_targets=5
    )
    
    print("混合模型架构:")
    print(model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")