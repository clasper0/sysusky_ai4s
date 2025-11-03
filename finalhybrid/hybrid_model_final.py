"""
最终版混合模型：结合图神经网络和分子描述符进行多任务pIC50预测
===================================================================

该模块实现了结合图神经网络特征和传统分子描述符的混合模型，
用于同时预测分子对多个靶点的pIC50值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class AttentionPooling(nn.Module):
    """
    注意力池化机制
    """

    def __init__(self, hidden_dim: int):
        """
        初始化注意力池化层
        
        Args:
            hidden_dim: 隐藏层维度
        """
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            batch: 批次分配 [num_nodes]
            
        Returns:
            torch.Tensor: 图级表示 [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1

        # 计算注意力权重
        attention_weights = self.attention(x)  # [num_nodes, 1]

        # 创建批次掩码
        batch_mask = F.one_hot(batch, num_classes=batch_size).float()  # [num_nodes, batch_size]

        # 归一化注意力权重
        attention_weights = attention_weights * batch_mask
        attention_sums = attention_weights.sum(dim=0, keepdim=True)
        attention_weights = attention_weights / (attention_sums + 1e-8)

        # 加权求和节点特征
        graph_embedding = torch.matmul(attention_weights.t(), x)  # [batch_size, hidden_dim]

        return graph_embedding


class MolecularGCN(nn.Module):
    """
    分子图卷积网络
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dims: List[int] = [128, 256, 512],
        dropout_rate: float = 0.2,
        attention_heads: int = 4
    ):
        """
        初始化分子GCN
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout率
            attention_heads: 注意力头数
        """
        super(MolecularGCN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.hidden_dims = hidden_dims
        
        # 构建GCN层
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()

        # 输入层
        self.gcn_layers.append(GCNConv(input_dim, hidden_dims[0]))
        self.gat_layers.append(GATConv(hidden_dims[0], hidden_dims[0] // attention_heads,
                                       heads=attention_heads, dropout=dropout_rate))

        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.gcn_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.gat_layers.append(GATConv(hidden_dims[i], hidden_dims[i] // attention_heads,
                                          heads=attention_heads, dropout=dropout_rate))
            
        # 注意力池化
        self.attention_pooling = AttentionPooling(hidden_dims[-1])
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            torch.Tensor: 图表示特征
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 存储初始输入用于残差连接
        x_in = x

        # 应用GCN和GAT层
        for i, (gcn_layer, gat_layer) in enumerate(zip(self.gcn_layers, self.gat_layers)):
            # GCN层
            x = gcn_layer(x, edge_index)

            # 应用激活和dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # GAT层用于注意力机制
            x_att = gat_layer(x, edge_index)
            x = x + x_att  # 与注意力的残差连接

            # 残差连接
            if i < len(self.gcn_layers) - 1:
                if x.size(-1) == x_in.size(-1):
                    x = x + x_in
                x_in = x

        # 全局池化
        graph_embedding = self.attention_pooling(x, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)

        # 拼接池化结果
        combined_pool = torch.cat([graph_embedding, mean_pool, max_pool, add_pool], dim=1)
        
        return combined_pool


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
        
        # 图神经网络分支
        self.gcn_branch = MolecularGCN(
            input_dim=gcn_input_dim,
            hidden_dims=gcn_hidden_dims,
            dropout_rate=gcn_dropout_rate
        )
        
        # 计算融合层输入维度
        gcn_output_dim = gcn_hidden_dims[-1] * 4  # 4种池化方式的拼接结果
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
        # 通过GCN分支提取图特征
        graph_features = self.gcn_branch(graph_data)  # [batch_size, gcn_output_dim]
        
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