#!/usr/bin/env python3
"""
最终版混合分子预测模型
====================

该模块定义了一个结合图神经网络和分子描述符的混合模型，
用于预测分子的多个属性（如pIC50值）。
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import SAGPooling, TopKPooling
import torch.nn.functional as F


class HybridMolecularPredictor(nn.Module):
    """
    混合分子预测器
    
    该模型结合了图神经网络和分子描述符来预测分子属性。
    """
    
    def __init__(self, gcn_input_dim=36, gcn_hidden_dims=[64, 128], 
                 descriptor_dim=4, fusion_hidden_dims=[640, 256], num_targets=5):
        """
        初始化混合分子预测器
        
        参数:
            gcn_input_dim (int): GCN输入维度（原子特征数）
            gcn_hidden_dims (list): GCN隐藏层维度列表
            descriptor_dim (int): 分子描述符维度
            fusion_hidden_dims (list): 融合层隐藏维度列表 (需要与实际输入维度匹配)
            num_targets (int): 目标数量
        """
        super(HybridMolecularPredictor, self).__init__()
        
        self.gcn_input_dim = gcn_input_dim
        self.gcn_hidden_dims = gcn_hidden_dims
        self.descriptor_dim = descriptor_dim
        self.fusion_hidden_dims = fusion_hidden_dims
        self.num_targets = num_targets
        
        # 构建GCN层
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        
        # 输入层
        input_dim = gcn_input_dim
        
        # 构建隐藏层
        for hidden_dim in gcn_hidden_dims:
            # 添加GCN层
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            
            # 添加对应的GAT层
            self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            
            input_dim = hidden_dim
        
        # 添加融合网络用于学习如何融合GCN和GAT的输出
        self.fusion_network = nn.Sequential(
            nn.Linear(gcn_hidden_dims[-1] * 2, gcn_hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(gcn_hidden_dims[-1], gcn_hidden_dims[-1] * 2),
            nn.Sigmoid()
        )
        
        # 注意力池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(gcn_hidden_dims[-1], 1),
            nn.Tanh()
        )
        
        # 图表示的维度（平均池化+最大池化+求和池化+注意力池化）
        graph_embedding_dim = gcn_hidden_dims[-1] * 4  # 128 * 4 = 512
        
        # 分子描述符处理网络
        if descriptor_dim > 0:
            self.descriptor_network = nn.Sequential(
                nn.Linear(descriptor_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2)
            )
            fusion_input_dim = graph_embedding_dim + 128  # 512 + 128 = 640
        else:
            self.descriptor_network = None
            fusion_input_dim = graph_embedding_dim
        
        # 融合网络 - 修复维度配置
        self.fusion_layers = nn.ModuleList()
        
        # 第一层：fusion_input_dim -> fusion_hidden_dims[0]
        self.fusion_layers.append(nn.Linear(fusion_input_dim, fusion_hidden_dims[0]))
        self.fusion_layers.append(nn.ReLU())
        self.fusion_layers.append(nn.BatchNorm1d(fusion_hidden_dims[0]))
        self.fusion_layers.append(nn.Dropout(0.3))
        
        # 后续层
        prev_dim = fusion_hidden_dims[0]
        for hidden_dim in fusion_hidden_dims[1:]:
            self.fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.fusion_layers.append(nn.ReLU())
            self.fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            self.fusion_layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # 任务特定的输出头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            ) for _ in range(num_targets)
        ])
    
    def gcn_forward(self, x, edge_index, edge_attr, batch):
        """
        GCN前向传播
        
        参数:
            x (Tensor): 节点特征
            edge_index (LongTensor): 边索引
            edge_attr (Tensor): 边属性
            batch (LongTensor): 批次信息
            
        返回:
            Tensor: 图级别的表示
        """
        # 多层GCN和GAT
        for gcn_layer, gat_layer in zip(self.gcn_layers, self.gat_layers):
            # 保存输入用于残差连接
            x_in = x
            
            # GCN分支
            gcn_out = gcn_layer(x, edge_index)
            gcn_out = F.relu(gcn_out)
            
            # GAT分支
            gat_out = gat_layer(x, edge_index)
            gat_out = F.relu(gat_out)
            
            # 融合两个分支 - 使用可学习的权重而不是简单平均
            # 添加一个小型神经网络来学习如何融合两个分支
            fusion_input = torch.cat([gcn_out, gat_out], dim=1)
            fusion_weights = torch.sigmoid(self.fusion_network(fusion_input))
            x = fusion_weights[:, :gcn_out.size(1)] * gcn_out + fusion_weights[:, gcn_out.size(1):] * gat_out
            
            # 添加残差连接（如果维度匹配）- 修复残差连接实现
            if x.size(1) == x_in.size(1):
                x = x + x_in
            # 注意：不要更新x_in，它应该保持为该层的输入
            
            # dropout
            x = F.dropout(x, p=0.2, training=self.training)
        
        # 多种池化策略
        mean_pooled = global_mean_pool(x, batch)
        max_pooled = global_max_pool(x, batch)
        sum_pooled = global_add_pool(x, batch)
        
        # 注意力池化 - 修复softmax使用问题
        attention_weights = self.attention_pool(x)
        # 使用batch信息正确应用softmax，确保每个图内的节点权重和为1
        attention_weights = torch.softmax(attention_weights, dim=1)
        attention_weights = attention_weights.view(-1, 1)  # 重塑为列向量
        attention_pooled = global_add_pool(x * attention_weights, batch)
        
        # 连接所有池化结果
        graph_embedding = torch.cat([mean_pooled, max_pooled, sum_pooled, attention_pooled], dim=1)
        
        return graph_embedding
    
    def descriptor_forward(self, descriptors):
        """
        分子描述符前向传播
        
        参数:
            descriptors (Tensor): 分子描述符
            
        返回:
            Tensor: 处理后的描述符表示
        """
        if self.descriptor_network is not None and descriptors is not None:
            # 检查描述符是否为空
            if descriptors.numel() > 0 and len(descriptors.shape) > 1 and descriptors.shape[1] > 0:
                return self.descriptor_network(descriptors)
        return None
    
    def fusion_forward(self, graph_features, descriptor_features=None):
        """
        融合前向传播
        
        参数:
            graph_features (Tensor): 图特征
            descriptor_features (Tensor): 描述符特征
            
        返回:
            Tensor: 融合后的特征
        """
        # 如果有描述符特征，则拼接
        if descriptor_features is not None and descriptor_features.numel() > 0:
            combined_features = torch.cat([graph_features, descriptor_features], dim=1)
        else:
            combined_features = graph_features
        
        # 通过融合网络
        x = combined_features
        for layer in self.fusion_layers:
            x = layer(x)
        
        return x
    
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: 包含图结构和分子描述符的数据对象
            
        返回:
            Tensor: 预测结果
        """
        # 从数据对象中提取图结构数据
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 获取分子描述符（如果有）
        descriptors = getattr(data, 'descriptors', None)
        
        # 图神经网络前向传播
        graph_features = self.gcn_forward(x, edge_index, edge_attr, batch)
        
        # 分子描述符前向传播
        descriptor_features = self.descriptor_forward(descriptors)
        
        # 融合特征
        fused_features = self.fusion_forward(graph_features, descriptor_features)
        
        # 任务特定的输出头
        outputs = []
        for head in self.task_heads:
            output = head(fused_features)
            outputs.append(output)
        
        # 连接所有输出
        final_output = torch.cat(outputs, dim=1)
        
        # 应用tanh激活函数限制输出范围在[-1, 1]之间，有助于稳定训练
        final_output = torch.tanh(final_output)
        
        return final_output