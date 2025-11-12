"""
半监督直推学习模型
==================

该模块实现了一个基于图神经网络的半监督直推学习模型，
用于提高小样本场景下的分子活性预测能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGEConv
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json
import os


class TransductiveGCN(nn.Module):
    """
    基于图卷积网络的直推学习模型
    """

    def __init__(
        self, 
        input_dim: int = 36, 
        hidden_dim: int = 64, 
        num_layers: int = 3, 
        dropout: float = 0.3, 
        target_dim: int = 5,
        use_sage: bool = False
    ):
        """
        初始化直推学习模型

        Args:
            input_dim: 输入特征维度（原子特征维度）
            hidden_dim: 隐藏层维度
            num_layers: GCN层数
            dropout: Dropout概率
            target_dim: 目标维度（5个靶点）
            use_sage: 是否使用GraphSAGE替代GCN
        """
        super(TransductiveGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.target_dim = target_dim
        self.use_sage = use_sage
        
        # 图卷积层
        self.convs = nn.ModuleList()
        if use_sage:
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 全局池化层
        self.global_mean_pool = global_mean_pool
        self.global_max_pool = global_max_pool
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, target_dim)
        )
        
    def forward(self, data):
        """
        前向传播

        Args:
            data: 图数据对象

        Returns:
            预测值
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 全局池化
        mean_pool = self.global_mean_pool(x, batch)
        max_pool = self.global_max_pool(x, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        
        # 分类器
        out = self.classifier(graph_embedding)
        return out


class TransductiveLearner:
    """
    直推学习训练器，实现半监督学习策略
    """

    def __init__(self, model: TransductiveGCN, device: str = 'cpu'):
        """
        初始化直推学习训练器

        Args:
            model: 图神经网络模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # 存储所有数据（用于直推学习）
        self.all_data = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
    def prepare_transductive_data(self, train_data, val_data, test_data, 
                                 unlabeled_data=None):
        """
        准备直推学习数据

        Args:
            train_data: 训练数据列表
            val_data: 验证数据列表
            test_data: 测试数据列表
            unlabeled_data: 无标签数据列表（可选）

        Returns:
            构建的图数据对象
        """
        # 合并所有数据
        all_graphs = train_data + val_data + test_data
        if unlabeled_data:
            all_graphs += unlabeled_data
            
        # 创建批次数据
        batch = Batch.from_data_list(all_graphs)
        
        # 创建掩码
        train_size = len(train_data)
        val_size = len(val_data)
        test_size = len(test_data)
        unlabeled_size = len(unlabeled_data) if unlabeled_data else 0
        
        total_size = train_size + val_size + test_size + unlabeled_size
        
        self.train_mask = torch.zeros(total_size, dtype=torch.bool)
        self.val_mask = torch.zeros(total_size, dtype=torch.bool)
        self.test_mask = torch.zeros(total_size, dtype=torch.bool)
        
        self.train_mask[:train_size] = True
        self.val_mask[train_size:train_size+val_size] = True
        self.test_mask[train_size+val_size:train_size+val_size+test_size] = True
        
        # 存储数据
        self.all_data = batch.to(self.device)
        
        print(f"直推学习数据准备完成:")
        print(f"  总样本数: {total_size}")
        print(f"  训练样本: {train_size}")
        print(f"  验证样本: {val_size}")
        print(f"  测试样本: {test_size}")
        print(f"  无标签样本: {unlabeled_size}")
        
        return batch

    def train_epoch(self):
        """
        训练一个epoch

        Returns:
            训练损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        out = self.model(self.all_data)
        
        # 只计算训练集上的损失
        train_out = out[self.train_mask]
        train_y = self.all_data.y[self.train_mask]
        
        loss = self.criterion(train_out, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, mask):
        """
        评估模型

        Args:
            mask: 评估掩码

        Returns:
            损失和预测结果
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.all_data)
            pred = out[mask]
            true = self.all_data.y[mask]
            
            loss = self.criterion(pred, true)
            
            # 计算指标
            pred_np = pred.cpu().numpy()
            true_np = true.cpu().numpy()
            
            mse = mean_squared_error(true_np, pred_np)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_np, pred_np)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            return loss.item(), metrics, pred_np, true_np

    def train(self, epochs: int = 200, patience: int = 20):
        """
        训练模型

        Args:
            epochs: 训练轮数
            patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练直推学习模型...")
        print(f"训练轮数: {epochs}")
        print(f"早停耐心值: {patience}")
        
        train_losses = []
        val_losses = []
        val_r2s = []
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 验证
            val_loss, val_metrics, _, _ = self.evaluate(self.val_mask)
            val_losses.append(val_loss)
            val_r2s.append(val_metrics['r2'])
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val R2: {val_metrics['r2']:.4f}")
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_transductive_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，最佳验证损失: {best_val_loss:.4f}")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_transductive_model.pth'))
        print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")

    def predict(self):
        """
        在测试集上进行预测

        Returns:
            测试集预测结果和评估指标
        """
        test_loss, test_metrics, test_pred, test_true = self.evaluate(self.test_mask)
        print(f"测试集评估结果:")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  R2: {test_metrics['r2']:.4f}")
        
        return test_pred, test_true, test_metrics

    def save_model(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 模型保存路径
        """
        # 创建模型保存目录
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存模型状态字典
        torch.save(self.model.state_dict(), filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """
        加载模型

        Args:
            filepath: 模型文件路径
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"模型已从 {filepath} 加载")