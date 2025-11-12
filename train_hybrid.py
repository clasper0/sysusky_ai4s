"""
训练混合模型预测多靶点pIC50值
================================

该脚本训练一个混合模型，结合图神经网络和分子描述符特征，
用于同时预测分子对多个靶点的pIC50值。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from typing import Dict, List
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from hybrid_model import HybridMolecularPredictor, load_descriptor_data
from data_loader import create_data_splits
from smiles_to_graph import SMILESToGraph
from torch_geometric.data import Data, Batch


class HybridTrainer:
    """混合模型训练器"""
    
    def __init__(
        self,
        model: HybridMolecularPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        descriptor_features: np.ndarray = None,
        device: str = "cpu"
    ):
        """
        初始化训练器
        
        Args:
            model: 混合模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            descriptor_features: 分子描述符特征
            device: 训练设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.descriptor_features = descriptor_features
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            
            # 获取对应的描述符特征
            if self.descriptor_features is not None:
                start_idx = batch_idx * self.train_loader.batch_size
                end_idx = min(start_idx + self.train_loader.batch_size, len(self.descriptor_features))
                descriptor_batch = torch.FloatTensor(self.descriptor_features[start_idx:end_idx]).to(self.device)
            else:
                descriptor_batch = None
            
            # 前向传播
            predictions = self.model(batch, descriptor_batch)
            targets = batch.y.view(-1, self.model.num_targets)
            
            # 计算损失
            loss = nn.MSELoss()(predictions, targets)
            
            # 反向传播
            optimizer = getattr(self, 'optimizer', None)
            if optimizer is None:
                raise ValueError("请先调用setup_optimizer方法设置优化器")
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积统计信息
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        # 计算整体指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        avg_loss = total_loss / total_samples
        rmse_per_target = []
        for i in range(self.model.num_targets):
            rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_predictions[:, i]))
            rmse_per_target.append(rmse)
        
        avg_rmse = np.mean(rmse_per_target)
        
        return {
            'loss': avg_loss,
            'rmse': avg_rmse,
            'rmse_per_target': rmse_per_target
        }
    
    def evaluate(self, data_loader: DataLoader, descriptor_features: np.ndarray = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            descriptor_features: 分子描述符特征
            
        Returns:
            Dict[str, float]: 评估指标
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                
                # 获取对应的描述符特征
                if descriptor_features is not None:
                    start_idx = batch_idx * data_loader.batch_size
                    end_idx = min(start_idx + data_loader.batch_size, len(descriptor_features))
                    descriptor_batch = torch.FloatTensor(descriptor_features[start_idx:end_idx]).to(self.device)
                else:
                    descriptor_batch = None
                
                # 前向传播
                predictions = self.model(batch, descriptor_batch)
                targets = batch.y.view(-1, self.model.num_targets)
                
                # 计算损失
                loss = nn.MSELoss()(predictions, targets)
                
                # 累积统计信息
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算整体指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        avg_loss = total_loss / total_samples
        rmse_per_target = []
        r2_per_target = []
        
        for i in range(self.model.num_targets):
            rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_predictions[:, i]))
            r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            rmse_per_target.append(rmse)
            r2_per_target.append(r2)
        
        avg_rmse = np.mean(rmse_per_target)
        avg_r2 = np.mean(r2_per_target)
        
        return {
            'loss': avg_loss,
            'rmse': avg_rmse,
            'r2': avg_r2,
            'rmse_per_target': rmse_per_target,
            'r2_per_target': r2_per_target
        }
    
    def setup_optimizer(self, optimizer_type: str = 'adam', learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        设置优化器
        
        Args:
            optimizer_type: 优化器类型 ('adam', 'adamw', 'sgd')
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def train(self, num_epochs: int = 100, patience: int = 20):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练 {num_epochs} 轮...")
        print(f"设备: {self.device}")
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate(self.val_loader, self.descriptor_features)
            
            # 记录历史
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_rmse'].append(train_metrics['rmse'])
            self.train_history['val_rmse'].append(val_metrics['rmse'])
            
            # 打印进度
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Train RMSE: {train_metrics['rmse']:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.4f}")
            
            # 早停检查
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_hybrid_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，在第 {epoch} 轮后停止训练")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_hybrid_model.pth'))
        print("训练完成，已加载最佳模型")
    
    def test(self) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Returns:
            Dict[str, float]: 测试指标
        """
        test_metrics = self.evaluate(self.test_loader, self.descriptor_features)
        return test_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练混合模型预测多靶点pIC50值")
    
    # 数据参数
    parser.add_argument("--data-path", type=str, default="data/candidate.csv",
                       help="数据文件路径")
    parser.add_argument("--smiles-col", type=str, default="SMILES",
                       help="SMILES列名")
    parser.add_argument("--target-cols", type=str, nargs='+', 
                       default=["target1_pIC50", "target2_pIC50", "target3_pIC50", "target4_pIC50", "target5_pIC50"],
                       help="目标列名列表")
    parser.add_argument("--descriptor-cols", type=str, nargs='+', default=[],
                       help="分子描述符列名列表")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="测试集比例")
    parser.add_argument("--val-size", type=float, default=0.1,
                       help="验证集比例")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批次大小")
    
    # 模型参数
    parser.add_argument("--gcn-hidden-dims", type=int, nargs='+', default=[128, 256, 512],
                       help="GCN隐藏层维度")
    parser.add_argument("--fusion-hidden-dims", type=int, nargs='+', default=[512, 256],
                       help="融合层隐藏层维度")
    parser.add_argument("--gcn-dropout", type=float, default=0.2,
                       help="GCN Dropout率")
    parser.add_argument("--fusion-dropout", type=float, default=0.1,
                       help="融合层Dropout率")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--optimizer", type=str, default="adam",
                       choices=["adam", "adamw", "sgd"],
                       help="优化器类型")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--patience", type=int, default=20,
                       help="早停耐心值")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="训练设备")
    
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("混合模型多靶点pIC50预测训练")
    print("=" * 50)
    print(f"数据文件: {args.data_path}")
    print(f"设备: {device}")
    print(f"靶点数量: {len(args.target_cols)}")
    print(f"分子描述符数量: {len(args.descriptor_cols)}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = create_data_splits(
        data_path=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        smiles_col=args.smiles_col,
        target_col=','.join(args.target_cols)  # 多目标列
    )
    
    # 加载分子描述符特征
    descriptor_features = None
    if args.descriptor_cols:
        print(f"加载分子描述符特征: {len(args.descriptor_cols)} 个特征")
        descriptor_features = load_descriptor_data(args.data_path, args.descriptor_cols)
        print(f"描述符特征形状: {descriptor_features.shape}")
    
    # 创建模型
    print("\n创建模型...")
    model = HybridMolecularPredictor(
        gcn_input_dim=36,  # 原子特征维度
        gcn_hidden_dims=args.gcn_hidden_dims,
        descriptor_dim=len(args.descriptor_cols),
        num_targets=len(args.target_cols),
        gcn_dropout_rate=args.gcn_dropout,
        fusion_hidden_dims=args.fusion_hidden_dims,
        fusion_dropout_rate=args.fusion_dropout
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 创建训练器
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        descriptor_features=descriptor_features,
        device=device
    )
    
    # 设置优化器
    trainer.setup_optimizer(
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 开始训练
    trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # 测试模型
    print("\n在测试集上评估模型...")
    test_metrics = trainer.test()
    
    print(f"测试结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    for i, target_col in enumerate(args.target_cols):
        print(f"  {target_col} - RMSE: {test_metrics['rmse_per_target'][i]:.4f}, R²: {test_metrics['r2_per_target'][i]:.4f}")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()