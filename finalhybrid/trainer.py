"""
模型训练器模块
==============

该模块提供了一个完整的训练框架，包括训练循环、验证、测试和模型保存功能。
"""

import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime


class MolecularTrainer:
    """
    分子模型训练器
    
    该类封装了模型训练的所有功能，包括训练循环、验证、测试和模型保存。
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device="cpu", experiment_dir="experiments", experiment_name=None):
        """
        初始化训练器
        
        参数:
            model (nn.Module): 要训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            test_loader (DataLoader): 测试数据加载器
            device (str): 训练设备 ("cpu" 或 "cuda")
            experiment_dir (str): 实验结果保存目录
            experiment_name (str): 实验名称
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_dir = experiment_dir
        
        # 设置实验名称和路径
        if experiment_name is None:
            self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
            
        self.experiment_path = os.path.join(self.experiment_dir, self.experiment_name)
        
        # 创建实验目录
        os.makedirs(self.experiment_path, exist_ok=True)
        
        # 初始化优化器和损失函数
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # 移动模型到指定设备
        self.model.to(self.device)
        print(f"Training on device: {self.device}")
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'lr': []
        }
    
    def train_epoch(self):
        """
        训练一个epoch
        
        返回:
            tuple: (平均损失, RMSE, R²)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in self.train_loader:
            # 将数据移动到设备
            batch = batch.to(self.device)
            
            # 获取目标值
            targets = batch.y
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch)
            
            # 确保输出和目标形状匹配
            if targets.dim() == 1:
                targets = targets.view(-1, 1)
            
            if outputs.shape != targets.shape:
                # 如果形状不匹配，尝试调整
                if outputs.size(0) == targets.size(0):
                    # 保持批次大小一致，调整特征维度
                    targets = targets.view(outputs.shape)
                else:
                    print(f"形状不匹配: outputs {outputs.shape}, targets {targets.shape}")
                    raise ValueError("输出和目标张量形状不匹配")
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累积损失和预测结果
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        
        # 计算RMSE和R²
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 确保数组形状正确
        if all_preds.ndim == 1:
            all_preds = all_preds.reshape(-1, 1)
        if all_targets.ndim == 1:
            all_targets = all_targets.reshape(-1, 1)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, rmse, r2
    
    def validate(self):
        """
        验证模型
        
        返回:
            tuple: (平均损失, RMSE, R²)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 将数据移动到设备
                batch = batch.to(self.device)
                
                # 获取目标值
                targets = batch.y
                
                # 前向传播
                outputs = self.model(batch)
                
                # 确保输出和目标形状匹配
                if targets.dim() == 1:
                    targets = targets.view(-1, 1)
                
                if outputs.shape != targets.shape:
                    # 如果形状不匹配，尝试调整
                    if outputs.size(0) == targets.size(0):
                        # 保持批次大小一致，调整特征维度
                        targets = targets.view(outputs.shape)
                    else:
                        print(f"形状不匹配: outputs {outputs.shape}, targets {targets.shape}")
                        raise ValueError("输出和目标张量形状不匹配")
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测结果和目标值
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算RMSE和R²
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 确保数组形状正确
        if all_preds.ndim == 1:
            all_preds = all_preds.reshape(-1, 1)
        if all_targets.ndim == 1:
            all_targets = all_targets.reshape(-1, 1)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, rmse, r2
    
    def train(self, num_epochs=100, patience=10, min_delta=1e-4, save_every=10):
        """
        训练模型
        
        参数:
            num_epochs (int): 训练轮数
            patience (int): 早停耐心值
            min_delta (float): 早停最小改善值
            save_every (int): 每N个epoch保存一次模型
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Patience: {patience}, Min delta: {min_delta}")
        
        if self.optimizer is None:
            raise ValueError("请先设置优化器 (trainer.optimizer = ...)")
        
        # 初始化早停变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(self.experiment_path, "best_model.pth")
        
        # 训练循环
        for epoch in range(num_epochs):
            start_time = datetime.now()
            
            # 训练一个epoch
            train_loss, train_rmse, train_r2 = self.train_epoch()
            
            # 验证
            val_loss, val_rmse, val_r2 = self.validate()
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新训练历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_rmse'].append(train_rmse)
            self.train_history['val_rmse'].append(val_rmse)
            self.train_history['train_r2'].append(train_r2)
            self.train_history['val_r2'].append(val_r2)
            self.train_history['lr'].append(current_lr)
            
            # 计算epoch时间
            epoch_time = (datetime.now() - start_time).total_seconds()
            
            # 打印训练信息
            improvement = best_val_loss - val_loss
            best_indicator = " *" if improvement > min_delta else ""
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | "
                  f"Train R²: {train_r2:6.3f} | "
                  f"Val R²: {val_r2:6.3f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s{best_indicator}")
            
            # 早停检查
            if improvement > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 定期保存模型
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(
                    self.experiment_path, 
                    f"checkpoint_epoch_{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
            
            # 检查早停条件
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best epoch: {epoch+1-patience}, Best validation loss: {best_val_loss:.4f}")
                break
        
        # 保存训练历史
        history_path = os.path.join(self.experiment_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 保存最终模型
        final_model_path = os.path.join(self.experiment_path, "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        
        print(f"Training curves saved to {os.path.join(self.experiment_path, 'training_curves.png')}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_history['train_loss'], label='Training Loss')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RMSE曲线
        axes[0, 1].plot(epochs, self.train_history['train_rmse'], label='Training RMSE')
        axes[0, 1].plot(epochs, self.train_history['val_rmse'], label='Validation RMSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Training and Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R²曲线
        axes[1, 0].plot(epochs, self.train_history['train_r2'], label='Training R²')
        axes[1, 0].plot(epochs, self.train_history['val_r2'], label='Validation R²')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Training and Validation R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(epochs, self.train_history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        curve_path = os.path.join(self.experiment_path, "training_curves.png")
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, data_loader=None):
        """
        评估模型性能
        
        参数:
            data_loader (DataLoader): 用于评估的数据加载器，默认使用测试加载器
            
        返回:
            dict: 包含评估指标的字典
        """
        if data_loader is None:
            data_loader = self.test_loader
            
        # 使用test方法获取结果
        _, rmse, r2, all_preds, all_targets = self.test()
        
        # 计算其他指标
        mae = mean_absolute_error(all_targets, all_preds)
        avg_loss = np.mean((all_preds - all_targets) ** 2)
        
        metrics = {
            'loss': float(avg_loss),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        return metrics
    
    def test(self):
        """
        测试模型
        
        返回:
            tuple: (平均损失, RMSE, R², 预测值, 真实值)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # 将数据移动到设备
                batch = batch.to(self.device)
                
                # 获取目标值
                targets = batch.y
                
                # 前向传播
                outputs = self.model(batch)
                
                # 确保输出和目标形状匹配
                if targets.dim() == 1:
                    targets = targets.view(-1, 1)
                
                if outputs.shape != targets.shape:
                    # 如果形状不匹配，尝试调整
                    if outputs.size(0) == targets.size(0):
                        # 保持批次大小一致，调整特征维度
                        targets = targets.view(outputs.shape)
                    else:
                        print(f"形状不匹配: outputs {outputs.shape}, targets {targets.shape}")
                        raise ValueError("输出和目标张量形状不匹配")
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测结果和目标值
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        
        # 计算RMSE和R²
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 确保数组形状正确
        if all_preds.ndim == 1:
            all_preds = all_preds.reshape(-1, 1)
        if all_targets.ndim == 1:
            all_targets = all_targets.reshape(-1, 1)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, rmse, r2, all_preds, all_targets

# Example usage
if __name__ == "__main__":
    from data_loader import load_example_data
    from gcn_model import MolecularGCN

    print("Setting up example training session...")

    try:
        # Load data
        train_loader, val_loader, test_loader, dataset = load_example_data()

        # Create model
        model = MolecularGCN(
            input_dim=40,
            hidden_dims=[128, 256, 512],
            output_dim=1,
            dropout_rate=0.2
        )

        # Create trainer
        trainer = MolecularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name="example_training"
        )

        # Setup optimizer and scheduler
        trainer.setup_optimizer("adam", learning_rate=0.001, weight_decay=1e-5)
        trainer.setup_scheduler("plateau", patience=10)

        print("Setup complete. Ready to start training!")
        print("To start training, call: trainer.train(num_epochs=100)")

    except Exception as e:
        print(f"Error during setup: {e}")