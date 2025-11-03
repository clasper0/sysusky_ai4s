#!/usr/bin/env python3
"""
诊断脚本，用于检查数据加载和模型创建的基本功能。
"""

import os
import sys
import torch

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_data_splits
from gcn_model import create_model

def main():
    print("诊断脚本开始执行...")
    print("=" * 30)
    
    # 检查数据文件
    DATA_PATH = "data/candidate_hybrid.csv"
    print(f"检查数据文件: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到数据文件 {DATA_PATH}")
        return
    
    print("数据文件存在")
    
    # 尝试加载数据
    print("尝试加载数据...")
    try:
        train_loader, val_loader, test_loader, dataset = create_data_splits(
            DATA_PATH,
            test_size=0.2,
            val_size=0.15,
            batch_size=16
        )
        print("数据加载成功!")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查批次数据
    print("检查批次数据...")
    try:
        batch = next(iter(train_loader))
        print(f"批次大小: {len(batch)}")
        print(f"节点特征形状: {batch.x.shape}")
        print(f"边索引形状: {batch.edge_index.shape}")
        print(f"目标值形状: {batch.y.shape}")
    except Exception as e:
        print(f"批次数据检查失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 尝试创建模型
    print("尝试创建模型...")
    try:
        model = create_model(
            "multi_task",
            input_dim=36,
            hidden_dims=[64, 128],
            output_dim=batch.y.shape[1] if len(batch.y.shape) > 1 else 1,
            dropout_rate=0.2
        )
        print("模型创建成功!")
        print(f"模型类型: {type(model).__name__}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数数量: {total_params:,}")
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 尝试前向传播
    print("尝试前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(batch)
            print(f"输出形状: {output.shape}")
        print("前向传播成功!")
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n诊断完成，所有测试通过!")

if __name__ == "__main__":
    main()