#!/usr/bin/env python3
"""
使用扩展分子特征训练混合模型
============================

该脚本使用包含DeepChem生成的扩展分子特征的数据集来训练混合模型。
"""

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from data_loader import create_data_splits, MolecularDataLoader
from hybrid_model_final import HybridMolecularPredictor
from trainer import MolecularTrainer
from smiles_to_graph import SMILESToGraph
from datetime import datetime


def main():
    """主训练函数"""
    print("使用扩展分子特征训练混合模型")
    print("=" * 50)
    
    # 数据和模型参数
    DATA_PATH = "data/candidate_hybrid_Extended.csv"
    SMILES_COL = "SMILES"
    TARGET_COL = "target1_pIC50,target2_pIC50,target3_pIC50,target4_pIC50,target5_pIC50"
    # 包含原有的分子描述符和新增的DeepChem特征
    DESCRIPTOR_COLS = ["MolWt", "LogP", "HBA", "HBD"] + [f"rdkit_{i}" for i in range(200)]
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    BATCH_SIZE = 16
    EPOCHS = 30000
    LEARNING_RATE = 0.001
    DEVICE = "cpu"
    EXPERIMENT_DIR = "experiments"
    EXPERIMENT_NAME = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"错误: 未找到数据文件 {DATA_PATH}")
        print("请先运行 generate_extended_features.py 脚本生成扩展数据集")
        return
    
    # 加载数据以检查格式
    df = pd.read_csv(DATA_PATH)
    print(f"数据集包含 {len(df)} 个分子")
    print(f"列名总数: {len(df.columns)}")
    print(f"前10列: {list(df.columns)[:10]}")
    print(f"后10列: {list(df.columns)[-10:]}")
    print()
    
    # 创建数据分割
    print("创建数据分割...")
    try:
        train_loader, val_loader, test_loader, full_dataset = create_data_splits(
            DATA_PATH,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            batch_size=BATCH_SIZE,
            smiles_col=SMILES_COL,
            target_col=TARGET_COL,
            descriptor_cols=DESCRIPTOR_COLS
        )
        print("数据分割创建成功!")
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  测试集: {len(test_loader.dataset)} 样本")
        print()
    except Exception as e:
        print(f"创建数据分割失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 打印训练集和测试集的SMILES示例
    print("训练集SMILES示例:")
    train_smiles_examples = [full_dataset.smiles_list[i] for i in train_loader.dataset.indices[:5]]
    for i, smiles in enumerate(train_smiles_examples, 1):
        print(f"  {i}. {smiles}")
    
    print("\n测试集SMILES示例:")
    test_smiles_examples = [full_dataset.smiles_list[i] for i in test_loader.dataset.indices[:5]]
    for i, smiles in enumerate(test_smiles_examples, 1):
        print(f"  {i}. {smiles}")
    print()

    # 创建模型
    print("创建混合模型...")
    try:
        # 使用实际的描述符维度（原有4个+新增200个RDKit描述符）
        descriptor_dim = len(DESCRIPTOR_COLS)
        model = HybridMolecularPredictor(
            gcn_input_dim=36,       # 原子特征维度
            gcn_hidden_dims=[64, 128],  # 图神经网络隐藏层维度
            descriptor_dim=descriptor_dim,  # 使用实际的描述符特征维度
            fusion_hidden_dims=[640, 256],  # 调整融合层维度以匹配实际输入 (512图特征 + 128描述符特征)
            num_targets=5           # 5个靶点
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("模型创建成功!")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数数量: {trainable_params:,}")
        print(f"  使用的分子描述符数量: {descriptor_dim}")
        print()
    except Exception as e:
        print(f"创建模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建训练器
    print("创建训练器...")
    try:
        trainer = MolecularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            experiment_dir=EXPERIMENT_DIR,
            experiment_name=EXPERIMENT_NAME
        )
        
        # 初始化优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        trainer.optimizer = optimizer
        
        print("训练器创建成功!")
        print()
    except Exception as e:
        print(f"创建训练器失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 开始训练
    print("开始训练...")
    print(f"训练将在 {DEVICE} 上进行")
    try:
        trainer.train(
            num_epochs=EPOCHS,
            patience=20,
            min_delta=1e-4,
            save_every=5
        )
        print("训练完成!")
        print()
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 保存最终模型
    try:
        final_model_path = os.path.join(trainer.experiment_path, "final_hybrid_model_extended.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
    except Exception as e:
        print(f"保存最终模型失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()