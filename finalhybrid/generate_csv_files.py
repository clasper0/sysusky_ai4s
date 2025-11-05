#!/usr/bin/env python3
"""
生成训练集、测试集真实值和预测值的CSV文件
"""

import torch
import os
import pandas as pd
import numpy as np
from data_loader import create_data_splits
from hybrid_model_final import HybridMolecularPredictor
from trainer import MolecularTrainer


def main():
    """主函数"""
    print("生成训练集、测试集真实值和预测值CSV文件")
    print("=" * 50)
    
    # 数据和模型参数
    DATA_PATH = "data/candidate_hybrid.csv"
    SMILES_COL = "SMILES"
    TARGET_COL = "target1_pIC50,target2_pIC50,target3_pIC50,target4_pIC50,target5_pIC50"
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    BATCH_SIZE = 16
    DEVICE = "cpu"
    
    # 实验目录
    experiment_dirs = [d for d in os.listdir("experiments") if d.startswith("experiment_")]
    if not experiment_dirs:
        print("未找到实验目录")
        return
    
    # 获取最新的实验目录
    experiment_dirs.sort(reverse=True)
    latest_experiment = experiment_dirs[0]
    experiment_path = os.path.join("experiments", latest_experiment)
    print(f"使用实验目录: {experiment_path}")
    
    # 加载原始数据
    print("加载原始数据...")
    original_df = pd.read_csv(DATA_PATH)
    print(f"原始数据集包含 {len(original_df)} 个分子")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader, full_dataset = create_data_splits(
        DATA_PATH,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE,
        smiles_col=SMILES_COL,
        target_col=TARGET_COL
    )
    
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    
    # 生成训练集CSV文件
    print("\n生成训练集CSV文件...")
    train_smiles = []
    train_targets = []
    
    for idx in train_loader.dataset.indices:
        train_smiles.append(full_dataset.smiles_list[idx])
        # 获取原始目标值（未标准化）
        original_idx = full_dataset.valid_indices[idx]
        target_row = original_df.iloc[original_idx]
        target_values = [target_row[col] for col in TARGET_COL.split(',')]
        train_targets.append(target_values)
    
    train_df = pd.DataFrame({
        'SMILES': train_smiles,
        'target1_pIC50': [t[0] for t in train_targets],
        'target2_pIC50': [t[1] for t in train_targets],
        'target3_pIC50': [t[2] for t in train_targets],
        'target4_pIC50': [t[3] for t in train_targets],
        'target5_pIC50': [t[4] for t in train_targets]
    })
    
    train_csv_path = os.path.join(experiment_path, "train_set.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"训练集已保存到: {train_csv_path}")
    
    # 生成测试集真实值CSV文件
    print("\n生成测试集真实值CSV文件...")
    test_smiles = []
    test_targets = []
    
    for idx in test_loader.dataset.indices:
        test_smiles.append(full_dataset.smiles_list[idx])
        # 获取原始目标值（未标准化）
        original_idx = full_dataset.valid_indices[idx]
        target_row = original_df.iloc[original_idx]
        target_values = [target_row[col] for col in TARGET_COL.split(',')]
        test_targets.append(target_values)
    
    test_targets_df = pd.DataFrame({
        'SMILES': test_smiles,
        'target1_pIC50_true': [t[0] for t in test_targets],
        'target2_pIC50_true': [t[1] for t in test_targets],
        'target3_pIC50_true': [t[2] for t in test_targets],
        'target4_pIC50_true': [t[3] for t in test_targets],
        'target5_pIC50_true': [t[4] for t in test_targets]
    })
    
    test_targets_csv_path = os.path.join(experiment_path, "test_targets.csv")
    test_targets_df.to_csv(test_targets_csv_path, index=False)
    print(f"测试集真实值已保存到: {test_targets_csv_path}")
    
    # 生成测试集预测值CSV文件
    print("\n生成测试集预测值CSV文件...")
    
    # 加载训练好的模型
    model = HybridMolecularPredictor(
        gcn_input_dim=36,
        gcn_hidden_dims=[64, 128],
        descriptor_dim=0,
        fusion_hidden_dims=[256, 128],
        num_targets=5
    )
    
    model_path = os.path.join(experiment_path, "best_model.pth")
    if os.path.exists(model_path):
        # 使用weights_only=False来解决加载问题
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"模型已加载: {model_path}")
    else:
        print(f"未找到模型文件: {model_path}")
        return
    
    model.to(DEVICE)
    model.eval()
    
    # 进行预测
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            predictions = outputs.cpu().numpy()
            test_predictions.append(predictions)
    
    # 合并所有预测结果
    test_predictions = np.concatenate(test_predictions, axis=0)
    
    # 现在不需要反标准化了，因为模型直接输出原始值
    test_predictions_df = pd.DataFrame({
        'SMILES': test_smiles,
        'target1_pIC50_pred': test_predictions[:, 0],
        'target2_pIC50_pred': test_predictions[:, 1],
        'target3_pIC50_pred': test_predictions[:, 2],
        'target4_pIC50_pred': test_predictions[:, 3],
        'target5_pIC50_pred': test_predictions[:, 4]
    })
    
    test_predictions_csv_path = os.path.join(experiment_path, "test_predictions.csv")
    test_predictions_df.to_csv(test_predictions_csv_path, index=False)
    print(f"测试集预测值已保存到: {test_predictions_csv_path}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()