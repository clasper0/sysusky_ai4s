"""
使用训练好的指纹模型进行预测
================================

该脚本加载训练好的指纹模型并对新的分子进行pIC50预测。
"""

import argparse
import os
import numpy as np
import pandas as pd
from typing import List
from fingerprint_extractor import FingerprintExtractor
from fingerprint_model import FingerprintModel
from sklearn.preprocessing import StandardScaler
import joblib
import json


def main():
    parser = argparse.ArgumentParser(description='使用指纹模型进行预测')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='训练好的模型文件路径')
    parser.add_argument('--input', type=str, required=True, 
                       help='输入文件路径（包含SMILES的CSV文件）')
    parser.add_argument('--output', type=str, required=True, 
                       help='输出文件路径')
    parser.add_argument('--smiles-column', type=str, default='smiles', 
                       help='SMILES列名')
    parser.add_argument('--id-column', type=str, default=None, 
                       help='分子ID列名，如果没有则自动生成')
    
    args = parser.parse_args()
    
    # 加载模型
    print("加载模型...")
    model = FingerprintModel.load_model(args.model_path)
    
    # 加载输入数据
    print("加载输入数据...")
    input_df = pd.read_csv(args.input)
    
    if args.smiles_column not in input_df.columns:
        raise ValueError(f"找不到SMILES列: {args.smiles_column}")
    
    # 提取SMILES
    smiles_list = input_df[args.smiles_column].tolist()
    
    # 提取分子ID
    if args.id_column and args.id_column in input_df.columns:
        molecule_ids = input_df[args.id_column].tolist()
    else:
        molecule_ids = [f"MOL_{i:04d}" for i in range(len(smiles_list))]
    
    # 提取指纹特征
    print("提取指纹特征...")
    # 从模型元数据获取指纹类型
    metadata_path = args.model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    extractor = FingerprintExtractor(
        fingerprint_type=metadata.get('fingerprint_type', 'morgan'),
        n_bits=2048  # 默认值，实际应该从模型训练时的配置获取
    )
    
    features, valid_indices = extractor.extract_features(smiles_list)
    
    if len(valid_indices) != len(smiles_list):
        print(f"警告: {len(smiles_list) - len(valid_indices)} 个分子无法处理")
        # 更新分子ID
        molecule_ids = [molecule_ids[i] for i in valid_indices]
    
    # 加载标准化器（从模型训练时的实验目录中）
    model_dir = os.path.dirname(args.model_path)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)
    else:
        print("警告: 未找到特征标准化器，使用原始特征进行预测")
    
    # 进行预测
    print("进行预测...")
    predictions = model.predict(features)
    
    # 保存结果
    print("保存结果...")
    results_df = pd.DataFrame(predictions, columns=model.target_columns)
    results_df.insert(0, 'molecule_id', molecule_ids)
    
    # 如果原始数据中有其他列，也一并保存
    valid_input_df = input_df.iloc[valid_indices].reset_index(drop=True)
    for col in valid_input_df.columns:
        if col not in results_df.columns:
            results_df[col] = valid_input_df[col]
    
    results_df.to_csv(args.output, index=False)
    print(f"预测结果已保存到: {args.output}")
    
    # 显示预测统计信息
    print("\n预测统计信息:")
    for target in model.target_columns:
        target_values = results_df[target]
        print(f"{target}: 均值={target_values.mean():.4f}, "
              f"标准差={target_values.std():.4f}, "
              f"最小值={target_values.min():.4f}, "
              f"最大值={target_values.max():.4f}")


if __name__ == "__main__":
    main()