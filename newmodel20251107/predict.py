"""
预测脚本
========

该脚本用于使用训练好的模型进行预测。
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from data_loader import extract_rdkit_features
from advanced_ensemble_model import AdvancedEnsembleModel, StackingEnsembleModel


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--model-dir', type=str, required=True, 
                        help='模型目录路径')
    parser.add_argument('--input', type=str, required=True, 
                        help='输入SMILES文件路径')
    parser.add_argument('--output', type=str, required=True, 
                        help='输出预测结果文件路径')
    parser.add_argument('--fingerprint-type', type=str, default='morgan', 
                        choices=['morgan', 'rdkit'], help='指纹类型')
    parser.add_argument('--n-bits', type=int, default=2048, help='指纹位数')
    
    args = parser.parse_args()
    
    # 检查模型目录是否存在
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"模型目录不存在: {args.model_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    # 加载配置
    config_path = os.path.join(args.model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"加载模型配置: {config}")
    else:
        print("未找到配置文件，使用默认参数")
    
    # 加载输入数据
    print("加载输入数据...")
    input_df = pd.read_csv(args.input)
    smiles_list = input_df['SMILES'].tolist()
    print(f"加载了 {len(smiles_list)} 个分子")
    
    # 提取特征
    print("提取RDKit特征...")
    features = extract_rdkit_features(smiles_list, args.fingerprint_type, args.n_bits)
    print(f"特征维度: {features.shape}")
    
    # 检查是否有无效特征
    invalid_count = np.sum(np.isnan(features)) + np.sum(np.isinf(features))
    if invalid_count > 0:
        print(f"警告: 发现 {invalid_count} 个无效特征值，将替换为0")
        features = np.nan_to_num(features)
    
    # 加载模型
    print("加载模型...")
    model_path = os.path.join(args.model_dir, 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 根据模型类型加载模型
    model_type = config.get('model_type', 'advanced') if 'config' in locals() else 'advanced'
    if model_type == 'advanced':
        model = AdvancedEnsembleModel.load_model(model_path)
    else:
        model = StackingEnsembleModel.load_model(model_path)
    
    # 进行预测
    print("进行预测...")
    predictions = model.predict(features)
    
    # 保存预测结果
    print("保存预测结果...")
    results_df = pd.DataFrame({
        'SMILES': smiles_list,
        'target1_pIC50': predictions[:, 0],
        'target2_pIC50': predictions[:, 1],
        'target3_pIC50': predictions[:, 2],
        'target4_pIC50': predictions[:, 3],
        'target5_pIC50': predictions[:, 4]
    })
    
    results_df.to_csv(args.output, index=False)
    print(f"预测结果已保存到: {args.output}")
    
    print("预测完成!")


if __name__ == "__main__":
    main()