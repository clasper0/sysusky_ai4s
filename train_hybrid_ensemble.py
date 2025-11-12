"""
训练混合集成学习模型
====================

该脚本训练一个混合集成学习模型，结合分子指纹、描述符和其他特征，
用于预测分子对多个靶点的pIC50值。
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List
from fingerprint_data_loader import FingerprintDataLoader
from hybrid_ensemble_model import HybridEnsembleModel
from sklearn.metrics import mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser(description='训练混合集成学习模型')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--fingerprint-type', type=str, default='morgan', 
                       choices=['morgan', 'rdkit'], help='指纹类型')
    parser.add_argument('--n-bits', type=int, default=2048, help='指纹位数')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--output-dir', type=str, default='hybrid_ensemble_experiments', 
                       help='输出目录')
    parser.add_argument('--experiment-name', type=str, default=None, 
                       help='实验名称，默认为时间戳')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.experiment_name is None:
        import datetime
        args.experiment_name = datetime.datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"开始训练混合集成模型，实验名称: {args.experiment_name}")
    print(f"指纹类型: {args.fingerprint_type}")
    print(f"指纹位数: {args.n_bits}")
    
    # 初始化数据加载器
    data_loader = FingerprintDataLoader(
        fingerprint_type=args.fingerprint_type,
        n_bits=args.n_bits
    )
    
    # 文件路径
    activity_file = os.path.join(args.data_dir, 'activity_train.csv')
    smiles_file = os.path.join(args.data_dir, 'molecule.smi')
    property_file = os.path.join(args.data_dir, 'property.csv')
    
    # 检查文件是否存在
    if not os.path.exists(activity_file):
        raise FileNotFoundError(f"活性数据文件不存在: {activity_file}")
    if not os.path.exists(smiles_file):
        raise FileNotFoundError(f"SMILES文件不存在: {smiles_file}")
    
    # 加载数据
    print("加载数据...")
    activity_df, smiles_df, property_df = data_loader.load_data(
        activity_file, smiles_file, property_file
    )
    
    # 准备数据集
    print("准备数据集...")
    features, targets, molecule_ids, target_columns = data_loader.prepare_dataset(
        activity_df, smiles_df, property_df
    )
    
    # 划分和标准化数据
    print("划分和标准化数据...")
    (X_train, X_val, X_test), (y_train, y_val, y_test) = data_loader.split_and_scale_data(
        features, targets, test_size=args.test_size, val_size=args.val_size
    )
    
    # 保存标准化器
    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
    data_loader.save_scaler(scaler_path)
    print(f"特征标准化器已保存到: {scaler_path}")
    
    # 初始化模型
    print("初始化模型...")
    # 指纹维度是n_bits，描述符维度是25（根据fingerprint_extractor中的定义）
    model = HybridEnsembleModel(
        fingerprint_dim=args.n_bits,
        descriptor_dim=25
    )
    
    # 训练模型
    print("训练模型...")
    model.fit(X_train, y_train, target_columns)
    
    # 评估模型
    print("评估模型...")
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)
    
    # 打印评估结果
    print("\n=== 训练集评估结果 ===")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")
        
    print("\n=== 验证集评估结果 ===")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")
        
    print("\n=== 测试集评估结果 ===")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 保存模型
    model_path = os.path.join(experiment_dir, 'hybrid_ensemble_model.pkl')
    model.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 保存评估结果
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"评估结果已保存到: {results_path}")
    
    # 保存预测结果
    test_predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(test_predictions, columns=target_columns)
    predictions_df['molecule_id'] = [f"TEST_{i:04d}" for i in range(len(test_predictions))]
    
    predictions_path = os.path.join(experiment_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"测试集预测结果已保存到: {predictions_path}")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()