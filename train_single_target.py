"""
训练单目标优化模型
==================

该脚本训练单目标优化模型，专门针对每个靶点进行优化，
以提高测试集R2分数。
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List
from fingerprint_data_loader import FingerprintDataLoader
from single_target_model import SingleTargetModel
from sklearn.metrics import mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser(description='训练单目标优化模型')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--fingerprint-type', type=str, default='morgan', 
                       choices=['morgan', 'rdkit'], help='指纹类型')
    parser.add_argument('--n-bits', type=int, default=2048, help='指纹位数')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--output-dir', type=str, default='single_target_experiments', 
                       help='输出目录')
    parser.add_argument('--experiment-name', type=str, default=None, 
                       help='实验名称，默认为时间戳')
    parser.add_argument('--target', type=str, default=None,
                       help='特定目标名称，如果未指定则训练所有目标')
    
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
    
    print(f"开始训练单目标优化模型，实验名称: {args.experiment_name}")
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
    
    # 确定要训练的目标
    targets_to_train = [args.target] if args.target else target_columns
    trained_models = {}
    results = {}
    
    # 为每个目标训练模型
    for i, target in enumerate(target_columns):
        if target not in targets_to_train:
            continue
            
        print(f"\n{'='*50}")
        print(f"训练目标: {target}")
        print(f"{'='*50}")
        
        # 初始化模型
        model = SingleTargetModel(
            target_name=target,
            fingerprint_dim=args.n_bits,
            descriptor_dim=25
        )
        
        # 训练模型
        model.fit(X_train, y_train[:, i])
        
        # 评估模型
        train_metrics = model.evaluate(X_train, y_train[:, i])
        val_metrics = model.evaluate(X_val, y_val[:, i])
        test_metrics = model.evaluate(X_test, y_test[:, i])
        
        # 保存模型
        model_dir = os.path.join(experiment_dir, target)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{target}_model.pkl')
        model.save_model(model_path)
        trained_models[target] = model_path
        
        # 保存评估结果
        target_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        results[target] = target_results
        
        # 打印评估结果
        print(f"\n--- {target} 训练集评估结果 ---")
        for key, value in train_metrics.items():
            print(f"{key}: {value:.4f}")
            
        print(f"\n--- {target} 验证集评估结果 ---")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")
            
        print(f"\n--- {target} 测试集评估结果 ---")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # 保存预测结果
        test_predictions = model.predict(X_test)
        predictions_df = pd.DataFrame({
            'molecule_id': [f"TEST_{j:04d}" for j in range(len(test_predictions))],
            'true_value': y_test[:, i],
            'predicted_value': test_predictions
        })
        
        predictions_path = os.path.join(model_dir, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"测试集预测结果已保存到: {predictions_path}")
    
    # 保存整体结果
    overall_results = {
        'trained_models': trained_models,
        'results': results
    }
    
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    print(f"\n整体评估结果已保存到: {results_path}")
    
    print("\n所有目标训练完成!")


if __name__ == "__main__":
    main()