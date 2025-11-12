"""
训练脚本
========

该脚本用于训练高级集成模型。
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys
sys.path.append('.')

from data_loader import load_candidate_data, extract_rdkit_features, split_data
from advanced_ensemble_model import AdvancedEnsembleModel, StackingEnsembleModel


def main():
    parser = argparse.ArgumentParser(description='训练高级集成模型')
    parser.add_argument('--data-path', type=str, default='../data/candidate_hybrid.csv', 
                        help='数据文件路径')
    parser.add_argument('--output-dir', type=str, default='experiments', 
                        help='输出目录')
    parser.add_argument('--fingerprint-type', type=str, default='morgan', 
                        choices=['morgan', 'rdkit'], help='指纹类型')
    parser.add_argument('--n-bits', type=int, default=2048, help='指纹位数')
    parser.add_argument('--n-features', type=int, default=1000, help='最大特征数量')
    parser.add_argument('--model-type', type=str, default='advanced', 
                        choices=['advanced', 'stacking'], help='模型类型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 保存配置
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("开始训练模型...")
    print(f"数据路径: {args.data_path}")
    print(f"指纹类型: {args.fingerprint_type}")
    print(f"指纹位数: {args.n_bits}")
    print(f"最大特征数: {args.n_features}")
    print(f"模型类型: {args.model_type}")
    
    # 加载数据
    print("加载数据...")
    smiles_list, labels, df = load_candidate_data(args.data_path)
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
    
    # 划分数据集
    print("划分数据集...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 初始化模型
    if args.model_type == 'advanced':
        model = AdvancedEnsembleModel(n_features=args.n_features)
    else:
        model = StackingEnsembleModel(n_features=args.n_features)
    
    # 训练模型
    print("训练模型...")
    model.fit(X_train, y_train)
    
    # 在验证集上评估
    print("在验证集上评估...")
    val_metrics = model.evaluate(X_val, y_val)
    print("验证集评估结果:")
    for target, metrics in val_metrics.items():
        print(f"  {target}:")
        print(f"    MSE: {metrics['mse']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    R2: {metrics['r2']:.4f}")
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_metrics = model.evaluate(X_test, y_test)
    print("测试集评估结果:")
    for target, metrics in test_metrics.items():
        print(f"  {target}:")
        print(f"    MSE: {metrics['mse']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    R2: {metrics['r2']:.4f}")
    
    # 保存模型
    model_path = os.path.join(args.output_dir, 'model.pkl')
    model.save_model(model_path)
    
    # 保存模型评估结果
    results = {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"评估结果已保存到: {results_path}")
    
    # 保存测试集预测结果
    y_test_pred = model.predict(X_test)
    predictions_df = pd.DataFrame({
        'true_target1': y_test[:, 0],
        'pred_target1': y_test_pred[:, 0],
        'true_target2': y_test[:, 1],
        'pred_target2': y_test_pred[:, 1],
        'true_target3': y_test[:, 2],
        'pred_target3': y_test_pred[:, 2],
        'true_target4': y_test[:, 3],
        'pred_target4': y_test_pred[:, 3],
        'true_target5': y_test[:, 4],
        'pred_target5': y_test_pred[:, 4]
    })
    predictions_path = os.path.join(args.output_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"测试集预测结果已保存到: {predictions_path}")
    
    print("模型训练完成!")


if __name__ == "__main__":
    main()