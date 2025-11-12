import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
from sklearn.model_selection import train_test_split
from data_loader import load_data, smiles_to_features, prepare_dataset
from ensemble_model import WeightedEnsembleModel, StackingEnsembleModel

def main():
    parser = argparse.ArgumentParser(description='训练集成学习模型')
    parser.add_argument('--data-path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--model-type', type=str, choices=['weighted', 'stacking'], default='weighted', 
                        help='模型类型: weighted (加权集成) 或 stacking (堆叠集成)')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = f"experiments_{args.model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    df = load_data(args.data_path)
    print(f"数据形状: {df.shape}")
    
    # 准备数据集
    print("准备数据集...")
    X, y, selector = prepare_dataset(df)
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标值形状: {y.shape}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 创建并训练模型
    if args.model_type == 'weighted':
        print("使用加权集成模型...")
        model = WeightedEnsembleModel()
    else:
        print("使用堆叠集成模型...")
        model = StackingEnsembleModel()
    
    print("开始训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成!")
    
    # 评估模型
    print("评估模型...")
    train_results = model.evaluate(X_train, y_train)
    test_results = model.evaluate(X_test, y_test)
    
    print("\n训练集评估结果:")
    print(f"  整体 R2: {train_results['overall']['R2']:.4f}")
    
    print("\n测试集评估结果:")
    print(f"  整体 R2: {test_results['overall']['R2']:.4f}")
    
    for target in ['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']:
        print(f"  {target}: R2 = {test_results[target]['R2']:.4f}")
    
    # 保存模型
    print("保存模型...")
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(model, model_path)
    
    # 保存特征选择器（如果有的话）
    if selector is not None:
        selector_path = os.path.join(output_dir, 'selector.pkl')
        joblib.dump(selector, selector_path)
    
    # 保存评估结果
    results = {
        'train': train_results,
        'test': test_results
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存测试集预测结果
    test_predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(test_predictions, columns=['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50'])
    predictions_path = os.path.join(output_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"模型已保存到: {model_path}")
    print(f"评估结果已保存到: {results_path}")
    print(f"测试集预测结果已保存到: {predictions_path}")

if __name__ == "__main__":
    main()