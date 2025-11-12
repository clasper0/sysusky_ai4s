import pandas as pd
import numpy as np
import joblib
import argparse
import os
from data_loader import smiles_to_features

def main():
    parser = argparse.ArgumentParser(description='使用训练好的集成模型进行预测')
    parser.add_argument('--model-dir', type=str, required=True, help='模型目录路径')
    parser.add_argument('--input', type=str, required=True, help='待预测数据路径 (CSV格式，需包含SMILES列)')
    parser.add_argument('--output', type=str, required=True, help='预测结果输出路径')
    
    args = parser.parse_args()
    
    # 加载模型
    print("加载模型...")
    model_path = os.path.join(args.model_dir, 'model.pkl')
    model = joblib.load(model_path)
    
    # 加载特征选择器（如果存在）
    selector_path = os.path.join(args.model_dir, 'selector.pkl')
    selector = None
    if os.path.exists(selector_path):
        selector = joblib.load(selector_path)
    
    # 加载待预测数据
    print("加载待预测数据...")
    df = pd.read_csv(args.input)
    
    # 检查是否有SMILES列
    if 'SMILES' not in df.columns:
        raise ValueError("输入数据必须包含'SMILES'列")
    
    smiles_list = df['SMILES'].tolist()
    
    # 提取特征
    print("提取分子特征...")
    X = smiles_to_features(smiles_list)
    
    # 应用特征选择器（如果存在）
    if selector is not None:
        X = selector.transform(X)
    
    # 进行预测
    print("进行预测...")
    predictions = model.predict(X)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(predictions, columns=[
        'target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50'
    ])
    
    # 添加SMILES列
    results_df.insert(0, 'SMILES', smiles_list)
    
    # 保存结果
    results_df.to_csv(args.output, index=False)
    print(f"预测结果已保存到: {args.output}")

if __name__ == "__main__":
    main()