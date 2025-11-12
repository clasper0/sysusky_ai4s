"""
直推学习模型预测脚本
====================

该脚本使用训练好的直推学习模型进行预测。
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

# 添加上级目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from smiles_to_graph import SMILESToGraph
from transductive_model import TransductiveGCN, TransductiveLearner
import warnings
warnings.filterwarnings('ignore')


def load_smiles_data(smiles_file):
    """
    加载SMILES数据

    Args:
        smiles_file: SMILES数据文件路径

    Returns:
        DataFrame格式的SMILES数据
    """
    smiles_df = pd.read_csv(smiles_file, sep=r'\s+', header=None, 
                           names=['SMILES', 'Molecule_ID'])
    return smiles_df


def prepare_prediction_data(smiles_df):
    """
    准备预测数据

    Args:
        smiles_df: SMILES数据

    Returns:
        图数据列表
    """
    # 初始化转换器
    converter = SMILESToGraph()
    
    graph_data_list = []
    failed_count = 0
    
    for idx, row in smiles_df.iterrows():
        smiles = row['SMILES']
        mol_id = row['Molecule_ID']
        
        # 将SMILES转换为图
        graph_data = converter.smiles_to_graph(smiles)
        if graph_data is not None:
            graph_data.mol_id = mol_id
            graph_data_list.append(graph_data)
        else:
            failed_count += 1
    
    print(f"成功转换 {len(graph_data_list)} 个分子为图结构")
    if failed_count > 0:
        print(f"转换失败 {failed_count} 个分子")
    
    return graph_data_list


def main():
    parser = argparse.ArgumentParser(description='使用直推学习模型进行预测')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入SMILES文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出预测结果文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用GPU进行预测")
    else:
        device = torch.device('cpu')
        print("使用CPU进行预测")
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    # 加载SMILES数据
    print("加载SMILES数据...")
    smiles_df = load_smiles_data(args.input)
    
    # 准备预测数据
    print("准备预测数据...")
    graph_data_list = prepare_prediction_data(smiles_df)
    
    if len(graph_data_list) == 0:
        raise ValueError("没有成功转换的图数据，请检查数据格式")
    
    # 初始化模型
    model = TransductiveGCN(
        input_dim=36,     # 原子特征维度
        hidden_dim=128,   # 需要与训练时保持一致
        num_layers=3,     # 需要与训练时保持一致
        dropout=0.2,      # 需要与训练时保持一致
        target_dim=5      # 需要与训练时保持一致
    )
    
    # 初始化训练器
    learner = TransductiveLearner(model, device)
    
    # 加载模型
    learner.load_model(args.model_path)
    
    # 准备数据（直推学习方式）
    batch = Batch.from_data_list(graph_data_list).to(device)
    
    # 进行预测
    print("进行预测...")
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        predictions = predictions.cpu().numpy()
    
    # 保存预测结果
    print("保存预测结果...")
    results_df = pd.DataFrame({
        'Molecule_ID': [data.mol_id for data in graph_data_list],
        'TAR_001': predictions[:, 0],
        'TAR_002': predictions[:, 1],
        'TAR_003': predictions[:, 2],
        'TAR_004': predictions[:, 3],
        'TAR_005': predictions[:, 4]
    })
    
    results_df.to_csv(args.output, index=False)
    print(f"预测结果已保存到: {args.output}")
    
    print("预测完成!")


if __name__ == "__main__":
    main()