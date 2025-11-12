"""
训练直推学习模型
================

该脚本训练基于图神经网络的直推学习模型，
利用半监督学习方法提高小样本场景下的预测能力。
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# 添加上级目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from smiles_to_graph import SMILESToGraph
from transductive_model import TransductiveGCN, TransductiveLearner
import warnings
warnings.filterwarnings('ignore')


def load_activity_data(activity_file):
    """
    加载活性数据

    Args:
        activity_file: 活性数据文件路径

    Returns:
        DataFrame格式的活性数据
    """
    activity_df = pd.read_csv(activity_file)
    return activity_df


def load_smiles_data(smiles_file):
    """
    加载SMILES数据

    Args:
        smiles_file: SMILES数据文件路径

    Returns:
        DataFrame格式的SMILES数据
    """
    smiles_df = pd.read_csv(smiles_file, sep=',', header=None, 
                           names=['Molecule_ID', 'SMILES'])
    return smiles_df


def prepare_graph_data(activity_df, smiles_df, target_columns):
    """
    准备图数据

    Args:
        activity_df: 活性数据
        smiles_df: SMILES数据
        target_columns: 目标列名列表

    Returns:
        图数据列表
    """
    # 重塑活性数据
    pivot_df = activity_df.pivot(index='molecule_id', columns='target_id', values='pIC50')
    
    # 重置索引
    pivot_df.reset_index(inplace=True)
    
    # 重命名列
    pivot_df.rename(columns={'molecule_id': 'Molecule_ID'}, inplace=True)
    
    # 合并数据
    merged_df = pd.merge(smiles_df, pivot_df, on='Molecule_ID')
    
    # 初始化转换器
    converter = SMILESToGraph()
    
    graph_data_list = []
    failed_count = 0
    
    for idx, row in merged_df.iterrows():
        smiles = row['SMILES']
        mol_id = row['Molecule_ID']
        
        # 将SMILES转换为图
        graph_data = converter.smiles_to_graph(smiles)
        if graph_data is not None:
            # 添加目标值
            target_values = row[target_columns].values.astype(np.float32)
            graph_data.y = torch.tensor(target_values, dtype=torch.float).unsqueeze(0)
            graph_data.mol_id = mol_id
            graph_data_list.append(graph_data)
        else:
            failed_count += 1
    
    print(f"成功转换 {len(graph_data_list)} 个分子为图结构")
    if failed_count > 0:
        print(f"转换失败 {failed_count} 个分子")
    
    return graph_data_list


def split_data(graph_data_list, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    划分数据集

    Args:
        graph_data_list: 图数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        训练集、验证集、测试集
    """
    total_size = len(graph_data_list)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = graph_data_list[:train_size]
    val_data = graph_data_list[train_size:train_size+val_size]
    test_data = graph_data_list[train_size+val_size:]
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description='训练直推学习模型')
    parser.add_argument('--data-dir', type=str, default='../data', help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default='experiments', help='输出目录')
    parser.add_argument('--experiment-name', type=str, default=None, help='实验名称')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--hidden-dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=3, help='GCN层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--use-sage', action='store_true', help='使用GraphSAGE替代GCN')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/cuda)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='测试集比例')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用GPU进行训练")
    else:
        device = torch.device('cpu')
        print("使用CPU进行训练")
    
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
    
    print(f"开始训练直推学习模型，实验名称: {args.experiment_name}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"GCN层数: {args.num_layers}")
    print(f"Dropout概率: {args.dropout}")
    print(f"学习率: {args.lr}")
    print(f"使用权重衰减: {args.weight_decay}")
    print(f"使用GraphSAGE: {args.use_sage}")
    
    # 文件路径
    activity_file = os.path.join(args.data_dir, 'activity_train.csv')
    smiles_file = os.path.join(args.data_dir, 'molecule.smi')
    
    # 检查文件是否存在
    if not os.path.exists(activity_file):
        raise FileNotFoundError(f"活性数据文件不存在: {activity_file}")
    if not os.path.exists(smiles_file):
        raise FileNotFoundError(f"SMILES文件不存在: {smiles_file}")
    
    # 目标列名
    target_columns = ['TAR_001', 'TAR_002', 'TAR_003', 'TAR_004', 'TAR_005']
    
    # 加载数据
    print("加载活性数据...")
    activity_df = load_activity_data(activity_file)
    
    print("加载SMILES数据...")
    smiles_df = load_smiles_data(smiles_file)
    
    # 准备图数据
    print("准备图数据...")
    graph_data_list = prepare_graph_data(activity_df, smiles_df, target_columns)
    
    if len(graph_data_list) == 0:
        raise ValueError("没有成功转换的图数据，请检查数据格式")
    
    # 划分数据集
    train_data, val_data, test_data = split_data(
        graph_data_list, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    # 初始化模型
    model = TransductiveGCN(
        input_dim=36,  # 原子特征维度
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        target_dim=len(target_columns),
        use_sage=args.use_sage
    )
    
    # 初始化训练器
    learner = TransductiveLearner(model, device)
    
    # 准备直推学习数据
    learner.prepare_transductive_data(train_data, val_data, test_data)
    
    # 训练模型
    learner.train(epochs=args.epochs, patience=20)
    
    # 在测试集上评估
    test_pred, test_true, test_metrics = learner.predict()
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'true_tar_001': test_true[:, 0],
        'pred_tar_001': test_pred[:, 0],
        'true_tar_002': test_true[:, 1],
        'pred_tar_002': test_pred[:, 1],
        'true_tar_003': test_true[:, 2],
        'pred_tar_003': test_pred[:, 2],
        'true_tar_004': test_true[:, 3],
        'pred_tar_004': test_pred[:, 3],
        'true_tar_005': test_true[:, 4],
        'pred_tar_005': test_pred[:, 4]
    })
    
    predictions_path = os.path.join(experiment_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"测试集预测结果已保存到: {predictions_path}")
    
    # 保存模型
    model_path = os.path.join(experiment_dir, 'transductive_model.pth')
    learner.save_model(model_path)
    
    # 保存评估结果
    results = {
        'test_metrics': test_metrics
    }
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"评估结果已保存到: {results_path}")
    
    print("直推学习模型训练完成!")


if __name__ == "__main__":
    main()