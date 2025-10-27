"""
编码方案对比：为什么选择one-hot编码
==================================

这个脚本演示了不同编码方案在分子表示中的优缺点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn

def demonstrate_encoding_methods():
    """
    演示不同编码方法的优缺点
    """
    print("=" * 60)
    print("分子特征编码方案对比")
    print("=" * 60)

    # 示例原子类型
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
    print(f"示例原子类型: {atom_types}")

    # 1. 标签编码 (Label Encoding) - 不推荐
    print("\n1. 标签编码 (不推荐):")
    print("-" * 30)
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(atom_types)

    for atom, label in zip(atom_types, label_encoded):
        print(f"{atom:3s} -> {label}")

    print("\n问题:")
    print("- 暗示数值关系: Br(6) > Cl(5) > F(4)")
    print("- 没有化学意义")
    print("- 距离关系错误: |Br-I| = |F-O|")

    # 2. One-hot编码 (推荐)
    print("\n2. One-hot编码 (推荐):")
    print("-" * 30)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(np.array(atom_types).reshape(-1, 1))

    for i, (atom, encoding) in enumerate(zip(atom_types, onehot_encoded)):
        encoding_str = '[' + ', '.join(f"{int(x)}" for x in encoding) + ']'
        print(f"{atom:3s} -> {encoding_str}")

    print("\n优势:")
    print("- 所有类型平等")
    print("- 无虚假数值关系")
    print("- 网络可学习独立特征")
    print("- 支持多标签情况")

    # 3. 原子属性编码 (替代方案)
    print("\n3. 原子属性编码 (替代方案):")
    print("-" * 30)

    # 原子周期表属性
    atomic_properties = {
        'C': [6, 4, 2.55, 2],   # 原子序数, 价电子, 电负性, 周期
        'N': [7, 5, 3.04, 2],
        'O': [8, 6, 3.44, 2],
        'S': [16, 6, 2.58, 3],
        'F': [9, 7, 3.98, 2],
        'Cl': [17, 7, 3.16, 3],
        'Br': [35, 7, 2.96, 4],
        'I': [53, 7, 2.66, 5]
    }

    print("原子 | 原子序数 | 价电子 | 电负性 | 周期")
    print("-" * 50)
    for atom, props in atomic_properties.items():
        print(f"{atom:3s} | {props[0]:8d} | {props[1]:6d} | {props[2]:6.2f} | {props[3]:4d}")

    print("\n优势:")
    print("- 信息密度高")
    print("- 有化学意义")
    print("- 数值关系合理")
    print("\n劣势:")
    print("- 需要领域知识")
    print("- 可能遗漏重要信息")
    print("- 数值范围差异大")

    return onehot_encoder

def neural_network_comparison():
    """
    比较不同编码在神经网络中的表现
    """
    print("\n" + "=" * 60)
    print("神经网络中的编码效果对比")
    print("=" * 60)

    # 模拟分子数据
    molecules = [
        ['C', 'C', 'O'],    # 乙醇
        ['C', 'N', 'C'],    # 乙胺
        ['C', 'C', 'C'],    # 丙烷
        ['O', 'C', 'O'],    # 二氧化碳
        ['N', 'C', 'N'],    # 氰化物
    ] * 100  # 复制100次

    # 模拟pIC50值
    pic50_values = np.array([
        2.5, 3.2, 1.8, 4.1, 2.9
    ] * 100) + np.random.normal(0, 0.1, 500)

    print(f"数据集大小: {len(molecules)} 个分子")
    print(f"目标变量: pIC50")

    # 编码方案1: 标签编码
    print("\n方案1: 标签编码")
    le = LabelEncoder()
    le.fit(['C', 'N', 'O'])

    encoded_label = []
    for mol in molecules:
        mol_encoded = le.transform(mol)
        encoded_label.append(mol_encoded.mean())  # 简单聚合

    X_label = torch.tensor(encoded_label, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(pic50_values, dtype=torch.float32).view(-1, 1)

    # 编码方案2: One-hot编码
    print("方案2: One-hot编码")
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit([['C'], ['N'], ['O']])

    encoded_onehot = []
    for mol in molecules:
        mol_encoded = ohe.transform(np.array(mol).reshape(-1, 1))
        mol_aggregated = mol_encoded.mean(axis=0)  # 简单聚合
        encoded_onehot.append(mol_aggregated)

    X_onehot = torch.tensor(encoded_onehot, dtype=torch.float32)

    # 简单线性回归比较
    class SimpleLinear(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.linear(x)

    # 训练和评估函数
    def train_and_evaluate(X, y, name):
        model = SimpleLinear(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(1000):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % 200 == 0:
                print(f"{name} - Epoch {epoch}: Loss = {loss.item():.4f}")

        return losses[-1], model

    # 训练和比较
    final_loss_label, model_label = train_and_evaluate(X_label, y, "标签编码")
    final_loss_onehot, model_onehot = train_and_evaluate(X_onehot, y, "One-hot编码")

    print(f"\n最终损失对比:")
    print(f"标签编码: {final_loss_label:.4f}")
    print(f"One-hot编码: {final_loss_onehot:.4f}")

    # 权重分析
    print(f"\n权重分析:")
    print(f"标签编码权重: {model_label.linear.weight.data.item():.4f}")
    print(f"One-hot编码权重: {model_onehot.linear.weight.data.numpy().flatten()}")

    return final_loss_label, final_loss_onehot

def feature_importance_analysis():
    """
    分析不同特征的重要性
    """
    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)

    # 模拟更复杂的特征
    features = {
        'onehot_carbon': [1, 0, 0],      # 碳原子one-hot
        'onehot_nitrogen': [0, 1, 0],    # 氮原子one-hot
        'onehot_oxygen': [0, 0, 1],     # 氧原子one-hot
        'atomic_number': [6, 7, 8],     # 原子序数
        'electronegativity': [2.55, 3.04, 3.44],  # 电负性
        'valence': [4, 5, 6]             # 价电子
    }

    print("特征类型 vs 信息容量:")
    print("-" * 40)

    df = pd.DataFrame(features, index=['C', 'N', 'O'])
    print(df.T)

    # 计算特征多样性
    print(f"\n特征分析:")
    for col in df.columns:
        unique_count = len(set(df[col]))
        if isinstance(df[col][0], list):
            # 对于列表特征，检查整个向量是否唯一
            unique_count = len(set(tuple(x) for x in df[col]))

        diversity = unique_count / len(df[col])
        print(f"{col:20s}: 多样性 = {diversity:.2f}")

    print(f"\n结论:")
    print("- One-hot编码提供完美的类型区分性 (多样性 = 1.0)")
    print("- 数值特征可能有相关性或冗余")
    print("- 混合使用可能是最佳策略")

def main():
    """
    主函数
    """
    # 编码方法演示
    encoder = demonstrate_encoding_methods()

    # 神经网络对比
    loss_label, loss_onehot = neural_network_comparison()

    # 特征重要性分析
    feature_importance_analysis()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. One-hot编码在分类特征中表现更好")
    print("2. 避免了虚假的数值关系")
    print("3. 神经网络学习更稳定")
    print("4. 特征解释性更强")
    print("\n最佳实践:")
    print("- 分类特征: 使用one-hot编码")
    print("- 数值特征: 标准化后直接使用")
    print("- 混合特征: 结合两种方法")

if __name__ == "__main__":
    main()