"""
Molecule Visualization and Graph Structure Analysis
==================================================

This module provides visualization tools to understand how SMILES strings
are converted to graph structures for GCN training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import networkx as nx
from smiles_to_graph import SMILESToGraph
import warnings
# warnings.filterwarnings('ignore')


class MoleculeVisualizer:
    """
    可视化分子结构和图表示的工具类
    """

    def __init__(self):
        """初始化可视化工具"""
        self.converter = SMILESToGraph()

    def visualize_smiles_structure(self, smiles: str, show_features: bool = True):
        """
        可视化SMILES的分子结构和图表示

        Args:
            smiles: SMILES字符串
            show_features: 是否显示原子和边特征
        """
        print(f"SMILES: {smiles}")
        print("-" * 50)

        # 解析分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("无法解析SMILES字符串")
            return

        mol = Chem.AddHs(mol)

        # 显示基本信息
        print(f"原子数: {mol.GetNumAtoms()}")
        print(f"键数: {mol.GetNumBonds()}")
        print(f"分子量: {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}")

        # 获取原子信息
        print("\n原子信息:")
        print("索引 | 符号 | 度数 | 形式电荷 | 芳香性 | 杂化类型")
        print("-" * 60)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            degree = atom.GetDegree()
            charge = atom.GetFormalCharge()
            aromatic = atom.GetIsAromatic()
            hybrid = atom.GetHybridization().name
            print(f"{idx:5d} | {symbol:4s} | {degree:5d} | {charge:9d} | {aromatic:6s} | {hybrid}")

        # 获取键信息
        print("\n键信息:")
        print("原子1 | 原子2 | 类型 | 芳香性 | 立体化学")
        print("-" * 50)
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            btype = bond.GetBondType().name
            aromatic = bond.GetIsAromatic()
            stereo = bond.GetStereo().name
            print(f"{atom1:5d} | {atom2:5d} | {btype:4s} | {aromatic:6s} | {stereo}")

        # 创建NetworkX图用于可视化
        G = nx.Graph()

        # 添加节点
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            G.add_node(idx, label=f"{symbol}_{idx}")

        # 添加边
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType().name
            G.add_edge(atom1, atom2, label=bond_type)

        # 绘制图结构
        plt.figure(figsize=(12, 6))

        # 子图1: 分子结构图
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("分子图结构")
        plt.axis('off')

        # 子图2: 分子图像
        plt.subplot(1, 2, 2)
        img = Draw.MolToImage(mol, size=(300, 300))
        plt.imshow(img)
        plt.title("分子结构")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 如果需要显示特征
        if show_features:
            self.show_features_analysis(smiles)

    def show_features_analysis(self, smiles: str):
        """
        显示分子特征分析

        Args:
            smiles: SMILES字符串
        """
        print("\n特征分析:")
        print("-" * 40)

        # 转换为图数据
        graph = self.converter.smiles_to_graph(smiles)
        if graph is None:
            return

        # 节点特征统计
        print(f"节点特征维度: {graph.x.shape[1]}")
        print(f"前3个节点的完整特征向量:")
        for i in range(min(3, graph.num_nodes)):
            print(f"节点 {i}: {graph.x[i].numpy()}")

        # 特征类型分解
        atom_types = len(self.converter.atom_to_idx)
        hybrid_types = len(self.converter.hybridization_to_idx)

        print(f"\n特征类型分解:")
        print(f"原子类型 (one-hot, {atom_types}维): [0:{atom_types}]")
        print(f"形式电荷: [{atom_types}]")
        print(f"度数: [{atom_types+1}]")
        print(f"氢键数: [{atom_types+2}]")
        print(f"价电子数: [{atom_types+3}]")
        print(f"芳香性: [{atom_types+4}]")
        print(f"杂化类型 (one-hot, {hybrid_types}维): [{atom_types+5}:{atom_types+5+hybrid_types}]")
        print(f"是否在环中: [{atom_types+5+hybrid_types}]")
        print(f"标准化原子质量: [{atom_types+6+hybrid_types}]")

        # 边特征统计
        print(f"\n边特征维度: {graph.edge_attr.shape[1]}")
        bond_types = len(self.converter.bond_type_to_idx)
        stereo_types = len(self.converter.stereo_to_idx)

        print(f"边特征类型分解:")
        print(f"化学键类型 (one-hot, {bond_types}维): [0:{bond_types}]")
        print(f"立体化学 (one-hot, {stereo_types}维): [{bond_types}:{bond_types+stereo_types}]")
        print(f"是否在环中: [{bond_types+stereo_types}]")
        print(f"共轭性: [{bond_types+stereo_types+1}]")

    def analyze_dataset_graphs(self, graphs, df=None):
        """
        分析数据集中所有图的统计信息

        Args:
            graphs: 图数据列表
            df: 包含SMILES的数据框
        """
        if not graphs:
            print("没有图数据可供分析")
            return

        print("=" * 60)
        print("数据集图结构统计分析")
        print("=" * 60)

        # 基本统计
        node_counts = [g.num_nodes for g in graphs]
        edge_counts = [g.edge_index.shape[1] // 2 for g in graphs]

        print(f"分子总数: {len(graphs)}")
        print(f"平均节点数: {np.mean(node_counts):.2f} ± {np.std(node_counts):.2f}")
        print(f"节点数范围: {np.min(node_counts)} - {np.max(node_counts)}")
        print(f"平均边数: {np.mean(edge_counts):.2f} ± {np.std(edge_counts):.2f}")
        print(f"边数范围: {np.min(edge_counts)} - {np.max(edge_counts)}")

        # 可视化分布
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 节点数分布
        axes[0, 0].hist(node_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('节点数分布')
        axes[0, 0].set_xlabel('节点数')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)

        # 边数分布
        axes[0, 1].hist(edge_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('边数分布')
        axes[0, 1].set_xlabel('边数')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)

        # 节点数 vs 边数散点图
        axes[1, 0].scatter(node_counts, edge_counts, alpha=0.6, color='coral')
        axes[1, 0].set_title('节点数 vs 边数')
        axes[1, 0].set_xlabel('节点数')
        axes[1, 0].set_ylabel('边数')
        axes[1, 0].grid(True, alpha=0.3)

        # 度分布
        all_degrees = []
        for g in graphs:
            # 计算每个节点的度数
            degrees = torch.zeros(g.num_nodes)
            for edge in g.edge_index.t():
                degrees[edge[0]] += 1
            all_degrees.extend(degrees.numpy().tolist())

        axes[1, 1].hist(all_degrees, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('节点度分布')
        axes[1, 1].set_xlabel('度数')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 如果有pIC50数据，显示与分子大小的关系
        if df is not None and 'pIC50' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(node_counts[:len(df)], df['pIC50'][:len(graphs)], alpha=0.6)
            plt.xlabel('分子大小 (节点数)')
            plt.ylabel('pIC50')
            plt.title('分子大小 vs pIC50')
            plt.grid(True, alpha=0.3)
            plt.show()


def main():
    """
    主函数：演示分子可视化
    """
    print("分子图结构可视化演示")
    print("=" * 60)

    # 创建可视化工具
    visualizer = MoleculeVisualizer()

    # 示例分子
    examples = [
        "CC(=O)c1ccc2nc(-c3ccccc3)n(O)c2c1",  # 复杂分子
        "CCO",  # 乙醇
        "c1ccccc1",  # 苯
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # 布洛芬
    ]

    # 可视化每个例子
    for i, smiles in enumerate(examples):
        print(f"\n示例 {i+1}:")
        visualizer.visualize_smiles_structure(smiles)

    # 分析实际数据集
    try:
        from smiles_to_graph import load_and_convert_data
        graphs, df = load_and_convert_data()

        if graphs:
            visualizer.analyze_dataset_graphs(graphs, df)

            # 显示几个复杂分子的详细分析
            print("\n复杂分子详细分析:")
            complex_molecules = []
            for i, g in enumerate(graphs):
                if g.num_nodes > 15:  # 选择较大的分子
                    if df is not None and i < len(df):
                        smiles = df.iloc[i]['SMILES']
                        complex_molecules.append(smiles)
                    break

            for smiles in complex_molecules[:3]:  # 最多显示3个
                visualizer.visualize_smiles_structure(smiles)

    except Exception as e:
        print(f"分析数据集时出错: {str(e)}")


if __name__ == "__main__":
    main()