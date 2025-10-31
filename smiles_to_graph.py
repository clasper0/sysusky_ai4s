"""
SMILES to Graph Structure Conversion for GCN-based pIC50 Prediction
==================================================================

This module demonstrates how to convert SMILES strings to graph structures
suitable for Graph Convolutional Networks (GCNs) to predict molecular pIC50.

Key Components:
1. SMILES parsing using RDKit
2. Node feature extraction
3. Edge construction and feature extraction
4. Graph data structure for PyTorch Geometric
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import warnings
with warnings.catch_warnings():
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import os


class SMILESToGraph:
    """
    Convert SMILES strings to graph structures for GCN.

    This class handles the conversion of molecular SMILES representations
    to graph structures with node and edge features suitable for
    Graph Convolutional Networks.
    """

    def __init__(self):
        """Initialize the converter with atom and bond feature mappings."""

        # 定义原子类型到索引的映射
        self.atom_to_idx = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7,
            'P': 8, 'B': 9, 'Si': 10, 'Se': 11, 'Zn': 12, 'Mg': 13, 'Ca': 14,
            'Fe': 15, 'Cu': 16, 'Al': 17, 'Ti': 18, 'Cr': 19, 'Mn': 20,
            'H': 21, 'Other': 22
        }

        # 定义杂化类型到索引的映射
        self.hybridization_to_idx = {
            'SP': 0, 'SP2': 1, 'SP3': 2, 'SP3D': 3, 'SP3D2': 4, 'OTHER': 5
        }

        # 定义化学键类型到索引的映射
        self.bond_type_to_idx = {
            'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3, 'OTHER': 4
        }

        # 定义立体化学到索引的映射
        self.stereo_to_idx = {
            'STEREONONE': 0, 'STEREOANY': 1, 'STEREOZ': 2,
            'STEREOE': 3, 'STEREOCIS': 4, 'STEREOTRANS': 5
        }

    def get_atom_features(self, atom: rdkit.Chem.rdchem.Atom) -> np.ndarray:
        """
        Extract features from a single atom.

        Args:
            atom: RDKit atom object

        Returns:
            np.ndarray: Feature vector for the atom
        """
        features = []

        # 1. 原子类型 (one-hot编码)
        atom_symbol = atom.GetSymbol()
        atom_type_idx = self.atom_to_idx.get(atom_symbol, self.atom_to_idx['Other'])
        features.extend([1 if i == atom_type_idx else 0 for i in range(len(self.atom_to_idx))])

        # 2. 形式电荷
        features.append(atom.GetFormalCharge())

        # 3. 度数 (连接的键数)
        features.append(atom.GetDegree())

        # 4. 氢键数
        features.append(atom.GetTotalNumHs())

        # 5. 价电子数
        features.append(atom.GetValence(rdkit.Chem.ValenceType.EXPLICIT))

        # 6. 芳香性
        features.append(1 if atom.GetIsAromatic() else 0)

        # 7. 杂化类型 (one-hot编码)
        hybridization = atom.GetHybridization().name
        hybrid_idx = self.hybridization_to_idx.get(hybridization, self.hybridization_to_idx['OTHER'])
        features.extend([1 if i == hybrid_idx else 0 for i in range(len(self.hybridization_to_idx))])

        # 8. 是否在环中
        features.append(1 if atom.IsInRing() else 0)

        # 9. 原子质量 (标准化)
        atomic_mass = atom.GetMass()
        features.append(atomic_mass / 100.0)  # 标准化到[0,1]范围

        return np.array(features, dtype=np.float32)

    def get_bond_features(self, bond: rdkit.Chem.rdchem.Bond) -> np.ndarray:
        """
        Extract features from a single bond.

        Args:
            bond: RDKit bond object

        Returns:
            np.ndarray: Feature vector for the bond
        """
        features = []

        # 1. 化学键类型 (one-hot编码)
        bond_type = bond.GetBondType().name
        bond_type_idx = self.bond_type_to_idx.get(bond_type, self.bond_type_to_idx['OTHER'])
        features.extend([1 if i == bond_type_idx else 0 for i in range(len(self.bond_type_to_idx))])

        # 2. 立体化学 (one-hot编码)
        stereo = bond.GetStereo().name
        stereo_idx = self.stereo_to_idx.get(stereo, self.stereo_to_idx['STEREONONE'])
        features.extend([1 if i == stereo_idx else 0 for i in range(len(self.stereo_to_idx))])

        # 3. 是否在环中
        features.append(1 if bond.IsInRing() else 0)

        # 4. 共轭性
        features.append(1 if bond.GetIsConjugated() else 0)

        return np.array(features, dtype=np.float32)

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES string to PyTorch Geometric graph data.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Data: PyTorch Geometric Data object or None if conversion fails
        """
        try:
            # 解析SMILES字符串
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles}")
                return None

            # 添加氢原子
            mol = Chem.AddHs(mol)

            # 获取原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self.get_atom_features(atom))

            # 获取边索引和边特征
            edge_indices = []
            edge_features = []

            for bond in mol.GetBonds():
                # 获取键连接的原子索引
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()

                # 添加边（双向）
                edge_indices.append([atom1_idx, atom2_idx])
                edge_indices.append([atom2_idx, atom1_idx])

                # 添加边特征（双向）
                bond_features = self.get_bond_features(bond)
                edge_features.append(bond_features)
                edge_features.append(bond_features)

            # 转换为PyTorch张量 (优化性能)
            x = torch.tensor(np.array(atom_features), dtype=torch.float32)
            edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)

            # 创建PyTorch Geometric Data对象
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles,
                num_nodes=mol.GetNumAtoms()
            )

            return data

        except Exception as e:
            print(f"Error converting SMILES {smiles}: {str(e)}")
            return None

    def batch_smiles_to_graphs(self, smiles_list: List[str]) -> List[Data]:
        """
        Convert a batch of SMILES strings to graph data objects.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List[Data]: List of PyTorch Geometric Data objects
        """
        graphs = []
        for i, smiles in enumerate(smiles_list):
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
            else:
                print(f"Failed to convert SMILES at index {i}: {smiles}")

        return graphs


def demonstrate_conversion():
    """
    演示SMILES到图结构的转换过程
    """
    print("=" * 60)
    print("SMILES到图结构转换演示")
    print("=" * 60)

    # 初始化转换器
    converter = SMILESToGraph()

    # 示例SMILES字符串
    example_smiles = [
        "CC(=O)c1ccc2nc(-c3ccccc3)n(O)c2c1",  # 来自候选分子
        "Cc1cccc(C2CC(N[C@@H](C)C(=O)NCc3ccco3)C2)c1",
        "O=S(=O)(c1ccc(NC(=S)N2CCOCC2)cc1)N1CCCCC1",
        "CCO",  # 乙醇 - 简单例子
        "c1ccccc1"  # 苯 - 另一个简单例子
    ]

    for i, smiles in enumerate(example_smiles):
        print(f"\n示例 {i+1}: {smiles}")
        print("-" * 40)

        # 转换为图
        graph = converter.smiles_to_graph(smiles)

        if graph is not None:
            print(f"转换成功!")
            print(f"节点数: {graph.num_nodes}")
            print(f"边数: {graph.edge_index.shape[1] // 2}")  # 除以2因为边是双向的
            print(f"节点特征维度: {graph.x.shape[1]}")
            print(f"边特征维度: {graph.edge_attr.shape[1]}")

            # 显示前几个原子的特征
            print(f"前3个节点的特征向量:")
            for j in range(min(3, graph.num_nodes)):
                print(f"  节点{j}: {graph.x[j][:10].tolist()}...")  # 只显示前10个特征

        else:
            print("转换失败!")


def load_and_convert_data():
    """
    加载并转换实际数据集中的SMILES
    """
    print("\n" + "=" * 60)
    print("加载并转换数据集")
    print("=" * 60)

    try:
        # 切换工作路径
        work_dir = "/home/clasper/2509/sysusky_ai4s"
        os.chdir(work_dir)

        # 读取候选分子数据
        candidate_df = pd.read_csv('data/molecule.smi', sep = ",", header = None, names = ["id", "SMILES"])
        print(f"候选分子数据形状: {candidate_df.shape}")
        print(f"列名: {candidate_df.columns.tolist()}")

        # 转换SMILES
        converter = SMILESToGraph()
        smiles_list = candidate_df['SMILES'].tolist()

        print(f"\n开始转换{len(smiles_list)}个SMILES字符串...")
        graphs = converter.batch_smiles_to_graphs(smiles_list)

        print(f"成功转换{len(graphs)}个分子图")

        # 显示统计信息
        if graphs:
            node_counts = [g.num_nodes for g in graphs]
            edge_counts = [g.edge_index.shape[1] // 2 for g in graphs]

            print(f"\n图结构统计:")
            print(f"平均节点数: {np.mean(node_counts):.2f}")
            print(f"最大节点数: {np.max(node_counts)}")
            print(f"最小节点数: {np.min(node_counts)}")
            print(f"平均边数: {np.mean(edge_counts):.2f}")
            print(f"最大边数: {np.max(edge_counts)}")
            print(f"最小边数: {np.min(edge_counts)}")

        return graphs, candidate_df

    except FileNotFoundError:
        print("未找到数据文件，请确保data/candidate.csv存在")
        return [], None


if __name__ == "__main__":
    # 运行演示
    # demonstrate_conversion()

    # 加载并转换实际数据
    graphs, df = load_and_convert_data()

    print("\n" + "=" * 60)
    print("SMILES到图结构转换完成!")
    print("=" * 60)
    print("\n下一步:")
    print("1. 设计图卷积网络架构")
    print("2. 创建数据加载器")
    print("3. 训练GCN模型")
    print("4. 评估模型性能")