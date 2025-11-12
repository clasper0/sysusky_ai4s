"""
生成示例数据：包含5个靶点pIC50值和分子描述符
==================================================

该脚本生成用于混合模型训练的示例数据，包括：
1. 分子SMILES
2. 5个靶点的pIC50值
3. 分子描述符特征
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import random


def calculate_molecular_descriptors(smiles: str) -> dict:
    """
    计算分子描述符
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        dict: 分子描述符字典
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        descriptors = {
            'MW': Descriptors.MolWt(mol),  # 分子量
            'LogP': Crippen.MolLogP(mol),  # LogP
            'NumHDonors': Lipinski.NumHDonors(mol),  # 氢键供体数
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),  # 氢键受体数
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),  # 可旋转键数
            'NumRings': Lipinski.RingCount(mol),  # 环数
            'TPSA': Descriptors.TPSA(mol),  # 极性表面积
            'NumAtoms': mol.GetNumAtoms(),  # 原子数
            'FractionCSP3': Lipinski.FractionCSP3(mol),  # sp3碳比例
            'HeavyAtomCount': Lipinski.HeavyAtomCount(mol)  # 重原子数
        }
        
        return descriptors
    except:
        return None


def generate_example_data(num_molecules: int = 1000, output_file: str = "data/candidate_hybrid.csv"):
    """
    生成示例数据
    
    Args:
        num_molecules: 分子数量
        output_file: 输出文件路径
    """
    # 示例SMILES列表
    example_smiles = [
        "CC1=CC=CC=C1",  # 甲苯
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "CC(C)C",  # 异丁烷
        "C1CCCCC1",  # 环己烷
        "CCN(CC)CC",  # 三乙胺
        "C1=CC=C(C=C1)O",  # 苯酚
        "CC(=O)N",  # 乙酰胺
        "C#C",  # 乙炔
        "C1CC1",  # 环丙烷
        "CCOC(C)=O",  # 乙酸乙酯
        "CN(C)C",  # 三甲胺
        "C1=CC=CC=C1C(=O)O",  # 苯甲酸
        "C(CO)N",  # 氨基乙醇
        "C1=CC=C2C(=C1)C=CC=C2",  # 萘
        "CC(C)(C)O",  # 叔丁醇
        "C1CCC(CC1)N",  # 环己胺
        "CC(=O)OC",  # 乙酸甲酯
        "C1=CC(=CC=C1N)N",  # 苯二胺
        "C1=CC=C(C=C1)Cl"  # 氯苯
    ]
    
    # 生成数据
    data = []
    
    for i in range(num_molecules):
        # 随机选择一个SMILES
        smiles = random.choice(example_smiles)
        
        # 计算分子描述符
        descriptors = calculate_molecular_descriptors(smiles)
        if descriptors is None:
            continue
            
        # 生成5个靶点的pIC50值（基于描述符和随机因素）
        # 模拟不同靶点对分子不同特性的偏好
        base_activity = 5.0 + np.random.normal(0, 1)  # 基础活性
        
        target1_pic50 = base_activity + descriptors['LogP'] * 0.2 + np.random.normal(0, 0.5)  # 疏水性相关
        target2_pic50 = base_activity + descriptors['NumHDonors'] * 0.3 + np.random.normal(0, 0.5)  # 氢键供体相关
        target3_pic50 = base_activity + descriptors['NumHAcceptors'] * 0.25 + np.random.normal(0, 0.5)  # 氢键受体相关
        target4_pic50 = base_activity - descriptors['MW'] * 0.01 + np.random.normal(0, 0.5)  # 分子量相关
        target5_pic50 = base_activity + descriptors['NumRings'] * 0.5 + np.random.normal(0, 0.5)  # 环结构相关
        
        # 创建数据记录
        record = {
            'SMILES': smiles,
            'target1_pIC50': target1_pic50,
            'target2_pIC50': target2_pic50,
            'target3_pIC50': target3_pic50,
            'target4_pIC50': target4_pic50,
            'target5_pIC50': target5_pic50
        }
        
        # 添加描述符
        record.update(descriptors)
        
        data.append(record)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到文件
    df.to_csv(output_file, index=False)
    print(f"已生成 {len(df)} 个分子的示例数据并保存到 {output_file}")
    
    # 显示数据统计信息
    print("\n数据统计信息:")
    print(f"  分子数量: {len(df)}")
    print(f"  特征列数量: {len(df.columns)}")
    print(f"  靶点数量: 5")
    print(f"  描述符数量: {len(descriptors)}")
    
    # 显示前几行
    print("\n前5行数据:")
    print(df.head())


if __name__ == "__main__":
    generate_example_data()