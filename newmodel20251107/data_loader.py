"""
数据加载器
==========

该模块用于加载和预处理分子数据，提取RDKit特征和标签。
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, Crippen
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_candidate_data(file_path):
    """
    加载候选分子数据

    Args:
        file_path: 数据文件路径

    Returns:
        SMILES列表、特征矩阵、标签矩阵
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 提取SMILES和标签
    smiles_list = df['SMILES'].tolist()
    labels = df[['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']].values
    
    return smiles_list, labels, df


def extract_rdkit_features(smiles_list, fingerprint_type='morgan', n_bits=2048):
    """
    提取RDKit特征

    Args:
        smiles_list: SMILES列表
        fingerprint_type: 指纹类型 ('morgan', 'rdkit')
        n_bits: 指纹位数

    Returns:
        特征矩阵
    """
    features = []
    
    for smiles in smiles_list:
        try:
            # 转换为分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # 如果转换失败，添加零向量
                features.append(np.zeros(n_bits + 25))  # 指纹+描述符
                continue
            
            # 生成指纹
            if fingerprint_type == 'morgan':
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            elif fingerprint_type == 'rdkit':
                fingerprint = Chem.RDKFingerprint(mol, fpSize=n_bits)
            else:
                raise ValueError(f"不支持的指纹类型: {fingerprint_type}")
            
            fingerprint_array = np.array(fingerprint)
            
            # 提取分子描述符 (25个)
            descriptors = [
                Descriptors.MolWt(mol),           # 分子量
                Descriptors.MolLogP(mol),         # LogP (疏水性)
                Descriptors.TPSA(mol),            # 极性表面积
                Descriptors.NumHDonors(mol),      # 氢键供体数
                Descriptors.NumHAcceptors(mol),   # 氢键受体数
                Descriptors.NumRotatableBonds(mol),  # 可旋转键数
                Descriptors.NumAromaticRings(mol),   # 芳香环数
                Descriptors.NumAliphaticRings(mol),  # 脂肪环数
                Descriptors.NHOHCount(mol),       # NH和OH基团数
                Descriptors.NOCount(mol),         # 氮和氧原子数
                Descriptors.RingCount(mol),       # 环总数
                QED.qed(mol),                     # 药物相似性指数
                Crippen.MolMR(mol),               # 摩尔折射率
            ]
            
            # Lipinski规则相关描述符
            descriptors.extend([
                Lipinski.NumHDonors(mol),         # 氢键供体数 (Lipinski)
                Lipinski.NumHAcceptors(mol),      # 氢键受体数 (Lipinski)
                Lipinski.NumRotatableBonds(mol),  # 可旋转键数 (Lipinski)
                Lipinski.NumAliphaticCarbocycles(mol),  # 脂肪族碳环数
                Lipinski.NumAliphaticHeterocycles(mol), # 脂肪族杂环数
                Lipinski.NumAliphaticRings(mol),        # 脂肪环数 (Lipinski)
                Lipinski.NumAromaticCarbocycles(mol),   # 芳香族碳环数
                Lipinski.NumAromaticHeterocycles(mol),  # 芳香族杂环数
                Lipinski.NumSaturatedCarbocycles(mol),  # 饱和碳环数
                Lipinski.NumSaturatedHeterocycles(mol), # 饱和杂环数
                Lipinski.NumSaturatedRings(mol),        # 饱和环数
            ])
            
            descriptors_array = np.array(descriptors)
            
            # 拼接指纹和描述符
            feature_vector = np.concatenate([fingerprint_array, descriptors_array])
            features.append(feature_vector)
            
        except Exception as e:
            # 如果出现任何错误，添加零向量
            print(f"处理分子 {smiles} 时出错: {str(e)}")
            features.append(np.zeros(n_bits + 25))  # 指纹+描述符
    
    return np.array(features)


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    划分数据集

    Args:
        X: 特征矩阵
        y: 标签矩阵
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子

    Returns:
        训练集、验证集、测试集
    """
    # 首先划分训练集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 再从训练集中划分出验证集
    val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test