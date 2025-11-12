import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, Lipinski
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    加载数据
    
    Parameters:
    file_path (str): 数据文件路径
    
    Returns:
    DataFrame: 包含SMILES和目标值的数据框
    """
    df = pd.read_csv(file_path)
    return df

def smiles_to_features(smiles_list):
    """
    从SMILES列表中提取分子特征
    
    Parameters:
    smiles_list (list): SMILES字符串列表
    
    Returns:
    np.array: 分子特征数组
    """
    features = []
    
    # 初始化Morgan指纹生成器
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 如果SMILES无效，则添加零向量
            fingerprint = np.zeros(2048)
            descriptors = np.zeros(25)
        else:
            # 生成Morgan指纹
            fingerprint = np.array(morgan_generator.GetFingerprint(mol))
            
            # 计算RDKit分子描述符
            descriptors = np.array([
                Descriptors.MolWt(mol),                    # 分子量
                Descriptors.MolLogP(mol),                  # LogP
                Descriptors.NumHDonors(mol),               # 氢键供体数
                Descriptors.NumHAcceptors(mol),            # 氢键受体数
                Descriptors.NumRotatableBonds(mol),        # 可旋转键数
                Lipinski.RingCount(mol),                   # 环数
                Descriptors.TPSA(mol),                     # 极性表面积
                mol.GetNumAtoms(),                         # 原子数
                Descriptors.FractionCSP3(mol),             # CSP3比例
                Descriptors.HeavyAtomCount(mol),           # 重原子数
                Lipinski.NHOHCount(mol),                   # NH和OH基团数
                Lipinski.NOCount(mol),                     # 氮和氧原子数
                Lipinski.NumAliphaticCarbocycles(mol),     # 脂肪族碳环数
                Lipinski.NumAliphaticHeterocycles(mol),    # 脂肪族杂环数
                Lipinski.NumAliphaticRings(mol),           # 脂肪环数
                Lipinski.NumAromaticCarbocycles(mol),      # 芳香族碳环数
                Lipinski.NumAromaticHeterocycles(mol),     # 芳香族杂环数
                Lipinski.NumAromaticRings(mol),            # 芳香环数
                Lipinski.NumSaturatedCarbocycles(mol),     # 饱和碳环数
                Lipinski.NumSaturatedHeterocycles(mol),    # 饱和杂环数
                Lipinski.NumSaturatedRings(mol),           # 饱和环数
                Descriptors.NumValenceElectrons(mol),      # 价电子数
                Descriptors.MaxAbsPartialCharge(mol),      # 最大绝对部分电荷
                Descriptors.MinAbsPartialCharge(mol),      # 最小绝对部分电荷
                Descriptors.NumRadicalElectrons(mol)       # 自由电子数
            ])
        
        # 合并指纹和描述符
        feature_vector = np.concatenate([fingerprint, descriptors])
        features.append(feature_vector)
    
    return np.array(features)

def prepare_dataset(df, max_features=1000):
    """
    准备数据集用于训练
    
    Parameters:
    df (DataFrame): 输入数据框
    max_features (int): 最大特征数
    
    Returns:
    tuple: (X, y, feature_selector) 特征矩阵、目标值和特征选择器
    """
    # 提取SMILES和目标值
    smiles_list = df['SMILES'].tolist()
    targets = df[['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']].values
    
    # 从SMILES生成特征
    X = smiles_to_features(smiles_list)
    
    # 特征选择（如果特征数超过max_features）
    if X.shape[1] > max_features:
        selector = SelectKBest(score_func=f_regression, k=max_features)
        X_selected = selector.fit_transform(X, targets[:, 0])  # 使用第一个目标进行特征选择
        return X_selected, targets, selector
    else:
        return X, targets, None