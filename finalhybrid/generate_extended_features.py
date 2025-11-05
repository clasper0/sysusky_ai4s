#!/usr/bin/env python3
"""
使用DeepChem生成扩展分子特征的脚本
================================

该脚本使用DeepChem库为每个分子生成更全面的分子描述符，
并将这些特征添加到现有数据中，创建扩展的训练数据集。
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from deepchem.feat import RDKitDescriptors
import warnings
warnings.filterwarnings("ignore")

def generate_deepchem_features(smiles_list):
    """
    使用DeepChem为SMILES列表生成分子描述符
    
    Parameters:
    smiles_list (list): SMILES字符串列表
    
    Returns:
    pandas.DataFrame: 包含各种分子描述符的DataFrame
    """
    print("初始化DeepChem特征生成器...")
    
    # 初始化不同的特征生成器
    rdkit_feat = RDKitDescriptors()
    
    print("开始生成分子特征...")
    features_data = []
    
    for i, smiles in enumerate(smiles_list):
        if i % 20 == 0:
            print(f"处理进度: {i}/{len(smiles_list)}")
            
        try:
            # 创建RDKit分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"警告: 无法解析SMILES: {smiles}")
                # 添加零向量作为占位符
                features_data.append(np.zeros(200))
                continue
                
            # 生成RDKit描述符 (大约200个特征)
            rdkit_features = rdkit_feat.featurize(smiles)
            
            # 确保特征维度一致
            if len(rdkit_features.flatten()) != 200:
                # 如果维度不匹配，进行填充或截断
                features_flat = rdkit_features.flatten()
                if len(features_flat) > 200:
                    features_flat = features_flat[:200]
                else:
                    features_flat = np.pad(features_flat, (0, 200 - len(features_flat)), 'constant')
            
            features_data.append(features_flat)
            
        except Exception as e:
            print(f"处理SMILES '{smiles}' 时出错: {e}")
            # 添加零向量作为占位符
            features_data.append(np.zeros(200))
    
    print(f"特征生成完成，共处理 {len(features_data)} 个分子")
    
    # 创建特征名称
    feature_names = [f"rdkit_{i}" for i in range(200)]
    
    # 创建DataFrame
    features_df = pd.DataFrame(features_data, columns=feature_names)
    return features_df

def main():
    """主函数"""
    print("扩展分子特征生成脚本")
    print("=" * 50)
    
    # 读取原始数据
    input_file = "data/candidate_hybrid.csv"
    output_file = "data/candidate_hybrid_Extended.csv"
    
    print(f"从 {input_file} 读取数据...")
    try:
        df = pd.read_csv(input_file)
        print(f"成功加载 {len(df)} 个分子数据")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 获取SMILES列表
    smiles_list = df["SMILES"].tolist()
    print(f"开始为 {len(smiles_list)} 个分子生成扩展特征...")
    
    # 生成DeepChem特征
    features_df = generate_deepchem_features(smiles_list)
    
    # 合并原始数据和新特征
    print("合并原始数据和扩展特征...")
    extended_df = pd.concat([df, features_df], axis=1)
    
    # 保存扩展数据
    print(f"保存扩展数据到 {output_file}...")
    try:
        extended_df.to_csv(output_file, index=False)
        print("数据保存成功!")
        print(f"新数据集包含 {len(extended_df)} 行，{len(extended_df.columns)} 列")
        print("列名预览:")
        print(list(extended_df.columns)[:10])  # 显示前10列
        print("...")
        print(list(extended_df.columns)[-10:])  # 显示后10列
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

if __name__ == "__main__":
    main()