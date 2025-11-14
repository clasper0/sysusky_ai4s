#!/usr/bin/env python3
"""
使用 RDKit 生成扩展分子特征的脚本
================================

本脚本使用 RDKit 为每个分子生成全面的分子描述符，
并将这些特征添加到现有数据中，创建扩展的训练数据集。
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings("ignore")


def get_rdkit_descriptor_names():
    """
    获取 RDKit 中所有分子描述符的名字（按 RDKit 的标准顺序）
    """
    # descList 是 [(name, function), ...]
    return [name for name, _ in Descriptors.descList]


def generate_deepchem_features(smiles_list):
    """
    使用 RDKit 为 SMILES 列表生成分子描述符（列名为真实 RDKit 描述符名）
    
    Parameters
    ----------
    smiles_list : list of str
        SMILES 字符串列表
    
    Returns
    -------
    pandas.DataFrame
        每一行对应一个分子，每一列是一个 RDKit 分子描述符
        列名为描述符的真实名字，如 'MolWt', 'TPSA', 'NumHAcceptors' 等
    """
    print("初始化 RDKit 分子描述符计算器...")

    # 1. 获取描述符名字
    descriptor_names = get_rdkit_descriptor_names()
    n_features = len(descriptor_names)
    print(f"将计算 {n_features} 个 RDKit 分子描述符。")

    features_data = []

    print("开始生成分子特征...")
    for i, smiles in enumerate(smiles_list):
        if i % 20 == 0:
            print(f"处理进度: {i}/{len(smiles_list)}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"警告: 无法解析 SMILES: {smiles!r}，使用全零特征代替。")
            features_data.append(np.zeros(n_features, dtype=float))
            continue

        try:
            # 返回的是 {descriptor_name: value} 的字典
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            # 按 descriptor_names 的顺序取值，保证列名和特征顺序一一对应
            row = [desc_dict[name] for name in descriptor_names]
            features_data.append(row)
        except Exception as e:
            print(f"处理 SMILES {smiles!r} 时出错: {e}，使用全零特征代替。")
            features_data.append(np.zeros(n_features, dtype=float))

    print(f"特征生成完成，共处理 {len(features_data)} 个分子。")

    # 2. 直接用真实的描述符名作为列名
    features_df = pd.DataFrame(features_data, columns=descriptor_names)
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

    # 获取 SMILES 列表
    if "SMILES" not in df.columns:
        print("错误: 输入文件中未找到 'SMILES' 列")
        return

    smiles_list = df["SMILES"].tolist()
    print(f"开始为 {len(smiles_list)} 个分子生成扩展特征...")

    # 生成 RDKit 特征
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
        print("前 10 列列名预览:")
        print(list(extended_df.columns)[:10])
        print("...")
        print("后 10 列列名预览:")
        print(list(extended_df.columns)[-10:])
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return


if __name__ == "__main__":
    main()
