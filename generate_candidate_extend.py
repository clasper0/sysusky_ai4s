from deepchem.feat import RDKitDescriptors
import pandas as pd
import numpy as np

def generate_rdkit_features(input_file, output_file, max_features=1074):
    """
    生成RDKit分子特征，使用真实的描述符名称作为列名
    """
    # 初始化RDKit特征生成器
    rdkit_feat = RDKitDescriptors()
    
    # 读取分子数据 (molecule.smi 没有标题行)
    molecules_df = pd.read_csv(input_file)
    
    # 存储特征的列表
    features_list = []
    failed_molecules = []
    
    # 为每个分子生成特征
    for idx, row in molecules_df.iterrows():
        try:
            smiles = row['SMILES']
            # 生成特征
            features = rdkit_feat.featurize(smiles)
            features_list.append(features[0])  # features是二维数组
        except Exception as e:
            print(f"处理分子 {row['molecule_id']} 时出错: {e}")
            failed_molecules.append(row['molecule_id'])
    
    # 使用真实的RDKit描述符名称作为列名
    feature_names = rdkit_feat.descriptors
    features_df = pd.DataFrame(features_list, columns=feature_names)
    
    # 合并原始数据和特征数据
    result_df = pd.concat([molecules_df, features_df], axis=1)
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    
    print(f"特征生成完成，共处理 {len(result_df)} 个分子")
    print(f"失败分子数: {len(failed_molecules)}")
    if failed_molecules:
        print("失败的分子:", failed_molecules)
    
    return result_df



# 生成特征
generate_rdkit_features('data/candidate.csv', 'data/candidate_features_extended.csv')