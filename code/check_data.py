import pandas as pd
import numpy as np

mol_df = pd.read_csv("data/molecule.smi", sep='\t', names=['molecule_id','smiles'], dtype=str, keep_default_na=False)
print("molecule.smi head:")
print(mol_df.head())

act_df = pd.read_csv("data/activity_train.csv")
print("activity_train columns:", act_df.columns.tolist())
# 检查是否含有 NaN smiles
merged = act_df.merge(mol_df, on='molecule_id', how='left')
na_smiles = merged['smiles'].isna().sum()
print("activity rows with missing smiles:", na_smiles)
# 打印出部分包含缺失 smiles 的行以便排查
print(merged[merged['smiles'].isna()].head(10))
