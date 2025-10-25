# robust_read_smiles.py
import os
import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smi):
    """Return canonical SMILES (isomeric) or None if invalid."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return None

def read_smiles(path, sep=',', save_invalid_to="results/invalid_smiles.csv", ensure_unique_ids=True):
    """
    读取类似格式的 SMILES 文件（无表头，格式：mol_id,SMILES）
    - path: 文件路径（相对或绝对）
    - sep: 分隔符，默认逗号（你的示例就是逗号）
    - save_invalid_to: 若存在无法解析为 RDKit 分子的行，保存这些行到该 CSV 文件（包含原始文本）
    - ensure_unique_ids: 若 mol_id 有重复，自动给后续重复项添加后缀 "_1", "_2", ...
    
    返回: pandas.DataFrame with columns:
      - 'molecule_id' (str)
      - 'smiles' (canonical smiles, str)
      - 'rdkit_mol' (RDKit Mol object)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMILES file not found: {path}")
    # 首先尝试用 pandas 直接读取两列（安全、会识别引号）
    try:
        df = pd.read_csv(path, header=None, sep=sep, engine='python', dtype=str, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read file with pandas: {e}")
    # 如果行只有一列（罕见），则按第一个逗号分割（fallback）
    if df.shape[1] == 1:
        # split by first comma manually (handles cases like "MOL_0001,SMILES" that pandas didn't split)
        records = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip("\n\r")
                if line.strip() == "":
                    continue
                # split at first comma occurrence
                if ',' in line:
                    mol_id, smiles = line.split(',', 1)
                else:
                    # no comma: treat whole line as smiles (no id)
                    mol_id, smiles = None, line
                records.append((mol_id, smiles))
        df = pd.DataFrame(records)
    # Now we expect at least two columns: take first two as id and smiles
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['molecule_id', 'smiles_raw']
    else:
        raise RuntimeError("Unable to parse file into two columns (mol_id, smiles).")
    # strip whitespace
    df['molecule_id'] = df['molecule_id'].astype(str).str.strip()
    df['smiles_raw'] = df['smiles_raw'].astype(str).str.strip()
    # canonicalize smiles and build rdkit mol; collect invalids
    canonical_smiles = []
    rdkit_mols = []
    invalid_rows = []
    for idx, row in df.iterrows():
        smi = row['smiles_raw']
        if smi == "" or smi.upper() == "NA":
            canonical_smiles.append(None)
            rdkit_mols.append(None)
            invalid_rows.append({'index': idx, 'molecule_id': row['molecule_id'], 'smiles_raw': smi, 'reason': 'empty_or_NA'})
            continue
        canon = canonicalize_smiles(smi)
        if canon is None:
            canonical_smiles.append(None)
            rdkit_mols.append(None)
            invalid_rows.append({'index': idx, 'molecule_id': row['molecule_id'], 'smiles_raw': smi, 'reason': 'rdkit_parse_failed'})
        else:
            canonical_smiles.append(canon)
            rdkit_mols.append(Chem.MolFromSmiles(canon))
    df['smiles'] = canonical_smiles
    df['rdkit_mol'] = rdkit_mols
    # drop rows with invalid smiles (but save them)
    if len(invalid_rows) > 0:
        os.makedirs(os.path.dirname(save_invalid_to) or '.', exist_ok=True)
        pd.DataFrame(invalid_rows).to_csv(save_invalid_to, index=False)
        print(f"[read_smiles] {len(invalid_rows)} invalid SMILES rows saved to {save_invalid_to}")
    df_valid = df[df['smiles'].notna()].copy().reset_index(drop=True)
    # ensure molecule_id uniqueness if requested
    if ensure_unique_ids:
        ids = df_valid['molecule_id'].astype(str).tolist()
        seen = {}
        unique_ids = []
        for i, mid in enumerate(ids):
            if mid == "" or mid.lower() == "nan":
                mid = f"mol_{i}"
            if mid not in seen:
                seen[mid] = 0
                unique_ids.append(mid)
            else:
                seen[mid] += 1
                new_id = f"{mid}_{seen[mid]}"
                unique_ids.append(new_id)
        df_valid['molecule_id'] = unique_ids
    # reorder columns and return
    out = df_valid[['molecule_id', 'smiles', 'rdkit_mol']]
    out = out.rename(columns={'molecule_id':'molecule_id', 'smiles':'smiles', 'rdkit_mol':'rdkit_mol'})
    return out
