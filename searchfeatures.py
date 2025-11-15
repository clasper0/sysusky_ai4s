# analysis_feature_importance_and_fragments.py
"""
分析脚本：输出 per-target Top-10 特征、特征重要性热力图、以及高活性分子的结构片段解读/可视化

要求的数据文件（与你原脚本相同路径）:
- data/activity_train.csv
- data/train_features_extended.csv
- data/candidate_features_extended.csv
- data/target.csv

输出文件（output/ 目录）:
- output/top10_features_per_target.csv
- output/feature_importance_matrix.csv
- output/feature_importance_heatmap.png
- output/top_active_molecule_fragments/  (每个分子的一些 PNG/highlight 图)
- output/fragment_explanations.csv

使用方法:
python analysis_feature_importance_and_fragments.py
"""
import os
import pickle
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# RDKit (for fragment->atom mapping and drawing)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKit_AVAILABLE = True
except Exception as e:
    RDKit_AVAILABLE = False
    print("RDKit not available. Morgan-bit-based fragment highlights will be skipped.")


# --- --------- Helper: feature column list (copied from your script) -----------
FEATURE_COLUMNS = [
"MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "qed", "SPS", "MolWt",
"HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons", "MaxPartialCharge",
"MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge", "FpDensityMorgan1", "FpDensityMorgan2",
"FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI",
"BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "AvgIpc", "BalabanJ", "BertzCT", "Chi0", "Chi0n",
"Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha",
"Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12",
"PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7",
"PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
"SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
"SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
"SlogP_VSA9", "TPSA", "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3",
"EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "VSA_EState1",
"VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7",
"VSA_EState8", "VSA_EState9", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
"NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAmideBonds",
"NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumAtomStereoCenters",
"NumBridgeheadAtoms", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumHeterocycles", "NumRotatableBonds",
"NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumSpiroAtoms",
"NumUnspecifiedAtomStereoCenters", "Phi", "RingCount", "MolLogP", "MolMR", "fr_Al_COO", "fr_Al_OH",
"fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O",
"fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1",
"fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide",
"fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo",
"fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine",
"fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
"fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam",
"fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho",
"fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond",
"fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine",
"fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole",
"fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea"
]
# ---------------------------------------------------------------------------


OUTPUT_DIR = "output"
FRAG_DIR = os.path.join(OUTPUT_DIR, "top_active_molecule_fragments")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAG_DIR, exist_ok=True)


def load_and_merge():
    """Load activity + extended features; returns merged_df and label encoder for target"""
    activity_df = pd.read_csv('data/activity_train.csv')
    features_df = pd.read_csv('data/train_features_extended.csv')
    merged_df = pd.merge(activity_df, features_df, on='molecule_id', how='left')
    le_target = LabelEncoder()
    merged_df['target_id_encoded'] = le_target.fit_transform(merged_df['target_id'])
    return merged_df, le_target


def detect_fingerprint_columns(df):
    """
    自动检测可能的 Morgan/bit 指纹列：
    - 列名中包含 'morgan'/'Morgan'/'fp_bit'/'fingerprint'/'FP' 等
    - 或者为仅含0/1 且列数很多（比如 >= 64）
    返回 fingerprint_columns list（可能为空）
    """
    cols = df.columns.tolist()
    candidates = [c for c in cols if any(k in c.lower() for k in ['morgan', 'fp', 'fingerprint', 'bit', 'morganbit', 'morgan_bit'])]
    if candidates:
        return candidates
    # fallback: pick binary columns
    bin_cols = []
    for c in cols:
        if df[c].dropna().isin([0, 1]).all():
            bin_cols.append(c)
    # if many binary columns (>=64), assume these are fingerprint bits
    if len(bin_cols) >= 64:
        return bin_cols
    return []


def prepare_features_for_model(merged_df):
    """
    准备 X,y 与 SelectKBest，使得后续能够按你的流程复现特征选择
    返回 X_selected (np.array), selected_feature_names (list), scaler, selector, X_full_df
    """
    X = merged_df[['target_id_encoded'] + FEATURE_COLUMNS].copy()
    y = merged_df['pIC50'].values

    # 标准化数值特征（仅对 FEATURE_COLUMNS 部分）
    scaler = StandardScaler()
    X_features = X[FEATURE_COLUMNS].fillna(0)
    X_features_scaled = scaler.fit_transform(X_features)

    # SelectKBest：k=250（和你原脚本一致）
    k = min(250, X_features.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_features_selected = selector.fit_transform(X_features_scaled, y)

    # 合并 target 编码 + selected features
    X_selected = np.hstack([X[['target_id_encoded']].values, X_features_selected])

    # 得到 selected feature names
    sel_idx = selector.get_support(indices=True)
    selected_feature_names = [FEATURE_COLUMNS[i] for i in sel_idx]

    # build a DataFrame for convenience: columns = ['target_id_encoded'] + selected_feature_names
    X_selected_df = pd.DataFrame(
        X_selected,
        columns=['target_id_encoded'] + selected_feature_names,
        index=merged_df.index
    )
    return X_selected_df, y, selected_feature_names, scaler, selector


def build_or_load_ensemble(model_path="output/final_ensemble.pkl"):
    """尝试加载已保存模型，否则按你提供参数构建并返回模型实例（未 fit）"""
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model = load(model_path)
        return model
    # otherwise construct same ensemble as your script (with same optimized hyperparams)
    rf = RandomForestRegressor(n_estimators=200, max_depth=9, min_samples_split=4,
                              min_samples_leaf=1, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                   subsample=0.85, random_state=42)
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 random_state=42, n_jobs=-1)
    ridge = Ridge(alpha=0.8)

    model = VotingRegressor([
        ('random_forest', rf),
        ('gradient_boosting', gb),
        ('xgboost', xgb_model),
        ('ridge', ridge)
    ], weights=[1, 1.2, 1.3, 0.5])  # default weights if not optimized externally
    return model


def train_and_save_model(model, X_train, y_train, path="output/final_ensemble.pkl"):
    """训练并保存模型（如果需要）"""
    print("Training ensemble model (this may take some time)...")
    model.fit(X_train, y_train)
    dump(model, path)
    print(f"Model saved to {path}")
    return model


def compute_permutation_importances_per_target(model, X_selected_df, y, merged_df, selected_feature_names, n_repeats=8, random_state=42):
    """
    对每个 target（使用该target的样本）计算 permutation importance
    返回:
      importance_by_target: dict target_id -> pandas.Series (index: feature_names, values: importances)
    """
    importance_by_target = {}
    # features for importance (include target_id_encoded as first column)
    feat_cols = X_selected_df.columns.tolist()

    # add predictions on full set to allow ranking later
    y_pred_full = model.predict(X_selected_df.values)

    merged_df = merged_df.copy()
    merged_df['predicted_pIC50_model'] = y_pred_full

    targets = merged_df['target_id'].unique().tolist()
    for t in targets:
        mask = merged_df['target_id'] == t
        if mask.sum() < 10:
            # 少样本时跳过或仍计算但提醒
            print(f"Warning: target {t} has only {mask.sum()} samples — permutation importance may be noisy.")
        X_sub = X_selected_df.loc[mask, :].values
        y_sub = merged_df.loc[mask, 'pIC50'].values
        if len(y_sub) < 3:
            # 太少了没法做置换重要性
            # 仍放入 zeros
            importance_by_target[t] = pd.Series(0.0, index=feat_cols)
            continue
        # permutation importance (on subset)
        try:
            perm = permutation_importance(model, X_sub, y_sub, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
            imp_means = perm.importances_mean
        except Exception as e:
            print(f"Permutation importance failed for target {t}: {e}")
            imp_means = np.zeros(len(feat_cols))
        importance_by_target[t] = pd.Series(imp_means, index=feat_cols).sort_values(ascending=False)
    return importance_by_target, merged_df


def save_top10_per_target(importance_by_target, out_csv=os.path.join(OUTPUT_DIR, "top10_features_per_target.csv")):
    """将每个靶点的 Top10 特征汇总并保存为 CSV"""
    rows = []
    for t, s in importance_by_target.items():
        top10 = s.head(10)
        for rank, (feat, val) in enumerate(top10.items(), start=1):
            rows.append({'target_id': t, 'rank': rank, 'feature': feat, 'importance': float(val)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved top10 per target to {out_csv}")


def build_feature_matrix_for_heatmap(importance_by_target, n_features=50):
    """
    汇总重要性矩阵：选择所有靶点 Top-n_features 的并集，然后构造矩阵(target x feature)
    返回 DataFrame (index: targets, columns: features)
    """
    all_top_feats = set()
    for s in importance_by_target.values():
        all_top_feats.update(s.head(n_features).index.tolist())
    all_top_feats = sorted(all_top_feats)
    mat = []
    targets = []
    for t, s in importance_by_target.items():
        row = [s.get(f, 0.0) for f in all_top_feats]
        mat.append(row)
        targets.append(t)
    mat_df = pd.DataFrame(mat, index=targets, columns=all_top_feats)
    mat_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_matrix.csv"))
    return mat_df


def plot_heatmap(mat_df, out_png=os.path.join(OUTPUT_DIR, "feature_importance_heatmap.png")):
    plt.figure(figsize=(max(8, mat_df.shape[1] * 0.15), max(6, mat_df.shape[0] * 0.25)))
    sns.set(font_scale=0.8)
    sns.heatmap(mat_df, cmap="viridis", linewidths=0.2)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved heatmap to {out_png}")


# --- Fragment explanation utilities -----------------------------------------
SMARTS_FOR_FR_FEATURES = {
    # 几个示例映射（fr_ 前缀项的化学意义）
    'fr_nitro': '–NO2 (nitro group)',
    'fr_phenol': 'phenolic –OH on aromatic ring',
    'fr_ester': 'ester group (R–COO–R)',
    'fr_ether': 'ether (R–O–R)',
    'fr_nitrile': '–C≡N (nitrile)',
    'fr_halogen': 'halogen atom (F/Cl/Br/I)',
    'fr_ketone': 'carbonyl in ketone (R–CO–R)',
    'fr_amide': 'amide (R–C(=O)–NR2)',
    # 你可以继续补充
}
# ---------------------------------------------------------------------------


def fragment_interpretation_and_visualization(top_pred_df, selected_feature_names, fingerprint_columns, merged_df):
    """
    对高活性分子进行片段解读
    - top_pred_df: DataFrame 包含 molecule_id, target_id, predicted_pIC50
    - fingerprint_columns: list 或空，表示在 merged_df 中检测到的指纹 bit 列名
    将结果保存：PNG 高亮图（如果 RDKit 可用且检测到 bits），以及 fragment_explanations.csv
    """
    explanations = []
    # map molecule_id -> SMILES if available in merged_df
    smiles_map = {}
    if 'smiles' in merged_df.columns:
        smiles_map = merged_df.set_index('molecule_id')['smiles'].to_dict()
    # else try to load from candidate_features_extended.csv
    if not smiles_map:
        try:
            cand = pd.read_csv('data/candidate_features_extended.csv')
            if 'smiles' in cand.columns:
                smiles_map.update(cand.set_index('molecule_id')['smiles'].to_dict())
        except Exception:
            pass

    # For each (molecule_id, target_id) in top_pred_df, inspect its features
    for _, row in top_pred_df.iterrows():
        mol_id = row['molecule_id']
        target = row['target_id']
        pred = row['predicted_pIC50']

        # gather feature vector row from merged_df if present else from candidate_features_extended
        if mol_id in merged_df['molecule_id'].values:
            feat_row = merged_df[merged_df['molecule_id'] == mol_id].iloc[0]
        else:
            try:
                cand = pd.read_csv('data/candidate_features_extended.csv')
                feat_row = cand[cand['molecule_id'] == mol_id].iloc[0]
            except Exception:
                feat_row = None

        # check if any 'fr_' features among top features for this target (we will report them)
        fr_hits = {}
        if feat_row is not None:
            for fname in selected_feature_names:
                if fname.startswith('fr_') and fname in feat_row.index:
                    val = feat_row[fname]
                    if pd.notna(val) and val > 0:
                        fr_hits[fname] = val

        # If fingerprint bits exist and RDKit available and we have SMILES, try to highlight
        highlight_info = None
        highlight_img_path = None
        if fingerprint_columns and RDKit_AVAILABLE and mol_id in smiles_map:
            smi = smiles_map[mol_id]
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # build the same Morgan fingerprint and bitInfo
                    bitInfo = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=len(fingerprint_columns), bitInfo=bitInfo)
                    # find bits that are set
                    set_bits = [b for b in range(len(fingerprint_columns)) if fp.GetBit(b)]
                    # map set bits to their atom environments
                    # pick top few bits (most frequent or arbitrary)
                    bits_to_draw = set_bits[:6]
                    atom_lists = []
                    for b in bits_to_draw:
                        if b in bitInfo:
                            # bitInfo[b] is list of tuples (atomIdx, radius)
                            atoms = set()
                            for (atomIdx, rad) in bitInfo[b]:
                                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atomIdx)
                                for aid in env:
                                    atoms.add(mol.GetAtomWithIdx(aid).GetIdx())
                                atoms.add(atomIdx)
                            atom_lists.append(sorted(list(atoms)))
                    # Draw molecule with highlights
                    if atom_lists:
                        # flatten and choose first 3 highlights to avoid clutter
                        highlight_atoms = sorted(set().union(*atom_lists[:3]))
                        drawer = Draw.MolDraw2DCairo(500, 300)
                        opts = drawer.drawOptions()
                        Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms)
                        drawer.FinishDrawing()
                        imgdata = drawer.GetDrawingText()
                        highlight_img_path = os.path.join(FRAG_DIR, f"{mol_id}_target_{target}_highlight.png")
                        with open(highlight_img_path, "wb") as f:
                            f.write(imgdata)
                        highlight_info = {
                            'highlight_atoms': highlight_atoms,
                            'bits_drawn': bits_to_draw
                        }
            except Exception as e:
                print(f"Warning: RDKit fragment drawing failed for {mol_id}: {e}")

        # Compose explanation text
        expl_text_parts = []
        if fr_hits:
            sorted_fr = sorted(fr_hits.items(), key=lambda x: -x[1])
            expl_text_parts.append("Detected fragment counts: " + ", ".join([f"{k}={v}" for k, v in sorted_fr[:6]]))
            # map to readable names where possible
            mapped = []
            for k, v in sorted_fr[:6]:
                mapped.append(SMARTS_FOR_FR_FEATURES.get(k, k))
            expl_text_parts.append("Likely functional groups: " + ", ".join(mapped))
        if highlight_info:
            expl_text_parts.append(f"Highlighted atoms: {highlight_info['highlight_atoms']}; bits: {highlight_info['bits_drawn']}")
        if not expl_text_parts:
            expl_text_parts.append("No direct fragment features (fr_*) or fingerprint bits found / or no SMILES available to highlight.")
        explanation = " | ".join(expl_text_parts)

        explanations.append({
            'molecule_id': mol_id,
            'target_id': target,
            'predicted_pIC50': pred,
            'explanation': explanation,
            'highlight_image': highlight_img_path if highlight_img_path else ""
        })
    expl_df = pd.DataFrame(explanations)
    expl_df.to_csv(os.path.join(OUTPUT_DIR, "fragment_explanations.csv"), index=False)
    print(f"Saved fragment explanations to {os.path.join(OUTPUT_DIR, 'fragment_explanations.csv')}")
    return expl_df


def main():
    merged_df, le_target = load_and_merge()
    X_selected_df, y, selected_feature_names, scaler, selector = prepare_features_for_model(merged_df)

    # Build or load model
    model = build_or_load_ensemble()
    model_path = "output/final_ensemble.pkl"
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        # quick train: split and train on 80% data to get a usable model for importance analysis
        X_train, X_test, y_train, y_test = train_test_split(X_selected_df.values, y, test_size=0.2, random_state=42)
        model = train_and_save_model(model, X_train, y_train, path=model_path)
        # report quick eval
        y_pred = model.predict(X_test)
        print("Quick eval on hold-out: RMSE=", np.sqrt(mean_squared_error(y_test, y_pred)), "R2=", r2_score(y_test, y_pred))

    # compute per-target permutation importances
    print("Detecting fingerprint columns (if present) in train features...")
    fingerprint_columns = detect_fingerprint_columns(pd.read_csv('data/train_features_extended.csv'))

    print("Computing permutation importances per target (may take time)...")
    importance_by_target, merged_df_with_pred = compute_permutation_importances_per_target(
        model, X_selected_df, y, merged_df, selected_feature_names, n_repeats=8)

    # save top10 per target
    save_top10_per_target(importance_by_target)

    # build matrix & heatmap (use top-50 features per target as candidate columns)
    mat_df = build_feature_matrix_for_heatmap(importance_by_target, n_features=50)
    plot_heatmap(mat_df)

    # For fragment interpretation: we will pick top predicted molecules per target
    # compute predictions on the whole dataset using the trained model
    preds = model.predict(X_selected_df.values)
    merged_df_with_pred['predicted_pIC50_model'] = preds

    top_per_target_rows = []
    for t in merged_df_with_pred['target_id'].unique():
        sub = merged_df_with_pred[merged_df_with_pred['target_id'] == t]
        topk = sub.sort_values('predicted_pIC50_model', ascending=False).head(10)
        for _, r in topk.iterrows():
            top_per_target_rows.append({'molecule_id': r['molecule_id'], 'target_id': t, 'predicted_pIC50': r['predicted_pIC50_model']})
    top_pred_df = pd.DataFrame(top_per_target_rows)

    # deduplicate to a reasonable smaller set if huge
    top_pred_df = top_pred_df.drop_duplicates(subset=['molecule_id', 'target_id']).reset_index(drop=True)

    # fragment interpretation and visualization
    expl_df = fragment_interpretation_and_visualization(top_pred_df, selected_feature_names, fingerprint_columns, merged_df)

    print("All done. Outputs in ./output/")

if __name__ == "__main__":
    main()
