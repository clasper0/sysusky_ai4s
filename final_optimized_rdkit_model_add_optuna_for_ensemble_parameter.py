import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

# 新增optuna导入
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Please install it using: pip install optuna")

def load_and_prepare_extended_data():
    """
    加载并准备数据，使用扩展的RDKit特征（包括1024维Morgan指纹）
    """
    # 读取原始活性数据
    activity_df = pd.read_csv('data/activity_train.csv')
    
    # 读取扩展的RDKit分子特征
    features_df = pd.read_csv('data/train_features_extended.csv')
    
    # 合并数据
    merged_df = pd.merge(activity_df, features_df, on='molecule_id', how='left')
    
    # 对靶点进行编码
    le_target = LabelEncoder()
    merged_df['target_id_encoded'] = le_target.fit_transform(merged_df['target_id'])
    
    # 准备特征和目标变量
    feature_columns = [
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
    features = ['target_id_encoded'] + feature_columns
    X = merged_df[features]
    y = merged_df['pIC50']
    
    return X, y, le_target, merged_df

def build_final_optimized_model(rf_weight=1, gb_weight=1.2, xgb_weight=1.3, ridge_weight=0.5):
    """
    构建最终优化的集成学习模型
    """
    # 定义多个基础模型，使用优化的参数
    rf = RandomForestRegressor(n_estimators=200, max_depth=9, min_samples_split=4,
                              min_samples_leaf=1, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                   subsample=0.85, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 random_state=42, n_jobs=-1)
    ridge = Ridge(alpha=0.8)
    
    # 创建集成模型，调整权重以优化性能
    ensemble_model = VotingRegressor([
        ('random_forest', rf),
        ('gradient_boosting', gb),
        ('xgboost', xgb_model),
        ('ridge', ridge)
    ], weights=[rf_weight, gb_weight, xgb_weight, ridge_weight])
    
    return ensemble_model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能，包括Pearson、Spearman相关系数和MAE
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)  # 新增MAE计算
    
    # 计算Pearson和Spearman相关系数
    pearson_corr, pearson_p = pearsonr(y_test, y_pred)
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)
    
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")  # 新增MAE输出
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    return y_pred, pearson_corr, spearman_corr, mae  # 返回MAE

def cross_validation(model, X, y):
    """
    进行交叉验证
    """
    cv_scores_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_scores_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    
    print(f"Cross-validation R² scores: {cv_scores_r2}")
    print(f"Average CV R² score: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")
    print(f"Average CV RMSE score: {np.sqrt(-cv_scores_mse.mean()):.4f} (+/- {np.sqrt(cv_scores_mse.std() * 2):.4f})")

def feature_selection_analysis(X, y):
    """
    进行特征选择分析
    """
    print("Performing feature selection analysis...")
    
    # 分离靶点编码和RDKit特征
    X_target = X[['target_id_encoded']]
    feature_columns = [
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
    X_features = X[feature_columns]
    
    # 标准化特征
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
    
    # 使用SelectKBest选择最佳特征，选择250个特征
    selector = SelectKBest(score_func=f_regression, k=250)
    X_features_selected = selector.fit_transform(X_features_scaled, y)
    
    # 合并靶点编码和选择的特征
    X_selected = np.hstack([X_target.values, X_features_selected])
    
    selected_features_indices = selector.get_support(indices=True)
    print(f"Selected {len(selected_features_indices)} features out of {X_features.shape[1]}")
    
    return X_selected, selected_features_indices, scaler, selector

def prepare_candidate_features_with_selection(selected_features_indices, scaler, selector):
    """
    准备经过特征选择的候选分子特征
    """
    # 读取候选分子特征
    candidate_features = pd.read_csv('data/candidate_features_extended.csv')
    
    # 提取特征列
    feature_columns = [
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
    features = candidate_features[feature_columns]
    
    # 应用标准化和特征选择
    features_scaled = scaler.transform(features)
    features_selected = selector.transform(features_scaled)
    
    # 创建新的DataFrame
    selected_columns = [f'selected_feature_{i}' for i in range(features_selected.shape[1])]
    features_selected_df = pd.DataFrame(features_selected, columns=selected_columns)
    features_selected_df.insert(0, 'molecule_id', candidate_features['molecule_id'])
    
    return features_selected_df

def predict_candidates(model, le_target, selected_features_indices, scaler, selector):
    """
    对候选分子进行预测
    """
    try:
        # 准备候选分子特征
        candidate_features_raw = pd.read_csv('data/candidate_features_extended.csv')
        candidate_features_selected = prepare_candidate_features_with_selection(selected_features_indices, scaler, selector)
        
        # 读取靶点数据
        targets = pd.read_csv('data/target.csv')
        
        # 准备预测数据
        predictions = []
        for idx, candidate in candidate_features_raw.iterrows():
            for _, target in targets.iterrows():
                # 构造特征
                tar_id_encoded = le_target.transform([target['target_id']])[0] if target['target_id'] in le_target.classes_ else -1
                
                # 获取选择的特征
                selected_features = candidate_features_selected.iloc[idx, 1:].values  # 跳过molecule_id列
                
                # 构造输入向量
                input_features = np.array([[tar_id_encoded] + list(selected_features)])
                
                # 预测
                pred = model.predict(input_features)[0]
                
                # 确保预测值在合理范围内 (pIC50通常在0-15之间)
                pred = max(0, min(15, pred))
                
                predictions.append({
                    'molecule_id': candidate['molecule_id'],
                    'target_id': target['target_id'],
                    'target_name': target['target_name'],
                    'predicted_pIC50': pred
                })
        
        # 转换为DataFrame并排序
        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.sort_values(['molecule_id', 'predicted_pIC50'], ascending=[True, False])
        
        print("\nTop predictions for candidate molecules:")
        print(pred_df.head(20))
        
        # 保存预测结果
        pred_df.to_csv('output/final_optimized_candidate_predictions.csv', index=False)
        print("\nPredictions saved to output/final_optimized_candidate_predictions.csv")
        
    except Exception as e:
        print(f"Error predicting candidates: {e}")

def analyze_model_performance_by_target(model, X_test, y_test, original_data, test_indices):
    """
    按靶点分析模型性能
    """
    print("\nPerformance by Target:")
    print("-" * 30)
    
    # 获取测试集中的靶点信息
    test_data = original_data.iloc[test_indices].copy()
    
    y_pred = model.predict(X_test)
    
    for target_id in test_data['target_id'].unique():
        mask = test_data['target_id'] == target_id
        if mask.sum() > 0:
            target_y_true = y_test[mask]
            target_y_pred = y_pred[mask]
            target_r2 = r2_score(target_y_true, target_y_pred)
            target_rmse = np.sqrt(mean_squared_error(target_y_true, target_y_pred))
            print(f"{target_id}: R² = {target_r2:.4f}, RMSE = {target_rmse:.4f}")

# 新增Optuna优化函数
def optimize_weights_with_optuna(X_train, y_train, X_val, y_val):
    """
    使用Optuna优化模型权重
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna is not available. Returning default weights.")
        return [1, 1.2, 1.3, 0.5]
    
    def objective(trial):
        # 定义权重搜索空间
        rf_weight = trial.suggest_float('rf_weight', 0.1, 2.0)
        gb_weight = trial.suggest_float('gb_weight', 0.1, 2.0)
        xgb_weight = trial.suggest_float('xgb_weight', 0.1, 2.0)
        ridge_weight = trial.suggest_float('ridge_weight', 0.1, 2.0)
        
        # 构建模型
        model = build_final_optimized_model(rf_weight, gb_weight, xgb_weight, ridge_weight)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 验证集评估
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return rmse
    
    # 创建study对象
    study = optuna.create_study(direction='minimize')
    
    # 运行优化
    study.optimize(objective, n_trials=5)
    
    # 输出最佳参数
    best_params = study.best_params
    print("Best weights found:")
    print(f"Random Forest weight: {best_params['rf_weight']:.3f}")
    print(f"Gradient Boosting weight: {best_params['gb_weight']:.3f}")
    print(f"XGBoost weight: {best_params['xgb_weight']:.3f}")
    print(f"Ridge weight: {best_params['ridge_weight']:.3f}")
    
    return [best_params['rf_weight'], best_params['gb_weight'], 
            best_params['xgb_weight'], best_params['ridge_weight']]

def main():
    """
    主函数
    """
    print("Loading and preparing data with extended RDKit features...")
    X, y, le_target, original_data = load_and_prepare_extended_data()
    
    print("Performing feature selection...")
    X_selected, selected_features_indices, scaler, selector = feature_selection_analysis(X, y)
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 进一步划分训练集，创建验证集用于权重优化
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    # 使用Optuna优化权重
    print("Optimizing model weights with Optuna...")
    optimal_weights = optimize_weights_with_optuna(X_train_split, y_train_split, X_val, y_val)
    
    print("Building final optimized ensemble model with optimal weights...")
    model = build_final_optimized_model(*optimal_weights)
    
    print("Training ensemble model...")
    model.fit(X_train, y_train)
    
    print("Evaluating ensemble model...")
    y_pred, pearson_corr, spearman_corr, mae = evaluate_model(model, X_test, y_test) 
    
    print("Performing cross-validation...")
    cross_validation(model, X_selected, y)
    
    print("Analyzing model performance by target...")
    # 获取测试集索引
    test_indices = original_data.sample(frac=0.2, random_state=42).index
    analyze_model_performance_by_target(model, X_test, y_test, original_data, test_indices)
    
    print("Predicting candidate molecules...")
    predict_candidates(model, le_target, selected_features_indices, scaler, selector)
    
    print("Final optimized model training and evaluation completed!")

if __name__ == "__main__":
    main()