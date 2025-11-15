import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_ensemble_overall_feature_importance():
    """
    分析集成模型整体的特征重要性
    """
    print("加载和准备数据...")
    X, y, le_target, original_data = load_and_prepare_extended_data()
    
    print("执行特征选择...")
    X_selected, selected_features_indices, scaler, selector = feature_selection_analysis(X, y)
    
    print("构建优化的集成模型...")
    ensemble_model = build_final_optimized_model()
    
    print("训练模型...")
    ensemble_model.fit(X_selected, y)
    
    # 获取特征名称
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
    selected_feature_names = [feature_columns[i] for i in selected_features_indices]
    feature_names = ['target_id_encoded'] + selected_feature_names
    
    print("计算集成模型整体的特征重要性...")
    
    # 获取基模型
    base_models = {}
    estimators = ensemble_model.estimators_
    model_names = ['random_forest', 'gradient_boosting', 'xgboost', 'ridge']
    for i, model in enumerate(estimators):
        base_models[model_names[i]] = model
    
    # 获取各模型的权重
    if hasattr(ensemble_model, 'weights_'):
        weights = ensemble_model.weights_
    else:
        weights = [1, 1, 1, 1]
    weights = np.array(weights) / sum(weights)  # 归一化权重
    
    # 收集各模型的特征重要性
    importances_list = []
    for model_name, model in base_models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):  # 对于线性模型如Ridge
            importances = np.abs(model.coef_)
        else:
            importances = np.zeros(len(feature_names))
        importances_list.append(importances)
    
    # 计算加权平均重要性
    weighted_importances = np.average(importances_list, axis=0, weights=weights)
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': weighted_importances
    }).sort_values('importance', ascending=False)
    
    # 保存整体特征重要性
    feature_importance_df.to_csv('output/ensemble_overall_feature_importance.csv', index=False)
    print("整体特征重要性已保存到 output/ensemble_overall_feature_importance.csv")
    
    # 保存Top 30重要特征
    top_30_features = feature_importance_df.head(30)
    top_30_features.to_csv('output/top30_ensemble_features.csv', index=False)
    print("Top 30重要特征已保存到 output/top30_ensemble_features.csv")
    
    # 创建特征重要性可视化
    create_feature_importance_plot(feature_importance_df.head(20))
    
    # 按特征类型分析重要性
    analyze_feature_importance_by_type(feature_importance_df)
    
    # 分析各靶点的重要特征和高活性分子
    all_target_predictions = analyze_per_target_features_and_molecules(
        ensemble_model, le_target, selected_features_indices, scaler, selector, 
        feature_names, feature_columns, original_data
    )
    
    # 创建热图可视化
    if all_target_predictions is not None:
        create_prediction_heatmap(all_target_predictions)
        create_target_comparison_heatmap(all_target_predictions)
        create_feature_correlation_heatmap(original_data, feature_columns, selected_features_indices)
    
    return feature_importance_df

def create_feature_importance_plot(feature_importance_df):
    """
    创建特征重要性条形图
    """
    print("创建特征重要性可视化...")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature', palette='viridis')
    plt.title('Top 20 Feature Importances in Ensemble Model')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('output/ensemble_feature_importance_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("特征重要性条形图已保存到 output/ensemble_feature_importance_barplot.png")

def analyze_feature_importance_by_type(feature_importance_df):
    """
    按特征类型分析重要性
    """
    print("按特征类型分析重要性...")
    
    # 分析靶点编码特征
    target_importance = feature_importance_df[feature_importance_df['feature'] == 'target_id_encoded']
    if not target_importance.empty:
        print(f"靶点编码特征重要性: {target_importance['importance'].iloc[0]:.4f}")
    
    # 分析RDKit描述符特征
    rdkit_features = feature_importance_df[feature_importance_df['feature'] != 'target_id_encoded']
    if not rdkit_features.empty:
        print(f"RDKit描述符特征平均重要性: {rdkit_features['importance'].mean():.4f}")
        print(f"最重要的RDKit特征: {rdkit_features.iloc[0]['feature']} ({rdkit_features.iloc[0]['importance']:.4f})")
    
    # 创建特征类型分析报告
    analysis_report = pd.DataFrame({
        'feature_type': ['target_id_encoded', 'rdkit_descriptors'],
        'avg_importance': [
            target_importance['importance'].iloc[0] if not target_importance.empty else 0,
            rdkit_features['importance'].mean() if not rdkit_features.empty else 0
        ],
        'max_importance': [
            target_importance['importance'].iloc[0] if not target_importance.empty else 0,
            rdkit_features['importance'].max() if not rdkit_features.empty else 0
        ],
        'count': [
            len(target_importance),
            len(rdkit_features)
        ]
    })
    
    analysis_report.to_csv('output/feature_type_importance_analysis.csv', index=False)
    print("特征类型分析报告已保存到 output/feature_type_importance_analysis.csv")

def analyze_per_target_features_and_molecules(model, le_target, selected_features_indices, scaler, selector, 
                                            feature_names, all_feature_columns, original_data):
    """
    分析各靶点的重要特征和高活性分子 - 完全重新设计的方法
    """
    print("分析各靶点的重要特征和高活性分子...")
    
    # 读取候选分子数据
    try:
        candidate_features = pd.read_csv('data/candidate_features_extended.csv')
    except FileNotFoundError:
        print("未找到候选分子特征文件 data/candidate_features_extended.csv")
        return None
    
    # 读取靶点数据
    try:
        targets = pd.read_csv('data/target.csv')
    except FileNotFoundError:
        print("未找到靶点数据文件 data/target.csv")
        return None
    
    # 获取所有靶点
    target_ids = targets['target_id'].unique()
    
    # 存储所有靶点的预测结果
    all_target_predictions = []
    
    # 为每个靶点分析
    for target_id in target_ids:
        print(f"\n分析靶点: {target_id}")
        
        # 1. 为该靶点分析特征重要性 - 使用相关性分析
        target_specific_features = analyze_target_specific_correlation(
            target_id, original_data, all_feature_columns, selected_features_indices
        )
        
        if target_specific_features is not None:
            # 保存该靶点的Top 5重要特征
            top_5_features = target_specific_features.head(5)
            top_5_filename = f'output/top5_features_{target_id}.csv'
            top_5_features.to_csv(top_5_filename, index=False)
            print(f"  {target_id}的Top 5重要特征已保存到 {top_5_filename}")
            
            # 显示该靶点的Top 5重要特征
            print(f"  {target_id}的Top 5重要特征:")
            for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
                print(f"    {i}. {row['feature']}: 相关性={row['correlation']:.4f}, p值={row['p_value']:.4f}")
        
        # 2. 预测候选分子在该靶点的活性
        target_predictions = predict_target_activity(
            model, target_id, le_target, candidate_features, 
            selected_features_indices, scaler, selector
        )
        
        if target_predictions is not None:
            # 保存Top 5高活性分子
            top_5_molecules = target_predictions.head(5)
            top_5_molecules_filename = f'output/top5_molecules_{target_id}.csv'
            top_5_molecules.to_csv(top_5_molecules_filename, index=False)
            print(f"  {target_id}的Top 5高活性分子已保存到 {top_5_molecules_filename}")
            
            # 显示Top 3高活性分子
            print(f"  {target_id}的Top 3高活性分子:")
            for i, (_, row) in enumerate(top_5_molecules.head(3).iterrows(), 1):
                print(f"    {i}. {row['molecule_id']}: {row['predicted_pIC50']:.4f}")
            
            # 存储用于热图的数据
            for _, row in target_predictions.iterrows():
                all_target_predictions.append({
                    'target_id': target_id,
                    'molecule_id': row['molecule_id'],
                    'predicted_pIC50': row['predicted_pIC50']
                })
    
    return pd.DataFrame(all_target_predictions) if all_target_predictions else None

def analyze_target_specific_correlation(target_id, original_data, all_feature_columns, selected_features_indices):
    """
    为特定靶点分析特征与活性的相关性 - 完全不同的方法
    """
    try:
        # 筛选该靶点的数据
        target_data = original_data[original_data['target_id'] == target_id].copy()
        
        if len(target_data) < 5:  # 数据量太少，无法有效分析
            print(f"  警告: 靶点 {target_id} 的数据量不足 ({len(target_data)} 个样本)，使用备选方法")
            return analyze_target_features_alternative(target_id, original_data, all_feature_columns, selected_features_indices)
        
        # 获取选择的特征名称
        selected_feature_names = [all_feature_columns[i] for i in selected_features_indices]
        
        # 计算每个特征与pIC50的相关性
        correlation_results = []
        for feature in selected_feature_names:
            # 检查特征是否存在且不是常量
            if feature in target_data.columns and target_data[feature].nunique() > 1:
                # 计算Pearson相关性
                corr, p_value = pearsonr(target_data[feature], target_data['pIC50'])
                
                # 计算Spearman相关性作为备选
                spearman_corr, spearman_p = spearmanr(target_data[feature], target_data['pIC50'])
                
                correlation_results.append({
                    'feature': feature,
                    'correlation': abs(corr),  # 使用绝对值，因为正负相关都重要
                    'p_value': p_value,
                    'spearman_correlation': abs(spearman_corr),
                    'spearman_p_value': spearman_p
                })
        
        if not correlation_results:
            print(f"  警告: 无法计算靶点 {target_id} 的特征相关性")
            return None
        
        # 创建相关性DataFrame并按相关性排序
        correlation_df = pd.DataFrame(correlation_results)
        correlation_df = correlation_df.sort_values('correlation', ascending=False)
        
        return correlation_df
        
    except Exception as e:
        print(f"  分析靶点 {target_id} 的特征相关性时出错: {str(e)}")
        return analyze_target_features_alternative(target_id, original_data, all_feature_columns, selected_features_indices)

def analyze_target_features_alternative(target_id, original_data, all_feature_columns, selected_features_indices):
    """
    备选方法：当数据量不足时，使用所有数据但按靶点分组分析
    """
    try:
        # 获取选择的特征名称
        selected_feature_names = [all_feature_columns[i] for i in selected_features_indices]
        
        # 计算每个特征在不同靶点中的变异系数
        feature_variability = []
        for feature in selected_feature_names:
            if feature in original_data.columns:
                # 计算该特征在不同靶点中的标准差/均值（变异系数）
                target_means = original_data.groupby('target_id')[feature].mean()
                if len(target_means) > 1 and target_means.std() > 0:
                    cv = target_means.std() / target_means.mean()  # 变异系数
                else:
                    cv = 0
                
                # 计算该特征与活性的整体相关性
                if original_data[feature].nunique() > 1:
                    corr, p_value = pearsonr(original_data[feature], original_data['pIC50'])
                else:
                    corr, p_value = 0, 1
                
                feature_variability.append({
                    'feature': feature,
                    'variability': abs(cv),
                    'correlation': abs(corr),
                    'p_value': p_value
                })
        
        if not feature_variability:
            return None
        
        # 创建特征重要性DataFrame并按重要性排序
        importance_df = pd.DataFrame(feature_variability)
        
        # 结合变异性和相关性计算综合重要性
        importance_df['importance'] = importance_df['variability'] * importance_df['correlation']
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df[['feature', 'importance', 'correlation', 'p_value']]
        
    except Exception as e:
        print(f"  备选方法分析靶点 {target_id} 的特征时出错: {str(e)}")
        return None

def predict_target_activity(model, target_id, le_target, candidate_features, 
                          selected_features_indices, scaler, selector):
    """
    预测候选分子在特定靶点的活性
    """
    try:
        # 准备候选分子特征
        exclude_columns = ['molecule_id', 'SMILES']
        candidate_feature_cols = [col for col in candidate_features.columns if col not in exclude_columns]
        candidate_features_only = candidate_features[candidate_feature_cols]
        
        # 应用标准化和特征选择
        candidate_features_scaled = scaler.transform(candidate_features_only)
        candidate_features_selected = selector.transform(candidate_features_scaled)
        
        # 获取靶点编码
        if target_id in le_target.classes_:
            target_encoded = le_target.transform([target_id])[0]
        else:
            print(f"  警告: 靶点 {target_id} 不在训练数据中，使用默认编码")
            target_encoded = 0  # 使用默认编码
        
        # 预测活性
        target_predictions = []
        for idx, candidate in candidate_features.iterrows():
            # 构造输入特征（靶点编码 + 选择的特征）
            input_features = np.array([[target_encoded] + list(candidate_features_selected[idx])])
            
            # 预测
            pred = model.predict(input_features)[0]
            
            # 确保预测值在合理范围内
            pred = max(0, min(15, pred))
            
            target_predictions.append({
                'molecule_id': candidate['molecule_id'],
                'predicted_pIC50': pred
            })
        
        # 转换为DataFrame并排序
        pred_df = pd.DataFrame(target_predictions)
        pred_df = pred_df.sort_values('predicted_pIC50', ascending=False)
        
        return pred_df
        
    except Exception as e:
        print(f"  预测靶点 {target_id} 的活性时出错: {str(e)}")
        return None

def create_prediction_heatmap(all_target_predictions):
    """
    创建各靶点Top分子预测活性热图
    """
    print("创建预测活性热图...")
    
    # 获取每个靶点的Top 5分子
    top_molecules_by_target = []
    for target_id in all_target_predictions['target_id'].unique():
        target_data = all_target_predictions[all_target_predictions['target_id'] == target_id]
        top_5 = target_data.nlargest(5, 'predicted_pIC50')
        top_5['rank'] = range(1, len(top_5) + 1)
        top_molecules_by_target.append(top_5)
    
    if not top_molecules_by_target:
        print("没有预测数据可用于创建热图")
        return
    
    top_molecules_df = pd.concat(top_molecules_by_target)
    
    # 创建热图数据
    heatmap_data = top_molecules_df.pivot_table(
        index='target_id', 
        columns='rank', 
        values='predicted_pIC50',
        aggfunc='first'
    )
    
    # 创建热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Predicted pIC50'},
                linewidths=0.5)
    plt.title('Top 5 Molecules pIC50 Prediction by Target (Ranked 1-5)')
    plt.xlabel('Rank')
    plt.ylabel('Target ID')
    plt.tight_layout()
    plt.savefig('output/top_molecules_prediction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("预测活性热图已保存到 output/top_molecules_prediction_heatmap.png")

def create_target_comparison_heatmap(all_target_predictions):
    """
    创建靶点间活性比较热图
    """
    print("创建靶点间活性比较热图...")
    
    # 计算每个靶点的统计信息
    target_stats = all_target_predictions.groupby('target_id')['predicted_pIC50'].agg([
        'mean', 'max', 'min', 'std', 'count'
    ]).round(3)
    
    # 创建统计热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(target_stats[['mean', 'max', 'min', 'std']], 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                linewidths=0.5)
    plt.title('Target Activity Statistics Comparison')
    plt.tight_layout()
    plt.savefig('output/target_statistics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计表格
    target_stats.to_csv('output/target_activity_statistics.csv')
    print("靶点统计热图已保存到 output/target_statistics_heatmap.png")
    print("靶点统计表格已保存到 output/target_activity_statistics.csv")

def create_feature_correlation_heatmap(original_data, feature_columns, selected_features_indices):
    """
    创建特征相关性热图
    """
    print("创建特征相关性热图...")
    
    try:
        # 获取选择的特征名称
        selected_feature_names = [feature_columns[i] for i in selected_features_indices]
        
        # 选择Top 20特征进行相关性分析（避免热图过于拥挤）
        top_features = selected_feature_names[:20] if len(selected_feature_names) > 20 else selected_feature_names
        
        # 计算特征相关性矩阵
        correlation_matrix = original_data[top_features].corr()
        
        # 创建相关性热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap (Top 20 Features)')
        plt.tight_layout()
        plt.savefig('output/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("特征相关性热图已保存到 output/feature_correlation_heatmap.png")
        
    except Exception as e:
        print(f"创建特征相关性热图时出错: {str(e)}")

def main():
    """
    主函数
    """
    print("开始集成模型整体特征重要性分析...")
    
    # 创建输出目录
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 分析特征重要性
    feature_importance_df = analyze_ensemble_overall_feature_importance()
    
    print("\n集成模型整体特征重要性分析完成！")
    print("生成的文件:")
    print("1. 整体特征重要性: output/ensemble_overall_feature_importance.csv")
    print("2. Top 30重要特征: output/top30_ensemble_features.csv")
    print("3. 特征重要性条形图: output/ensemble_feature_importance_barplot.png")
    print("4. 特征类型分析报告: output/feature_type_importance_analysis.csv")
    print("5. 各靶点Top 5重要特征: output/top5_features_*.csv")
    print("6. 各靶点Top 5高活性分子: output/top5_molecules_*.csv")
    print("7. Top分子预测活性热图: output/top_molecules_prediction_heatmap.png")
    print("8. 靶点统计热图: output/target_statistics_heatmap.png")
    print("9. 靶点活性统计: output/target_activity_statistics.csv")
    print("10. 特征相关性热图: output/feature_correlation_heatmap.png")

if __name__ == "__main__":
    main()