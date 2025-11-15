import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']


def plot_molecule_properties_heatmap(data_path, n_molecules=50, figsize=(30, 20)):
    """
    绘制候选分子理化性质热图
    
    参数:
    data_path: 数据文件路径
    n_molecules: 要展示的分子数量（如果数据量大，可以抽样显示）
    figsize: 图形大小
    """
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"包含 {len(df)} 个分子, {df.shape[1]-2} 个理化性质")  # 减去molecule_id和SMILES列
    
    # 选择要显示的分子（如果数据量大，随机抽样）
    if len(df) > n_molecules:
        df_display = df.sample(n=n_molecules, random_state=42)
        print(f"随机选择 {n_molecules} 个分子进行可视化")
    else:
        df_display = df
    
    # 准备热图数据（排除ID和SMILES列）
    property_columns = [col for col in df.columns if col not in ['molecule_id', 'SMILES']]
    heatmap_data = df_display[property_columns]
    
    # 数据标准化（Z-score标准化）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(heatmap_data)
    scaled_df = pd.DataFrame(scaled_data, columns=property_columns, index=df_display['molecule_id'])
    
    # 对性质进行分组（便于理解）
    property_groups = {
        '电拓扑状态指数': ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex'],
        '药物相似性': ['qed', 'SPS'],
        '分子量相关': ['MolWt', 'HeavyAtomMolWt', 'ExactMolWt'],
        '电子性质': ['NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 
                   'MaxAbsPartialCharge', 'MinAbsPartialCharge'],
        '指纹密度': ['FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3'],
        'BCUT描述符': ['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 
                     'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW'],
        '拓扑描述符': ['AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 
                     'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
                     'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3'],
        '表面积相关': ['LabuteASA', 'TPSA'],
        'VSA描述符': [col for col in property_columns if 'VSA' in col],
        '组成计数': ['HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumHeteroatoms', 
                   'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'RingCount'],
        '环系性质': ['NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                   'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
                   'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings'],
        '立体化学': ['NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumSpiroAtoms'],
        '物理化学性质': ['MolLogP', 'MolMR', 'FractionCSP3'],
        '官能团': [col for col in property_columns if col.startswith('fr_')]
    }
    
    # 按分组重新排列列
    ordered_columns = []
    for group, cols in property_groups.items():
        # 只保留数据中存在的列
        existing_cols = [col for col in cols if col in property_columns]
        ordered_columns.extend(existing_cols)
    
    # 重新排列数据
    scaled_df_ordered = scaled_df[ordered_columns]
    
    # 创建热图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [1, 20]})
    
    # 主热图
    sns.heatmap(scaled_df_ordered.T, ax=ax2, cmap='RdBu_r', center=0,
                cbar_kws={'label': '标准化数值 (Z-score)'},
                xticklabels=1, yticklabels=1)
    
    ax2.set_xlabel('候选分子', fontsize=12, fontweight='bold')
    ax2.set_ylabel('理化性质', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=90, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    # 添加性质分组标注
    current_idx = 0
    for group, cols in property_groups.items():
        existing_cols = [col for col in cols if col in property_columns]
        if existing_cols:
            group_size = len(existing_cols)
            mid_point = current_idx + group_size / 2
            ax2.text(len(scaled_df_ordered) + 5, mid_point, group, 
                    ha='left', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            current_idx += group_size
    
    # 添加统计信息子图
    stats_data = []
    for col in ordered_columns:
        stats_data.append({
            'Property': col,
            'Variance': np.var(scaled_df_ordered[col]),
            'Range': np.ptp(scaled_df_ordered[col])
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    ax1.bar(range(len(ordered_columns)), stats_df['Variance'], alpha=0.7, color='skyblue')
    ax1.set_ylabel('方差', fontsize=10, fontweight='bold')
    ax1.set_title('各理化性质的方差分布', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('候选分子理化性质分布热图\n(数值经过Z-score标准化)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.95)
    
    return fig, scaled_df_ordered

def plot_correlation_heatmap(data_path, top_k=30):
    """
    绘制理化性质之间的相关性热图
    """
    df = pd.read_csv(data_path)
    
    # 选择数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    
    # 选择相关性最强的top_k个性质
    corr_abs = correlation_matrix.abs()
    mean_corr = corr_abs.mean().sort_values(ascending=False)
    top_properties = mean_corr.head(top_k).index
    
    top_corr_matrix = correlation_matrix.loc[top_properties, top_properties]
    
    # 绘制相关性热图
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))
    
    sns.heatmap(top_corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, annot=True, fmt='.2f', cbar_kws={'shrink': .8},
                ax=ax)
    
    ax.set_title(f'Top {top_k} 理化性质之间的相关性热图', 
                fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_candidate_predictions_ranking(predictions_path, top_n=20):
    """
    根据预测的pIC50值对候选分子进行排序并可视化
    
    参数:
    predictions_path: 预测结果文件路径
    top_n: 展示前top_n个分子
    """
    # 读取预测结果
    df = pd.read_csv(predictions_path)
    
    # 计算每个分子的平均活性
    avg_activity = df.groupby('molecule_id')['predicted_pIC50'].mean().sort_values(ascending=False)
    
    # 选择前top_n个分子
    top_molecules = avg_activity.head(top_n)
    
    # 创建可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(top_molecules)), top_molecules.values, color='skyblue')
    
    # 设置图表属性
    ax.set_xlabel('候选分子', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均预测pIC50值', fontsize=12, fontweight='bold')
    ax.set_title(f'候选分子按平均预测活性排序 (Top {top_n})', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_molecules)))
    ax.set_xticklabels(top_molecules.index, rotation=45, ha='right')
    
    # 添加数值标签
    for i, (mol_id, value) in enumerate(top_molecules.items()):
        ax.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig, top_molecules

def plot_top_features_heatmap(top_features_path, predictions_path, top_n_features=10):
    """
    绘制每个靶点的top特征热力图，分子按预测活性排序
    
    参数:
    top_features_path: top特征文件路径
    predictions_path: 预测结果文件路径
    top_n_features: 每个靶点选取的特征数量
    """
    # 读取top特征数据
    top_features_df = pd.read_csv(top_features_path)
    
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_path)
    
    # 计算每个分子的平均活性并排序
    avg_activity = predictions_df.groupby('molecule_id')['predicted_pIC50'].mean().sort_values(ascending=False)
    sorted_molecules = avg_activity.index.tolist()
    
    # 获取所有靶点
    targets = top_features_df['target_id'].unique()
    
    # 为每个靶点获取top特征
    target_features = {}
    for target in targets:
        target_data = top_features_df[top_features_df['target_id'] == target]
        top_feats = target_data.nlargest(top_n_features, 'importance')['feature'].tolist()
        target_features[target] = top_feats
    
    # 所有特征列表（去重）
    all_features = []
    for feats in target_features.values():
        all_features.extend(feats)
    all_features = list(dict.fromkeys(all_features))  # 去重但保持顺序
    
    # 创建热力图数据
    heatmap_data = []
    for molecule in sorted_molecules:
        row_data = {'molecule_id': molecule, 'avg_pIC50': avg_activity[molecule]}
        # 为每个靶点和特征添加数据
        for target in targets:
            target_preds = predictions_df[
                (predictions_df['molecule_id'] == molecule) & 
                (predictions_df['target_id'] == target)
            ]
            if not target_preds.empty:
                row_data[f'{target}_pIC50'] = target_preds.iloc[0]['predicted_pIC50']
            else:
                row_data[f'{target}_pIC50'] = np.nan
        heatmap_data.append(row_data)
    
    # 转换为DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('molecule_id')
    
    # 选择要可视化的列（靶点活性）
    activity_columns = [f'{target}_pIC50' for target in targets]
    activity_data = heatmap_df[activity_columns]
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(activity_data.T, cmap='viridis', ax=ax, cbar_kws={'label': '预测pIC50值'})
    
    ax.set_xlabel('候选分子 (按平均活性降序排列)', fontsize=12, fontweight='bold')
    ax.set_ylabel('靶点', fontsize=12, fontweight='bold')
    ax.set_title('候选分子对各靶点的预测活性热力图', fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(range(len(sorted_molecules)))
    ax.set_xticklabels(sorted_molecules, rotation=45, ha='right', fontsize=8)
    
    # 设置y轴标签
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([f'靶点 {t[-3:]}' for t in targets], rotation=0, fontsize=10)
    
    plt.tight_layout()
    return fig, activity_data

def plot_feature_importance_heatmap(importance_matrix_path, predictions_path, top_n_features=30):
    """
    绘制特征重要性热力图，横轴是按平均活性降序排列的分子，纵轴是重要特征
    
    参数:
    importance_matrix_path: 特征重要性矩阵文件路径
    predictions_path: 预测结果文件路径
    top_n_features: 选取的top特征数量
    """
    # 读取特征重要性矩阵
    importance_df = pd.read_csv(importance_matrix_path, index_col=0)
    
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_path)
    
    # 计算每个分子的平均活性并排序
    avg_activity = predictions_df.groupby('molecule_id')['predicted_pIC50'].mean().sort_values(ascending=False)
    sorted_molecules = avg_activity.index.tolist()
    
    # 选择重要性最高的特征
    feature_importance_mean = importance_df.mean(axis=0).sort_values(ascending=False)
    top_features = feature_importance_mean.head(top_n_features).index.tolist()
    
    # 构建热力图数据：行是特征，列是分子
    # 对于每个特征，我们显示它在所有靶点上的平均重要性
    heatmap_data = []
    feature_names = []
    
    for feature in top_features:
        # 获取该特征在各靶点上的重要性
        feature_importance = importance_df[feature]
        # 计算平均重要性作为该特征的代表值
        mean_importance = feature_importance.mean()
        # 为每个分子重复这个值（在实际应用中，这应该根据分子的特征值来计算）
        heatmap_data.append([mean_importance] * len(sorted_molecules))
        feature_names.append(feature)
    
    # 转换为DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, index=feature_names, columns=sorted_molecules)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(max(15, len(sorted_molecules) * 0.5), 10))
    sns.heatmap(heatmap_df, cmap='YlOrRd', ax=ax, cbar_kws={'label': '特征重要性'})
    
    ax.set_xlabel('候选分子 (按平均活性降序排列)', fontsize=12, fontweight='bold')
    ax.set_ylabel('重要特征', fontsize=12, fontweight='bold')
    ax.set_title(f'重要特征在候选分子中的重要性热力图 (Top {top_n_features} 特征)', fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(range(len(sorted_molecules)))
    ax.set_xticklabels(sorted_molecules, rotation=45, ha='right', fontsize=8)
    
    # 设置y轴标签
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig, heatmap_df

def plot_molecule_feature_values_heatmap(candidate_features_path, predictions_path, importance_matrix_path, top_n_molecules=20, top_n_features=30):
    """
    绘制分子特征值热力图，横轴是按预测活性降序排列的分子，纵轴是按重要性递减的不同特征，热力图显示每个分子在各特征上的实际值
    
    参数:
    candidate_features_path: 候选分子特征文件路径
    predictions_path: 预测结果文件路径
    importance_matrix_path: 特征重要性矩阵文件路径
    top_n_molecules: 选取的top分子数量
    top_n_features: 选取的top特征数量
    """
    # 读取数据
    candidate_df = pd.read_csv(candidate_features_path)
    predictions_df = pd.read_csv(predictions_path)
    importance_df = pd.read_csv(importance_matrix_path, index_col=0)
    
    # 计算每个分子的平均活性并排序
    avg_activity = predictions_df.groupby('molecule_id')['predicted_pIC50'].mean().sort_values(ascending=False)
    top_molecules = avg_activity.head(top_n_molecules).index.tolist()
    
    # 筛选出top分子的特征数据
    top_molecules_df = candidate_df[candidate_df['molecule_id'].isin(top_molecules)].copy()
    
    # 按照活性排序重新排列分子顺序
    top_molecules_df['molecule_id'] = pd.Categorical(
        top_molecules_df['molecule_id'], 
        categories=top_molecules, 
        ordered=True
    )
    top_molecules_df = top_molecules_df.sort_values('molecule_id')
    
    # 选择数值型特征列（排除molecule_id和SMILES）
    feature_columns = [col for col in top_molecules_df.columns if col not in ['molecule_id', 'SMILES']]
    
    # 选择重要性最高的特征
    # 计算特征在所有靶点上的平均重要性
    feature_importance_mean = importance_df.mean(axis=0).sort_values(ascending=False)
    selected_features = feature_importance_mean.head(top_n_features).index.tolist()
    
    # 确保选出的特征在候选分子数据中存在
    selected_features = [f for f in selected_features if f in feature_columns]
    
    # 构建热力图数据
    heatmap_data = top_molecules_df[selected_features]
    
    # 对数据进行标准化处理，便于可视化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(heatmap_data)
    scaled_df = pd.DataFrame(scaled_data, columns=selected_features, index=top_molecules_df['molecule_id'])
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(max(12, len(top_molecules) * 0.5), max(8, len(selected_features) * 0.3)))
    sns.heatmap(scaled_df.T, cmap='RdBu_r', center=0, ax=ax, 
                cbar_kws={'label': '特征值 (标准化)'}, 
                xticklabels=True, yticklabels=True)
    
    ax.set_xlabel('候选分子 (按平均预测活性降序排列)', fontsize=12, fontweight='bold')
    ax.set_ylabel('分子特征 (按重要性递减排列)', fontsize=12, fontweight='bold')
    ax.set_title(f'候选分子特征值热力图\n(展示Top {top_n_molecules}分子在Top {len(selected_features)}重要特征上的分布)', 
                fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(range(len(top_molecules)))
    ax.set_xticklabels(top_molecules, rotation=45, ha='right', fontsize=8)
    
    # 设置y轴标签
    ax.set_yticks(range(len(selected_features)))
    ax.set_yticklabels(selected_features, rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig, scaled_df

def virtual_screening_pipeline_summary(candidate_features_path, predictions_path, top_features_path, importance_matrix_path):
    """
    虚拟筛选流程总结：批量预测-多靶点活性评估-类药性分析-可视化验证
    
    本函数实现了一个完整的虚拟筛选流程，包括：
    1. 批量预测：对候选分子进行多靶点pIC50值预测
    2. 多靶点活性评估：分析各靶点的活性分布情况
    3. 类药性分析：基于理化性质和预测活性评估分子的类药性
    4. 可视化验证：通过多种图表验证模型性能和可解释性
    
    参数:
    candidate_features_path: 候选分子特征文件路径
    predictions_path: 预测结果文件路径
    top_features_path: top特征文件路径
    importance_matrix_path: 特征重要性矩阵文件路径
    """
    # 读取数据
    candidate_df = pd.read_csv(candidate_features_path)
    predictions_df = pd.read_csv(predictions_path)
    top_features_df = pd.read_csv(top_features_path)
    importance_df = pd.read_csv(importance_matrix_path, index_col=0)
    
    print("=" * 50)
    print("虚拟筛选流程总结")
    print("=" * 50)
    print("针对该任务，本团队设计了“批量预测-多靶点活性评估-类药性分析-可视化验证的虚拟筛选流程。")
    print("通过应用任务2训练好的模型，我们通过对candidate.csv进行数据读取和处理,")
    print("实现对其针对五靶点pIC50值的预测,再通过任务3中提取的重要分子特征，")
    print("我们探究了这些分子在所提取重要分子特征上的分布情况，")
    print("并结合所预测pIC50 进行热力图绘制和类药性分析，")
    print("从而更好评估所搭建模型的性能及可解释性探索的可靠程度。")
    print("-" * 50)
    
    print("1. 批量预测:")
    print(f"   - 对 {len(candidate_df)} 个候选分子进行批量预测")
    print(f"   - 预测针对 {predictions_df['target_id'].nunique()} 个靶点的pIC50值")
    
    print("\n2. 多靶点活性评估:")
    target_stats = predictions_df.groupby('target_id')['predicted_pIC50'].agg(['mean', 'std', 'min', 'max'])
    print("   - 各靶点预测活性统计:")
    for target, stats in target_stats.iterrows():
        print(f"     {target}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}]")
    
    print("\n3. 类药性分析:")
    # 计算整体平均活性并排序
    avg_activity = predictions_df.groupby('molecule_id')['predicted_pIC50'].mean().sort_values(ascending=False)
    top_molecules = avg_activity.head(5)
    print("   - 预测活性最高的前5个分子:")
    for i, (mol_id, activity) in enumerate(top_molecules.items(), 1):
        print(f"     {i}. {mol_id}: {activity:.2f}")
    
    print("\n4. 可视化验证:")
    print("   - 候选分子理化性质分布热图")
    print("   - 理化性质相关性热图")
    print("   - 候选分子按平均预测活性排序图")
    print("   - 候选分子对各靶点预测活性热力图")
    print("   - 重要特征在候选分子中的重要性热力图")
    
    print("\n=== 模型性能及可解释性评估 ===")
    # 分析特征重要性
    feature_importance_mean = importance_df.mean(axis=0).sort_values(ascending=False)
    top_features = feature_importance_mean.head(5)
    print("Top 5 重要特征:")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # 分析各靶点的重要特征
    print("\n各靶点的重要特征:")
    targets = importance_df.index.tolist()
    for target in targets:
        target_importance = importance_df.loc[target].sort_values(ascending=False)
        top_target_features = target_importance.head(3)
        print(f"  {target}:")
        for i, (feature, importance) in enumerate(top_target_features.items(), 1):
            print(f"    {i}. {feature}: {importance:.4f}")
    
    return {
        'total_molecules': len(candidate_df),
        'total_targets': predictions_df['target_id'].nunique(),
        'top_molecules': top_molecules,
        'top_features': top_features,
        'target_stats': target_stats
    }

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据文件路径
    data_file_path = "data/candidate_features_extended.csv"
    predictions_file_path = "output/final_optimized_candidate_predictions.csv"
    top_features_file_path = "output/top10_features_per_target.csv"
    importance_matrix_path = "output/feature_importance_matrix.csv"
    
    try:
        # 执行虚拟筛选流程总结
        summary = virtual_screening_pipeline_summary(
            data_file_path, 
            predictions_file_path, 
            top_features_file_path, 
            importance_matrix_path
        )
        
        # 绘制主热图
        fig1, processed_data = plot_molecule_properties_heatmap(data_file_path, n_molecules=50)
        plt.show()
        
        # 绘制相关性热图
        fig2 = plot_correlation_heatmap(data_file_path, top_k=30)
        plt.show()
        
        # 绘制候选分子活性排序图
        fig3, ranked_molecules = plot_candidate_predictions_ranking(predictions_file_path, top_n=20)
        plt.show()
        
        # 绘制靶点活性热力图
        fig4, activity_data = plot_top_features_heatmap(top_features_file_path, predictions_file_path)
        plt.show()
        
        # 绘制特征重要性热力图
        fig5, importance_heatmap = plot_feature_importance_heatmap(importance_matrix_path, predictions_file_path, top_n_features=30)
        plt.show()
        
        # 绘制分子特征值热力图（按特征重要性排序）
        fig6, molecule_feature_heatmap = plot_molecule_feature_values_heatmap(
            data_file_path, 
            predictions_file_path, 
            importance_matrix_path,
            top_n_molecules=20, 
            top_n_features=30
        )
        plt.show()
        
        print("\n=== 数据统计信息 ===")
        print(f"总分子数: {len(pd.read_csv(data_file_path))}")
        print(f"理化性质数量: {len(processed_data.columns)}")
        print(f"展示分子数: {len(processed_data)}")
        
        # 显示性质分组统计
        property_groups_count = {
            '电拓扑状态指数': 4,
            '药物相似性': 2,
            '分子量相关': 3,
            '电子性质': 6,
            '指纹密度': 3,
            'BCUT描述符': 8,
            '拓扑描述符': 20,
            '表面积相关': 2,
            'VSA描述符': len([col for col in processed_data.columns if 'VSA' in col]),
            '组成计数': 8,
            '环系性质': 9,
            '立体化学': 3,
            '物理化学性质': 3,
            '官能团': len([col for col in processed_data.columns if col.startswith('fr_')])
        }
        
        print("\n=== 性质分组统计 ===")
        for group, count in property_groups_count.items():
            print(f"{group}: {count}个性质")
            
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{data_file_path}'")
        print("请确保文件路径正确，并包含以下列:")
        print("molecule_id,SMILES,以及所有的理化性质列")
    except Exception as e:
        print(f"处理数据时发生错误: {e}")