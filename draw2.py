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

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据文件路径
    data_file_path = "data\candidate_features_extended.csv"  # 请替换为实际文件路径
    
    try:
        # 绘制主热图
        fig1, processed_data = plot_molecule_properties_heatmap(data_file_path, n_molecules=50)
        plt.show()
        
        # 绘制相关性热图
        fig2 = plot_correlation_heatmap(data_file_path, top_k=30)
        plt.show()
        
        # 显示数据统计信息
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