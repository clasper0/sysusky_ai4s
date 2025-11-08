import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    加载训练数据
    """
    # 读取训练数据
    activity_df = pd.read_csv('data/activity_train.csv')
    features_df = pd.read_csv('data/molecular_features_extended.csv')
    
    # 合并训练数据
    train_data = pd.merge(activity_df, features_df, on='molecule_id', how='left')
    
    return train_data

def prepare_features(data):
    """
    准备特征数据
    """
    # 对靶点进行编码
    le_target = LabelEncoder()
    data['target_id_encoded'] = le_target.fit_transform(data['target_id'])
    
    # 准备特征列
    feature_columns = [f'feature_{i}' for i in range(1074)]
    features = ['target_id_encoded'] + feature_columns
    
    X = data[features]
    y = data['pIC50']
    
    return X, y, le_target

def analyze_feature_importance_by_target(X, y, le_target, train_data):
    """
    分析每个靶点的特征重要性
    """
    target_ids = train_data['target_id'].unique()
    
    # 创建一个字典来存储每个靶点的特征重要性
    target_importances = {}
    
    print("分析各靶点的特征重要性...")
    
    for target_id in target_ids:
        print(f"正在分析靶点 {target_id}...")
        
        # 筛选特定靶点的数据
        target_mask = train_data['target_id'] == target_id
        target_data = train_data[target_mask]
        
        if len(target_data) < 10:  # 如果数据太少，跳过
            print(f"  跳过靶点 {target_id}，数据量不足")
            continue
            
        # 准备特征
        feature_columns = [f'feature_{i}' for i in range(1074)]
        features = feature_columns
        X_target = target_data[features]
        y_target = target_data['pIC50']
        
        # 训练随机森林模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_target, y_target)
        
        # 获取特征重要性
        importances = rf_model.feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        target_importances[target_id] = feature_importance_df
        
        # 保存Top 10重要特征
        top_10_features = feature_importance_df.head(10)
        top_10_features.to_csv(f'output/top10_features_{target_id}.csv', index=False)
        print(f"  靶点 {target_id} 的Top 10重要特征已保存到 output/top10_features_{target_id}.csv")
    
    return target_importances

def create_feature_importance_heatmap(target_importances, top_n=20):
    """
    创建特征重要性热力图
    """
    print("创建特征重要性热力图...")
    
    # 获取所有靶点的Top N特征
    all_top_features = set()
    for target_id, importance_df in target_importances.items():
        top_features = importance_df.head(top_n)['feature'].tolist()
        all_top_features.update(top_features)
    
    # 创建热力图数据
    heatmap_data = []
    heatmap_index = []
    
    for target_id, importance_df in target_importances.items():
        heatmap_index.append(target_id)
        feature_dict = dict(zip(importance_df['feature'], importance_df['importance']))
        
        row_data = []
        for feature in all_top_features:
            row_data.append(feature_dict.get(feature, 0))
        
        heatmap_data.append(row_data)
    
    # 转换为DataFrame
    heatmap_df = pd.DataFrame(
        heatmap_data, 
        columns=list(all_top_features), 
        index=heatmap_index
    )
    
    # 绘制热力图
    plt.figure(figsize=(20, 8))
    sns.heatmap(heatmap_df, annot=False, cmap='YlOrRd', cbar=True)
    plt.title('Top Feature Importances by Target')
    plt.xlabel('Features')
    plt.ylabel('Targets')
    plt.tight_layout()
    plt.savefig('output/feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("特征重要性热力图已保存到 output/feature_importance_heatmap.png")
    
    return heatmap_df

def analyze_high_activity_molecules(train_data, target_importances, top_n=5):
    """
    分析高活性分子的特征
    """
    print("分析高活性分子的特征...")
    
    target_ids = train_data['target_id'].unique()
    
    for target_id in target_ids:
        if target_id not in target_importances:
            continue
            
        print(f"正在分析靶点 {target_id} 的高活性分子...")
        
        # 筛选特定靶点的数据
        target_mask = train_data['target_id'] == target_id
        target_data = train_data[target_mask]
        
        # 按pIC50排序，获取高活性分子
        high_activity_molecules = target_data.nlargest(10, 'pIC50')
        
        # 保存高活性分子信息
        high_activity_molecules[['molecule_id', 'pIC50']].to_csv(
            f'output/high_activity_molecules_{target_id}.csv', 
            index=False
        )
        
        print(f"  靶点 {target_id} 的高活性分子已保存到 output/high_activity_molecules_{target_id}.csv")
        
        # 分析这些高活性分子的重要特征
        top_features = target_importances[target_id].head(top_n)
        
        # 获取这些分子的特征值
        feature_columns = [f'feature_{i}' for i in range(1074)]
        high_activity_features = high_activity_molecules[feature_columns]
        
        # 计算平均特征值
        mean_features = high_activity_features.mean()
        
        # 获取Top特征的值
        top_feature_values = {}
        for _, feature_row in top_features.iterrows():
            feature_name = feature_row['feature']
            feature_importance = feature_row['importance']
            feature_value = mean_features[feature_name]
            top_feature_values[feature_name] = {
                'importance': feature_importance,
                'value': feature_value
            }
        
        # 保存Top特征分析结果
        top_features_analysis = pd.DataFrame([
            {
                'feature': feature_name,
                'importance': values['importance'],
                'mean_value_in_high_activity': values['value']
            }
            for feature_name, values in top_feature_values.items()
        ])
        
        top_features_analysis.to_csv(
            f'output/high_activity_molecule_features_{target_id}.csv', 
            index=False
        )
        
        print(f"  靶点 {target_id} 的高活性分子特征分析已保存到 output/high_activity_molecule_features_{target_id}.csv")

def main():
    """
    主函数
    """
    print("开始特征重要性分析...")
    
    # 创建输出目录
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 加载数据
    print("加载数据...")
    train_data = load_data()
    
    # 准备特征
    print("准备特征...")
    X, y, le_target = prepare_features(train_data)
    
    # 分析各靶点的特征重要性
    target_importances = analyze_feature_importance_by_target(X, y, le_target, train_data)
    
    # 创建特征重要性热力图
    if target_importances:
        heatmap_df = create_feature_importance_heatmap(target_importances)
    
    # 分析高活性分子
    if target_importances:
        analyze_high_activity_molecules(train_data, target_importances)
    
    print("特征重要性分析完成！")
    print("生成的文件:")
    print("1. 各靶点的Top 10重要特征: output/top10_features_*.csv")
    print("2. 特征重要性热力图: output/feature_importance_heatmap.png")
    print("3. 高活性分子列表: output/high_activity_molecules_*.csv")
    print("4. 高活性分子特征分析: output/high_activity_molecule_features_*.csv")

if __name__ == "__main__":
    main()