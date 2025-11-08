import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_extended_data():
    """
    加载并准备数据，使用扩展的RDKit特征（包括1024维Morgan指纹）
    """
    # 读取原始活性数据
    activity_df = pd.read_csv('data/activity_train.csv')
    
    # 读取扩展的RDKit分子特征
    features_df = pd.read_csv('data/molecular_features_extended.csv')
    
    # 合并数据
    merged_df = pd.merge(activity_df, features_df, on='molecule_id', how='left')
    
    # 对靶点进行编码
    le_target = LabelEncoder()
    merged_df['target_id_encoded'] = le_target.fit_transform(merged_df['target_id'])
    
    # 准备特征和目标变量
    feature_columns = [f'feature_{i}' for i in range(1074)]
    features = ['target_id_encoded'] + feature_columns
    X = merged_df[features]
    y = merged_df['pIC50']
    
    return X, y, le_target, merged_df

def build_final_optimized_model():
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
    ], weights=[1, 1.2, 1.3, 0.5])
    
    return ensemble_model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred

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
    feature_columns = [f'feature_{i}' for i in range(1074)]
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
    feature_columns = [f'feature_{i}' for i in range(1074)]
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
    
    print("Building final optimized ensemble model...")
    model = build_final_optimized_model()
    
    print("Training ensemble model...")
    model.fit(X_train, y_train)
    
    print("Evaluating ensemble model...")
    evaluate_model(model, X_test, y_test)
    
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