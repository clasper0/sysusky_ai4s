import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
import warnings
import optuna
import json
import os
warnings.filterwarnings('ignore')

def load_data():
    """
    加载训练数据和测试数据
    """
    # 读取训练数据
    activity_df = pd.read_csv('data/activity_train.csv')
    features_df = pd.read_csv('data/molecular_features_extended.csv')
    
    # 合并训练数据
    train_data = pd.merge(activity_df, features_df, on='molecule_id', how='left')
    
    # 读取测试数据（候选分子）
    test_features_df = pd.read_csv('data/candidate_features_extended.csv')
    
    return train_data, test_features_df

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

def load_best_params_from_json(filename='output/best_hyperparameters.json'):
    """
    从JSON文件加载最佳超参数
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            best_params = json.load(f)
        print(f"已从 {filename} 加载最佳超参数")
        return best_params
    else:
        print(f"未找到 {filename} 文件，将使用默认参数")
        return None

def train_model_with_params(X_train, y_train, best_params=None):
    """
    使用指定参数训练集成学习模型
    """
    if best_params is not None:
        # 使用加载的最佳参数
        rf_model = RandomForestRegressor(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            min_samples_split=best_params['rf_min_samples_split'],
            min_samples_leaf=best_params['rf_min_samples_leaf'],
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=best_params['gb_n_estimators'],
            max_depth=best_params['gb_max_depth'],
            learning_rate=best_params['gb_learning_rate'],
            random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=best_params['xgb_n_estimators'],
            max_depth=best_params['xgb_max_depth'],
            learning_rate=best_params['xgb_learning_rate'],
            random_state=42,
            n_jobs=-1
        )
        
        ridge_model = Ridge(alpha=best_params['ridge_alpha'])
    else:
        # 使用默认参数
        print("使用默认参数训练模型")
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        ridge_model = Ridge(alpha=1.0)
    
    # 创建集成模型
    ensemble_model = VotingRegressor([
        ('random_forest', rf_model),
        ('gradient_boosting', gb_model),
        ('xgboost', xgb_model),
        ('ridge', ridge_model)
    ])
    
    # 训练集成模型
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model

def train_model(X_train, y_train, use_optimization=False):
    """
    训练集成学习模型，包含多个基模型
    """
    # 定义多个基模型
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    gb_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    ridge_model = Ridge(alpha=1.0)
    
    # 创建集成模型
    ensemble_model = VotingRegressor([
        ('random_forest', rf_model),
        ('gradient_boosting', gb_model),
        ('xgboost', xgb_model),
        ('ridge', ridge_model)
    ])
    
    if use_optimization:
        # 网格搜索优化随机森林参数作为示例
        param_grid = {
            'random_forest__n_estimators': [100, 150],
            'random_forest__max_depth': [10, 15],
            'gradient_boosting__n_estimators': [100, 150],
            'xgboost__n_estimators': [100, 150]
        }
        grid_search = GridSearchCV(
            estimator=ensemble_model, 
            param_grid=param_grid, 
            cv=3, 
            n_jobs=-1, 
            verbose=2,
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        ensemble_model = grid_search.best_estimator_
    else:
        # 训练集成模型
        ensemble_model.fit(X_train, y_train)
    
    return ensemble_model

def save_best_params_to_json(best_params, filename='output/best_hyperparameters.json'):
    """
    将最佳超参数保存到JSON文件
    """
    # 确保output目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 保存参数到文件
    with open(filename, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"最佳超参数已保存到: {filename}")

def perform_cross_validation(model, X, y, cv=5):
    """
    执行交叉验证评估模型性能
    """
    print("进行交叉验证评估...")
    
    # 计算各种指标的交叉验证分数
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    rmse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    # 转换为正数
    mae_scores = -mae_scores
    rmse_scores = -rmse_scores
    
    print(f"交叉验证 MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std() * 2:.4f})")
    print(f"交叉验证 RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    print(f"交叉验证 R²: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    
    return {
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std()
    }

def optimize_model_with_optuna(X_train, y_train, n_trials=50):
    """
    使用Optuna优化模型超参数
    """
    def objective(trial):
        # 定义超参数搜索空间
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 100, 300, step=50)
        rf_max_depth = trial.suggest_int('rf_max_depth', 5, 20)
        rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
        rf_min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 5)
        
        gb_n_estimators = trial.suggest_int('gb_n_estimators', 100, 300, step=50)
        gb_max_depth = trial.suggest_int('gb_max_depth', 3, 10)
        gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
        
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 100, 300, step=50)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
        xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
        
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.1, 10.0, log=True)
        
        # 创建模型
        rf_model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            max_depth=gb_max_depth,
            learning_rate=gb_learning_rate,
            random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        ridge_model = Ridge(alpha=ridge_alpha)
        
        # 创建集成模型
        ensemble_model = VotingRegressor([
            ('random_forest', rf_model),
            ('gradient_boosting', gb_model),
            ('xgboost', xgb_model),
            ('ridge', ridge_model)
        ])
        
        # 训练模型
        ensemble_model.fit(X_train, y_train)
        
        return ensemble_model
    
    def objective_function(trial):
        model = objective(trial)
        # 使用交叉验证评估模型
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        return -scores.mean()  # Optuna最小化目标函数，所以需要取负值
    
    # 创建Optuna研究对象
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_function, n_trials=n_trials)
    
    # 获取最佳参数
    best_params = study.best_params
    print("最佳参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"最佳RMSE: {np.sqrt(study.best_value):.4f}")
    
    # 保存最佳参数到文件
    save_best_params_to_json(best_params)
    
    # 使用最佳参数训练最终模型
    best_rf_model = RandomForestRegressor(
        n_estimators=best_params['rf_n_estimators'],
        max_depth=best_params['rf_max_depth'],
        min_samples_split=best_params['rf_min_samples_split'],
        min_samples_leaf=best_params['rf_min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    
    best_gb_model = GradientBoostingRegressor(
        n_estimators=best_params['gb_n_estimators'],
        max_depth=best_params['gb_max_depth'],
        learning_rate=best_params['gb_learning_rate'],
        random_state=42
    )
    
    best_xgb_model = xgb.XGBRegressor(
        n_estimators=best_params['xgb_n_estimators'],
        max_depth=best_params['xgb_max_depth'],
        learning_rate=best_params['xgb_learning_rate'],
        random_state=42,
        n_jobs=-1
    )
    
    best_ridge_model = Ridge(alpha=best_params['ridge_alpha'])
    
    best_ensemble_model = VotingRegressor([
        ('random_forest', best_rf_model),
        ('gradient_boosting', best_gb_model),
        ('xgboost', best_xgb_model),
        ('ridge', best_ridge_model)
    ])
    
    best_ensemble_model.fit(X_train, y_train)
    
    return best_ensemble_model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # 添加R2分数
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    # 计算相关系数
    pearson_corr, pearson_p = pearsonr(y_test, y_pred)
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)
    
    print("模型评估结果:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation Coefficient: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    return y_pred, mae, rmse, r2, pearson_corr, spearman_corr

def predict_test_set(model, train_le_target, train_data, test_features_df):
    """
    预测测试集数据
    """
    # 获取训练数据中的靶点ID
    target_ids = train_data['target_id'].unique()
    
    predictions = []
    
    # 对每个候选分子和每个靶点进行预测
    for _, candidate in test_features_df.iterrows():
        for target_id in target_ids:
            # 构造特征
            try:
                target_encoded = train_le_target.transform([target_id])[0]
            except ValueError:
                # 如果靶点不在训练数据中，跳过
                continue
            
            # 提取分子特征
            feature_columns = [f'feature_{i}' for i in range(1074)]
            features = candidate[feature_columns].values
            
            # 构造输入向量
            input_features = np.array([[target_encoded] + list(features)])
            
            # 预测
            pred_pIC50 = model.predict(input_features)[0]
            
            # 保存预测结果
            predictions.append({
                'molecule_id': candidate['molecule_id'],
                'target_id': target_id,
                'predicted_pIC50': pred_pIC50
            })
    
    # 转换为DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # 按分子ID和预测pIC50值排序
    pred_df = pred_df.sort_values(['molecule_id', 'predicted_pIC50'], ascending=[True, False])
    
    return pred_df

def compare_models(X_train, y_train, X_test, y_test):
    """
    比较不同的模型
    """
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    results = {}
    
    print("比较不同模型的性能:")
    print("-" * 50)
    
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        }
        
        print(f"{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Pearson: {pearson_corr:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print()
    
    # 添加集成模型
    ensemble_model = train_model(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    
    results['Ensemble Model'] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2,
        'Pearson': pearson_corr,
        'Spearman': spearman_corr
    }
    
    print("Ensemble Model:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Pearson: {pearson_corr:.4f}")
    print(f"  Spearman: {spearman_corr:.4f}")
    print()
    
    return results

def main(use_optimization=False, compare_models_flag=False, use_optuna=False, use_cv=False, use_best_params=False):
    """
    主函数
    """
    print("加载数据...")
    train_data, test_features_df = load_data()
    
    print("准备特征...")
    X, y, le_target = prepare_features(train_data)
    
    print("分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if compare_models_flag:
        print("比较不同模型...")
        model_results = compare_models(X_train, y_train, X_test, y_test)
        
        # 保存模型比较结果
        results_df = pd.DataFrame(model_results).T
        if not os.path.exists('output'):
            os.makedirs('output')
        results_df.to_csv('output/model_comparison_results.csv')
        print("模型比较结果已保存到: output/model_comparison_results.csv")
    
    print("训练模型...")
    if use_optuna:
        print("使用Optuna优化模型超参数...")
        model = optimize_model_with_optuna(X_train, y_train)
    elif use_best_params:
        print("使用之前优化的最佳参数训练模型...")
        best_params = load_best_params_from_json()
        model = train_model_with_params(X_train, y_train, best_params)
    else:
        model = train_model(X_train, y_train, use_optimization)
    
    # 如果启用交叉验证，则在整个数据集上进行
    if use_cv:
        cv_results = perform_cross_validation(model, X, y)
        
        # 保存交叉验证结果
        cv_df = pd.DataFrame([cv_results])
        cv_df.to_csv('output/cross_validation_results.csv', index=False)
        print("交叉验证结果已保存到: output/cross_validation_results.csv")
    
    print("评估模型...")
    y_pred, mae, rmse, r2, pearson_corr, spearman_corr = evaluate_model(model, X_test, y_test)
    
    print("预测测试集...")
    test_predictions = predict_test_set(model, le_target, train_data, test_features_df)
    
    # 保存预测结果
    if not os.path.exists('output'):
        os.makedirs('output')
    
    test_predictions.to_csv('output/test_set_predictions.csv', index=False)
    print("测试集预测结果已保存到: output/test_set_predictions.csv")
    
    # 保存模型评估指标
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2,
        'Pearson Correlation': pearson_corr,
        'Spearman Correlation': spearman_corr
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('output/model_evaluation_metrics.csv', index=False)
    print("模型评估指标已保存到: output/model_evaluation_metrics.csv")
    
    return model, test_predictions, metrics

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    use_optimization = '--optimize' in sys.argv
    compare_models_flag = '--compare' in sys.argv
    use_optuna = '--optuna' in sys.argv
    use_cv = '--cv' in sys.argv
    use_best_params = '--best-params' in sys.argv
    
    print("参数设置:")
    print(f"  超参数优化: {use_optimization}")
    print(f"  模型比较: {compare_models_flag}")
    print(f"  Optuna优化: {use_optuna}")
    print(f"  交叉验证: {use_cv}")
    print(f"  使用最佳参数: {use_best_params}")
    print()
    
    model, predictions, metrics = main(use_optimization, compare_models_flag, use_optuna, use_cv, use_best_params)
