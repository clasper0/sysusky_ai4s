import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class WeightedEnsembleModel:
    """
    加权集成学习模型，用于预测多个靶点的pIC50值
    """
    
    def __init__(self):
        """
        初始化模型
        """
        # 定义基础模型
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(random_state=42),
            'lgb': lgb.LGBMRegressor(random_state=42),
            'gb': GradientBoostingRegressor(random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'svr': SVR(kernel='rbf', C=1.0),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 为每个靶点存储模型和权重
        self.target_models = {}
        self.target_weights = {}
        self.target_names = ['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']
    
    def _calculate_model_weights(self, X, y, cv=3):
        """
        基于交叉验证得分计算模型权重
        
        Parameters:
        X: 特征矩阵
        y: 目标值
        cv: 交叉验证折数
        
        Returns:
        dict: 模型权重字典
        """
        weights = {}
        scores = {}
        
        for name, model in self.models.items():
            try:
                # 使用负MSE作为评分标准
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                avg_score = np.mean(cv_scores)
                scores[name] = avg_score
            except Exception as e:
                # 如果模型训练失败，给予权重0
                scores[name] = float('-inf')
        
        # 将得分转化为权重（正值）
        max_score = max(scores.values())
        shifted_scores = {name: score - max_score for name, score in scores.items()}
        
        # 计算softmax权重
        exp_scores = {name: np.exp(score) if score != float('-inf') else 0 for name, score in shifted_scores.items()}
        total_exp_score = sum(exp_scores.values())
        
        # 避免除零错误
        if total_exp_score == 0:
            weights = {name: 1.0 / len(self.models) for name in self.models}
        else:
            weights = {name: exp_score / total_exp_score for name, exp_score in exp_scores.items()}
        
        # 过滤掉权重非常小的模型
        weights = {name: weight if weight > 0.01 else 0 for name, weight in weights.items()}
        
        # 重新归一化权重
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {name: weight / weight_sum for name, weight in weights.items()}
        
        return weights
    
    def fit(self, X, y):
        """
        训练模型
        
        Parameters:
        X: 特征矩阵
        y: 目标值 (n_samples, n_targets)
        """
        n_targets = y.shape[1]
        
        for i in range(n_targets):
            target_name = self.target_names[i]
            y_target = y[:, i]
            
            print(f"训练 {target_name} 的模型...")
            
            # 计算模型权重
            weights = self._calculate_model_weights(X, y_target)
            self.target_weights[target_name] = weights
            
            # 训练各个模型
            trained_models = {}
            for name, model in self.models.items():
                if weights[name] > 0:  # 只训练有权重的模型
                    model.fit(X, y_target)
                    trained_models[name] = model
            
            self.target_models[target_name] = trained_models
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        X: 特征矩阵
        
        Returns:
        np.array: 预测结果 (n_samples, n_targets)
        """
        predictions = np.zeros((X.shape[0], len(self.target_names)))
        
        for i, target_name in enumerate(self.target_names):
            # 获取该目标的模型和权重
            models = self.target_models[target_name]
            weights = self.target_weights[target_name]
            
            # 加权预测
            weighted_pred = np.zeros(X.shape[0])
            total_weight = sum(weights.values())
            
            if total_weight > 0:
                for name, model in models.items():
                    pred = model.predict(X)
                    weighted_pred += weights[name] * pred
                
                # 归一化预测结果
                weighted_pred /= total_weight
            
            predictions[:, i] = weighted_pred
        
        return predictions
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Parameters:
        X: 特征矩阵
        y: 真实目标值
        
        Returns:
        dict: 评估结果
        """
        predictions = self.predict(X)
        
        results = {}
        for i, target_name in enumerate(self.target_names):
            y_true = y[:, i]
            y_pred = predictions[:, i]
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            results[target_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
        
        # 计算整体指标
        overall_mse = mean_squared_error(y, predictions)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(y, predictions)
        
        results['overall'] = {
            'MSE': overall_mse,
            'RMSE': overall_rmse,
            'R2': overall_r2
        }
        
        return results

class StackingEnsembleModel:
    """
    堆叠集成学习模型
    """
    
    def __init__(self):
        """
        初始化堆叠模型
        """
        # 基础模型
        self.base_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(random_state=42),
            'lgb': lgb.LGBMRegressor(random_state=42)
        }
        
        # 元模型（用于组合基础模型的预测）
        self.meta_model = Ridge(alpha=1.0)
        
        # 存储每个靶点的模型
        self.target_models = {}
        self.target_meta_models = {}
        self.target_names = ['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']
    
    def fit(self, X, y):
        """
        训练堆叠模型
        
        Parameters:
        X: 特征矩阵
        y: 目标值 (n_samples, n_targets)
        """
        from sklearn.model_selection import KFold
        
        n_targets = y.shape[1]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i in range(n_targets):
            target_name = self.target_names[i]
            y_target = y[:, i]
            
            print(f"训练 {target_name} 的堆叠模型...")
            
            # 存储基础模型的预测结果（用于训练元模型）
            base_predictions = np.zeros((X.shape[0], len(self.base_models)))
            
            # 训练基础模型并生成元特征
            trained_base_models = {}
            for j, (name, model) in enumerate(self.base_models.items()):
                # 使用交叉验证生成基础模型预测
                meta_features = np.zeros(X.shape[0])
                
                for train_idx, val_idx in kf.split(X):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold = y_target[train_idx]
                    
                    model_clone = type(model)(**model.get_params())
                    model_clone.fit(X_train_fold, y_train_fold)
                    meta_features[val_idx] = model_clone.predict(X_val_fold)
                
                base_predictions[:, j] = meta_features
                
                # 在全数据集上重新训练模型
                model.fit(X, y_target)
                trained_base_models[name] = model
            
            # 训练元模型
            meta_model = type(self.meta_model)(**self.meta_model.get_params())
            meta_model.fit(base_predictions, y_target)
            
            # 保存模型
            self.target_models[target_name] = trained_base_models
            self.target_meta_models[target_name] = meta_model
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        X: 特征矩阵
        
        Returns:
        np.array: 预测结果 (n_samples, n_targets)
        """
        predictions = np.zeros((X.shape[0], len(self.target_names)))
        
        for i, target_name in enumerate(self.target_names):
            # 获取该目标的基础模型
            base_models = self.target_models[target_name]
            
            # 生成基础模型的预测
            base_predictions = np.zeros((X.shape[0], len(base_models)))
            for j, (name, model) in enumerate(base_models.items()):
                base_predictions[:, j] = model.predict(X)
            
            # 使用元模型进行最终预测
            meta_model = self.target_meta_models[target_name]
            final_prediction = meta_model.predict(base_predictions)
            
            predictions[:, i] = final_prediction
        
        return predictions
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Parameters:
        X: 特征矩阵
        y: 真实目标值
        
        Returns:
        dict: 评估结果
        """
        predictions = self.predict(X)
        
        results = {}
        for i, target_name in enumerate(self.target_names):
            y_true = y[:, i]
            y_pred = predictions[:, i]
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            results[target_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
        
        # 计算整体指标
        overall_mse = mean_squared_error(y, predictions)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(y, predictions)
        
        results['overall'] = {
            'MSE': overall_mse,
            'RMSE': overall_rmse,
            'R2': overall_r2
        }
        
        return results
