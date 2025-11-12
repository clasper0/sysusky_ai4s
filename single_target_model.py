"""
单目标优化模型
================

该模块实现了一个专门针对单个目标优化的模型，
用于提高特定靶点的pIC50值预测准确率。
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os


class SingleTargetModel:
    """
    单目标优化模型，专门优化单个靶点的预测性能
    """

    def __init__(self, target_name: str, fingerprint_dim: int = 2048, descriptor_dim: int = 25):
        """
        初始化单目标模型

        Args:
            target_name: 目标名称
            fingerprint_dim: 分子指纹维度
            descriptor_dim: 分子描述符维度
        """
        self.target_name = target_name
        self.fingerprint_dim = fingerprint_dim
        self.descriptor_dim = descriptor_dim
        self.models = {}
        self.weights = {}
        self.feature_selector = None
        self.is_fitted = False
        
        # 初始化基础模型
        self._initialize_base_models()

    def _initialize_base_models(self):
        """
        初始化基础模型，使用更适合单目标优化的参数
        """
        # 1. 随机森林模型 - 稳定且不易过拟合
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 2. XGBoost模型 - 强大的梯度提升算法
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        
        # 3. LightGBM模型 - 高效的梯度提升框架
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        
        # 4. Ridge回归 - 线性模型作为基准
        self.models['ridge'] = Ridge(alpha=1.0, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SingleTargetModel':
        """
        训练单目标模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值向量 (n_samples,)

        Returns:
            训练好的模型
        """
        print(f"训练单目标模型 {self.target_name}...")
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")
        
        # 特征选择 - 选择最重要的特征
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(1000, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        print(f"特征选择后维度: {X_selected.shape}")
        
        # 使用选定的特征训练各个基础模型
        model_scores = {}
        for name, model in self.models.items():
            print(f"  训练 {name} 模型...")
            
            # 训练模型
            model.fit(X_selected, y)
                
            # 使用交叉验证评估模型
            try:
                # 使用分层折叠交叉验证
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='r2')
                model_scores[name] = np.mean(cv_scores)
            except Exception as e:
                print(f"    交叉验证失败: {e}")
                # 如果交叉验证失败，使用简单验证
                model_scores[name] = 0.0
            
            print(f"    {name} 模型交叉验证 R2 分数: {model_scores[name]:.4f}")
        
        # 根据验证分数设置模型权重，确保权重为正
        # 将负分转换为0，然后归一化
        for name in model_scores:
            if model_scores[name] < 0:
                model_scores[name] = 0
        
        # 如果所有模型得分都是0，则平均分配权重
        total_score = sum(model_scores.values())
        if total_score <= 0:
            for name in model_scores:
                self.weights[name] = 1.0 / len(model_scores)
                print(f"{name} 模型权重: {self.weights[name]:.4f}")
        else:
            # 根据得分比例分配权重
            for name, score in model_scores.items():
                self.weights[name] = score / total_score
                print(f"{name} 模型权重: {self.weights[name]:.4f}")
        
        self.is_fitted = True
        print(f"单目标模型 {self.target_name} 训练完成")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测值向量 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 应用特征选择
        X_selected = self.feature_selector.transform(X)
        
        # 获取各个模型的预测结果
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_selected)
        
        # 加权集成预测结果
        final_prediction = np.zeros(X_selected.shape[0])
        for name, pred in predictions.items():
            final_prediction += self.weights[name] * pred
        
        return final_prediction

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        评估模型

        Args:
            X: 特征矩阵
            y: 真实目标值

        Returns:
            评估指标字典
        """
        predictions = self.predict(X)
        
        # 计算指标
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        return metrics

    def save_model(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 模型保存路径
        """
        # 创建模型保存目录
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存基础模型
        models_path = filepath.replace('.pkl', '_models.pkl')
        joblib.dump(self.models, models_path)
        
        # 保存特征选择器
        selector_path = filepath.replace('.pkl', '_selector.pkl')
        joblib.dump(self.feature_selector, selector_path)
        
        # 保存模型元数据
        metadata = {
            'target_name': self.target_name,
            'fingerprint_dim': self.fingerprint_dim,
            'descriptor_dim': self.descriptor_dim,
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'SingleTargetModel':
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            加载的模型
        """
        # 加载元数据
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 创建模型实例
        model = cls(
            target_name=metadata['target_name'],
            fingerprint_dim=metadata['fingerprint_dim'],
            descriptor_dim=metadata['descriptor_dim']
        )
        
        # 加载基础模型
        models_path = filepath.replace('.pkl', '_models.pkl')
        model.models = joblib.load(models_path)
        
        # 加载特征选择器
        selector_path = filepath.replace('.pkl', '_selector.pkl')
        model.feature_selector = joblib.load(selector_path)
        
        # 恢复其他属性
        model.weights = metadata['weights']
        model.is_fitted = metadata['is_fitted']
        
        return model