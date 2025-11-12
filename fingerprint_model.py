import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Tuple, Optional
import xgboost as xgb
import joblib
import json
import os


class FingerprintModel:
    """
    基于分子指纹的预测模型，针对小样本数据优化
    """

    def __init__(self, model_type: str = 'rf', fingerprint_type: str = 'morgan', **kwargs):
        """
        初始化指纹模型

        Args:
            model_type: 模型类型 ('rf' for Random Forest, 'xgb' for XGBoost)
            fingerprint_type: 指纹类型
            **kwargs: 模型参数
        """
        self.model_type = model_type
        self.fingerprint_type = fingerprint_type
        self.model = None
        self.target_columns = []
        self.feature_importances_ = None

        # 根据模型类型初始化模型
        if model_type == 'rf':
            # 随机森林对小样本数据更稳定
            n_estimators = kwargs.get('n_estimators', 200)  # 增加树的数量
            max_depth = kwargs.get('max_depth', 15)         # 增加最大深度
            min_samples_split = kwargs.get('min_samples_split', 3)
            min_samples_leaf = kwargs.get('min_samples_leaf', 1)
            
            base_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            self.model = MultiOutputRegressor(base_model)
        elif model_type == 'xgb':
            # XGBoost带正则化项，适合小样本
            n_estimators = kwargs.get('n_estimators', 200)  # 增加树的数量
            max_depth = kwargs.get('max_depth', 8)          # 增加最大深度
            learning_rate = kwargs.get('learning_rate', 0.05)
            subsample = kwargs.get('subsample', 0.8)
            reg_alpha = kwargs.get('reg_alpha', 0.1)
            reg_lambda = kwargs.get('reg_lambda', 1.0)
            
            base_model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42
            )
            self.model = MultiOutputRegressor(base_model)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, target_columns: List[str] = None) -> 'FingerprintModel':
        """
        训练模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值矩阵 (n_samples, n_targets)
            target_columns: 目标列名称

        Returns:
            训练好的模型
        """
        print(f"训练 {self.model_type} 模型...")
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")

        # 训练模型
        self.model.fit(X, y)
        
        # 保存目标列名
        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = [f"target_{i}" for i in range(y.shape[1])]
        
        # 获取特征重要性（如果是随机森林）
        if self.model_type == 'rf':
            importances = []
            for estimator in self.model.estimators_:
                importances.append(estimator.feature_importances_)
            self.feature_importances_ = np.mean(importances, axis=0)
        elif self.model_type == 'xgb':
            importances = []
            for estimator in self.model.estimators_:
                importances.append(estimator.feature_importances_)
            self.feature_importances_ = np.mean(importances, axis=0)
        
        print("模型训练完成")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测值矩阵 (n_samples, n_targets)
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型

        Args:
            X: 特征矩阵
            y: 真实目标值

        Returns:
            评估指标字典
        """
        predictions = self.predict(X)
        
        # 计算每个目标的指标
        metrics = {}
        for i, target in enumerate(self.target_columns):
            mse = mean_squared_error(y[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y[:, i], predictions[:, i])
            
            metrics[f"{target}_mse"] = mse
            metrics[f"{target}_rmse"] = rmse
            metrics[f"{target}_r2"] = r2
            
        # 计算总体指标
        total_mse = mean_squared_error(y, predictions)
        total_rmse = np.sqrt(total_mse)
        total_r2 = r2_score(y, predictions)
        
        metrics["total_mse"] = total_mse
        metrics["total_rmse"] = total_rmse
        metrics["total_r2"] = total_r2
        
        return metrics

    def save_model(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 模型保存路径
        """
        # 保存模型
        joblib.dump(self.model, filepath)
        
        # 保存元数据
        metadata = {
            'model_type': self.model_type,
            'fingerprint_type': self.fingerprint_type,
            'target_columns': self.target_columns
        }
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'FingerprintModel':
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
            model_type=metadata['model_type'],
            fingerprint_type=metadata['fingerprint_type']
        )
        
        # 加载训练好的模型
        model.model = joblib.load(filepath)
        model.target_columns = metadata['target_columns']
        
        return model