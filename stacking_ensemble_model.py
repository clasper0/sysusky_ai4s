"""
堆叠集成学习模型
================

该模块实现了一个堆叠集成学习模型，结合多种基础模型的优势，
用于基于分子指纹和描述符的pIC50值预测。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
from typing import List, Dict, Tuple, Optional, Any


class StackingEnsembleModel:
    """
    堆叠集成学习模型，使用堆叠方法结合多种基础模型
    """

    def __init__(self, fingerprint_dim: int = 2048, descriptor_dim: int = 25, **kwargs):
        """
        初始化堆叠集成模型

        Args:
            fingerprint_dim: 分子指纹维度
            descriptor_dim: 分子描述符维度
            **kwargs: 其他参数
        """
        self.fingerprint_dim = fingerprint_dim
        self.descriptor_dim = descriptor_dim
        self.models = {}
        self.stacking_models = {}
        self.target_columns = []
        self.is_fitted = False

    def _create_base_models(self):
        """
        创建基础模型
        """
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )),
        ]
        return base_models

    def fit(self, X: np.ndarray, y: np.ndarray, target_columns: List[str] = None) -> 'StackingEnsembleModel':
        """
        训练堆叠集成模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值矩阵 (n_samples, n_targets)
            target_columns: 目标列名称

        Returns:
            训练好的模型
        """
        print(f"训练堆叠集成模型...")
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")
        
        # 保存目标列名
        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = [f"target_{i}" for i in range(y.shape[1])]
        
        # 为每个目标训练一个堆叠模型
        for i, target in enumerate(self.target_columns):
            print(f"训练 {target} 目标的堆叠模型...")
            
            # 创建基础模型
            base_models = self._create_base_models()
            
            # 创建堆叠模型
            stacking_model = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=1.0),
                cv=3,
                n_jobs=-1
            )
            
            # 训练模型
            stacking_model.fit(X, y[:, i])
            
            # 保存模型
            self.stacking_models[target] = stacking_model
            
        self.is_fitted = True
        print("堆叠集成模型训练完成")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测值矩阵 (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 对每个目标进行预测
        predictions = []
        for target in self.target_columns:
            pred = self.stacking_models[target].predict(X)
            predictions.append(pred)
        
        # 合并预测结果
        final_predictions = np.column_stack(predictions)
        return final_predictions

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
        # 创建模型保存目录
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存堆叠模型
        models_path = filepath.replace('.pkl', '_models.pkl')
        joblib.dump(self.stacking_models, models_path)
        
        # 保存模型元数据
        metadata = {
            'fingerprint_dim': self.fingerprint_dim,
            'descriptor_dim': self.descriptor_dim,
            'target_columns': self.target_columns,
            'is_fitted': self.is_fitted
        }
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'StackingEnsembleModel':
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
            fingerprint_dim=metadata['fingerprint_dim'],
            descriptor_dim=metadata['descriptor_dim']
        )
        
        # 加载堆叠模型
        models_path = filepath.replace('.pkl', '_models.pkl')
        model.stacking_models = joblib.load(models_path)
        
        # 恢复其他属性
        model.target_columns = metadata['target_columns']
        model.is_fitted = metadata['is_fitted']
        
        return model