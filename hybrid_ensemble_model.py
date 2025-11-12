"""
混合集成学习模型
================

该模块实现了一个混合集成学习模型，结合多种基础模型的优势，
用于基于分子指纹和描述符的pIC50值预测。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
from typing import List, Dict, Tuple, Optional, Any


class HybridEnsembleModel:
    """
    混合集成学习模型，结合多种基础模型的优势
    """

    def __init__(self, fingerprint_dim: int = 2048, descriptor_dim: int = 25, **kwargs):
        """
        初始化混合集成模型

        Args:
            fingerprint_dim: 分子指纹维度
            descriptor_dim: 分子描述符维度
            **kwargs: 其他参数
        """
        self.fingerprint_dim = fingerprint_dim
        self.descriptor_dim = descriptor_dim
        self.models = {}
        self.weights = {}
        self.target_columns = []
        self.is_fitted = False
        
        # 初始化基础模型
        self._initialize_base_models()

    def _initialize_base_models(self):
        """
        初始化基础模型
        """
        # 1. 随机森林模型 - 对小样本数据稳定
        self.models['rf'] = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
        
        # 2. XGBoost模型 - 强大的梯度提升算法
        self.models['xgb'] = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.85,
                reg_alpha=0.2,
                reg_lambda=1.5,
                random_state=42
            )
        )
        
        # 3. LightGBM模型 - 高效的梯度提升框架
        self.models['lgb'] = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.85,
                reg_alpha=0.2,
                reg_lambda=1.5,
                random_state=42
            )
        )
        
        # 4. Gradient Boosting模型 - 传统的梯度提升算法
        self.models['gb'] = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                random_state=42
            )
        )
        
        # 5. Ridge回归 - 线性模型作为基准
        self.models['ridge'] = MultiOutputRegressor(
            Ridge(alpha=0.8, random_state=42)
        )

    def fit(self, X: np.ndarray, y: np.ndarray, target_columns: List[str] = None) -> 'HybridEnsembleModel':
        """
        训练混合集成模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值矩阵 (n_samples, n_targets)
            target_columns: 目标列名称

        Returns:
            训练好的模型
        """
        print(f"训练混合集成模型...")
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")
        
        # 保存目标列名
        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = [f"target_{i}" for i in range(y.shape[1])]
        
        # 训练各个基础模型
        model_scores = {}
        for name, model in self.models.items():
            print(f"训练 {name} 模型...")
            
            # 所有模型都使用完整特征
            model.fit(X, y)
                
            # 使用交叉验证评估模型
            try:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                model_scores[name] = np.mean(cv_scores)
            except:
                # 如果交叉验证失败，使用默认权重
                model_scores[name] = 0.0
            
            print(f"{name} 模型交叉验证 R2 分数: {model_scores[name]:.4f}")
        
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
        print("混合集成模型训练完成")
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
        
        # 获取各个模型的预测结果
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # 加权集成预测结果
        final_predictions = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            final_predictions += self.weights[name] * pred
        
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
        
        # 保存基础模型
        models_path = filepath.replace('.pkl', '_models.pkl')
        joblib.dump(self.models, models_path)
        
        # 保存模型元数据
        metadata = {
            'fingerprint_dim': self.fingerprint_dim,
            'descriptor_dim': self.descriptor_dim,
            'target_columns': self.target_columns,
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'HybridEnsembleModel':
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
        
        # 加载基础模型
        models_path = filepath.replace('.pkl', '_models.pkl')
        model.models = joblib.load(models_path)
        
        # 恢复其他属性
        model.target_columns = metadata['target_columns']
        model.weights = metadata['weights']
        model.is_fitted = metadata['is_fitted']
        
        return model