"""
优化的混合集成学习模型
=====================

该模块实现了一个优化的混合集成学习模型，专门针对小样本场景优化，
用于基于分子指纹和描述符的pIC50值预测。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
from typing import List, Dict, Tuple, Optional, Any


class OptimizedEnsembleModel:
    """
    优化的混合集成学习模型，专为提高测试集R2分数而设计
    """

    def __init__(self, fingerprint_dim: int = 2048, descriptor_dim: int = 25, **kwargs):
        """
        初始化优化的混合集成模型

        Args:
            fingerprint_dim: 分子指纹维度
            descriptor_dim: 分描述符维度
            **kwargs: 其他参数
        """
        self.fingerprint_dim = fingerprint_dim
        self.descriptor_dim = descriptor_dim
        self.models = {}
        self.weights = {}
        self.target_columns = []
        self.is_fitted = False
        self.feature_selectors = {}  # 为每个目标存储特征选择器
        
        # 初始化基础模型
        self._initialize_base_models()

    def _initialize_base_models(self):
        """
        初始化基础模型，使用更适合小样本数据的参数
        """
        # 1. 随机森林模型 - 对小样本数据稳定，减少过拟合
        self.models['rf'] = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=150,          # 减少树的数量防止过拟合
                max_depth=12,              # 限制最大深度
                min_samples_split=5,       # 增加分裂所需的最小样本数
                min_samples_leaf=3,        # 增加叶节点最小样本数
                max_features='sqrt',       # 限制每次分裂考虑的特征数量
                random_state=42,
                n_jobs=-1
            )
        )
        
        # 2. XGBoost模型 - 强大的梯度提升算法，加入更强正则化
        self.models['xgb'] = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=150,          # 减少树的数量
                max_depth=6,               # 减少最大深度
                learning_rate=0.08,        # 稍微提高学习率
                subsample=0.7,             # 减少样本采样比例
                colsample_bytree=0.7,      # 减少特征采样比例
                reg_alpha=0.5,             # 增加L1正则化
                reg_lambda=2.0,            # 增加L2正则化
                random_state=42
            )
        )
        
        # 3. LightGBM模型 - 高效的梯度提升框架，同样加强正则化
        self.models['lgb'] = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42
            )
        )
        
        # 4. Gradient Boosting模型 - 传统的梯度提升算法
        self.models['gb'] = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.7,
                random_state=42
            )
        )
        
        # 5. Ridge回归 - 线性模型作为基准，适度正则化
        self.models['ridge'] = MultiOutputRegressor(
            Ridge(alpha=1.0, random_state=42)
        )
        
        # 6. ElasticNet回归 - 线性模型，结合L1和L2正则化
        self.models['elastic'] = MultiOutputRegressor(
            ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=2000)
        )

    def fit(self, X: np.ndarray, y: np.ndarray, target_columns: List[str] = None) -> 'OptimizedEnsembleModel':
        """
        训练优化的混合集成模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值矩阵 (n_samples, n_targets)
            target_columns: 目标列名称

        Returns:
            训练好的模型
        """
        print(f"训练优化的混合集成模型...")
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")
        
        # 保存目标列名
        if target_columns is not None:
            self.target_columns = target_columns
        else:
            self.target_columns = [f"target_{i}" for i in range(y.shape[1])]
        
        # 为每个目标单独进行特征选择和模型训练
        model_scores = {}
        for name in self.models.keys():
            model_scores[name] = []
            
        # 对每个目标分别处理
        for i, target in enumerate(self.target_columns):
            print(f"\n处理目标 {target}...")
            
            # 特征选择 - 为每个目标选择最重要的特征
            selector = SelectKBest(score_func=f_regression, k=min(500, X.shape[1]))  # 限制最多500个特征
            X_selected = selector.fit_transform(X, y[:, i])
            self.feature_selectors[target] = selector
            print(f"特征选择后维度: {X_selected.shape}")
            
            # 使用选定的特征训练各个基础模型
            target_scores = {}
            for name, model in self.models.items():
                print(f"  训练 {name} 模型...")
                
                # 训练模型
                # 注意：MultiOutputRegressor需要二维y，所以这里要reshape
                model.estimator.fit(X_selected, y[:, i].reshape(-1, 1))
                    
                # 使用交叉验证评估模型
                try:
                    # 使用分层折叠交叉验证
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model.estimator, X_selected, y[:, i], cv=kf, scoring='r2')
                    target_scores[name] = np.mean(cv_scores)
                except Exception as e:
                    print(f"    交叉验证失败: {e}")
                    # 如果交叉验证失败，使用简单验证
                    target_scores[name] = 0.0
                
                print(f"    {name} 模型交叉验证 R2 分数: {target_scores[name]:.4f}")
                model_scores[name].append(target_scores[name])
        
        # 计算每个模型的平均得分
        avg_model_scores = {}
        for name in model_scores:
            if len(model_scores[name]) > 0:
                avg_model_scores[name] = np.mean(model_scores[name])
            else:
                avg_model_scores[name] = 0.0
            print(f"{name} 模型平均 R2 分数: {avg_model_scores[name]:.4f}")
        
        # 根据验证分数设置模型权重，确保权重为正
        # 将负分转换为0，然后归一化
        for name in avg_model_scores:
            if avg_model_scores[name] < 0:
                avg_model_scores[name] = 0
        
        # 如果所有模型得分都是0，则平均分配权重
        total_score = sum(avg_model_scores.values())
        if total_score <= 0:
            for name in avg_model_scores:
                self.weights[name] = 1.0 / len(avg_model_scores)
                print(f"{name} 模型权重: {self.weights[name]:.4f}")
        else:
            # 根据得分比例分配权重
            for name, score in avg_model_scores.items():
                self.weights[name] = score / total_score
                print(f"{name} 模型权重: {self.weights[name]:.4f}")
        
        self.is_fitted = True
        print("优化的混合集成模型训练完成")
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
        
        # 对每个目标分别进行预测
        predictions_per_target = []
        for i, target in enumerate(self.target_columns):
            # 应用特征选择
            X_selected = self.feature_selectors[target].transform(X)
            
            # 获取各个模型的预测结果
            predictions = {}
            for name, model in self.models.items():
                # 注意：MultiOutputRegressor的预测结果需要squeeze
                pred = model.estimator.predict(X_selected).squeeze()
                predictions[name] = pred
            
            # 加权集成预测结果
            final_prediction = np.zeros(X_selected.shape[0])
            for name, pred in predictions.items():
                final_prediction += self.weights[name] * pred
            
            predictions_per_target.append(final_prediction)
        
        # 合并所有目标的预测结果
        final_predictions = np.column_stack(predictions_per_target)
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
        
        # 保存特征选择器
        selectors_path = filepath.replace('.pkl', '_selectors.pkl')
        joblib.dump(self.feature_selectors, selectors_path)
        
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
    def load_model(cls, filepath: str) -> 'OptimizedEnsembleModel':
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
        
        # 加载特征选择器
        selectors_path = filepath.replace('.pkl', '_selectors.pkl')
        model.feature_selectors = joblib.load(selectors_path)
        
        # 恢复其他属性
        model.target_columns = metadata['target_columns']
        model.weights = metadata['weights']
        model.is_fitted = metadata['is_fitted']
        
        return model