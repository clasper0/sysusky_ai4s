"""
高级集成模型
==============

该模块实现了一个具有强泛化能力的集成学习模型，结合多种机器学习算法和集成策略，
用于提高小样本场景下的分子活性预测能力。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


class AdvancedEnsembleModel:
    """
    高级集成模型，结合多种机器学习算法和集成策略
    """

    def __init__(self, n_features=1000, cv_folds=5):
        """
        初始化高级集成模型

        Args:
            n_features: 最大特征数量
            cv_folds: 交叉验证折数
        """
        self.n_features = n_features
        self.cv_folds = cv_folds
        self.models = {}
        self.feature_selectors = {}
        self.scalers = {}
        self.target_names = ['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']
        self.model_weights = {}
        
        # 初始化基础模型
        self._initialize_models()
        
    def _initialize_models(self):
        """
        初始化基础模型
        """
        # 随机森林 - 适用于小样本数据，泛化能力强
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost - 梯度提升算法
        self.models['xgb'] = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # LightGBM - 高效的梯度提升框架
        self.models['lgb'] = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # 梯度提升回归器
        self.models['gbr'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Extra Trees - 额外的树模型
        self.models['etr'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # 岭回归 - 线性模型，正则化防止过拟合
        self.models['ridge'] = Ridge(alpha=1.0)
        
        # Lasso回归 - 稀疏线性模型
        self.models['lasso'] = Lasso(alpha=0.1)
        
        # 弹性网络 - 结合岭回归和Lasso的优点
        self.models['elastic'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # 支持向量回归
        self.models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        # 多层感知机
        self.models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )

    def _prepare_features(self, X):
        """
        准备特征数据

        Args:
            X: 输入特征

        Returns:
            处理后的特征
        """
        # 标准化特征
        X_scaled = StandardScaler().fit_transform(X)
        return X_scaled

    def _select_features(self, X, y, target_name):
        """
        为特定目标选择特征

        Args:
            X: 输入特征
            y: 目标值
            target_name: 目标名称

        Returns:
            选择的特征
        """
        # 使用SelectKBest选择与目标最相关的特征
        selector = SelectKBest(score_func=f_regression, k=min(self.n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # 保存特征选择器
        self.feature_selectors[target_name] = selector
        
        return X_selected

    def fit(self, X, y):
        """
        训练模型

        Args:
            X: 输入特征
            y: 目标值 (n_samples, n_targets)
        """
        print("开始训练高级集成模型...")
        
        # 准备特征
        X_processed = self._prepare_features(X)
        
        # 为每个目标分别训练模型
        for i, target_name in enumerate(self.target_names):
            print(f"训练目标 {target_name} 的模型...")
            
            # 获取当前目标的标签
            y_target = y[:, i]
            
            # 特征选择
            X_selected = self._select_features(X_processed, y_target, target_name)
            
            # 保存标准化器
            self.scalers[target_name] = StandardScaler()
            self.scalers[target_name].fit(X)
            
            # 存储选择后的特征
            X_train = self.feature_selectors[target_name].transform(
                self.scalers[target_name].transform(X)
            )
            
            # 训练所有基础模型并计算权重
            model_scores = {}
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for model_name, model in self.models.items():
                try:
                    # 使用交叉验证评估模型
                    cv_scores = cross_val_score(
                        model, X_train, y_target, 
                        cv=cv, scoring='r2', n_jobs=-1
                    )
                    model_scores[model_name] = np.mean(cv_scores)
                    print(f"  {model_name}: CV R2 = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
                    
                    # 训练模型
                    model.fit(X_train, y_target)
                except Exception as e:
                    print(f"  {model_name}: 训练失败 ({str(e)})")
                    model_scores[model_name] = -np.inf
            
            # 计算模型权重（基于验证集表现）
            # 表现差的模型（R2 <= 0）权重为0
            weights = np.array(list(model_scores.values()))
            weights = np.maximum(weights, 0)  # 负值设为0
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)  # 归一化
            else:
                # 如果所有模型表现都差，则平均分配权重
                weights = np.ones(len(weights)) / len(weights)
            
            self.model_weights[target_name] = dict(zip(model_scores.keys(), weights))
            print(f"  模型权重: {self.model_weights[target_name]}")
        
        print("高级集成模型训练完成!")

    def predict(self, X):
        """
        预测

        Args:
            X: 输入特征

        Returns:
            预测值 (n_samples, n_targets)
        """
        predictions = np.zeros((X.shape[0], len(self.target_names)))
        
        # 为每个目标进行预测
        for i, target_name in enumerate(self.target_names):
            # 特征预处理
            X_scaled = self.scalers[target_name].transform(X)
            X_selected = self.feature_selectors[target_name].transform(X_scaled)
            
            # 加权集成预测
            pred = np.zeros(X_selected.shape[0])
            for model_name, weight in self.model_weights[target_name].items():
                if weight > 0:  # 只使用权重大于0的模型
                    model_pred = self.models[model_name].predict(X_selected)
                    pred += weight * model_pred
            
            predictions[:, i] = pred
        
        return predictions

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 输入特征
            y: 真实标签

        Returns:
            评估指标
        """
        # 预测
        y_pred = self.predict(X)
        
        # 计算每个目标的评估指标
        metrics = {}
        for i, target_name in enumerate(self.target_names):
            mse = mean_squared_error(y[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y[:, i], y_pred[:, i])
            
            metrics[target_name] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        
        # 计算总体指标
        total_mse = mean_squared_error(y, y_pred)
        total_rmse = np.sqrt(total_mse)
        total_r2 = r2_score(y, y_pred)
        
        metrics['overall'] = {
            'mse': total_mse,
            'rmse': total_rmse,
            'r2': total_r2
        }
        
        return metrics

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        import pickle
        import os
        
        # 创建目录
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 保存模型
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"模型已保存到: {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            加载的模型
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"模型已从 {filepath} 加载")
        return model


class StackingEnsembleModel:
    """
    堆叠集成模型
    """

    def __init__(self, n_features=1000):
        """
        初始化堆叠集成模型

        Args:
            n_features: 最大特征数量
        """
        self.n_features = n_features
        self.base_models = {}
        self.meta_models = {}
        self.feature_selectors = {}
        self.scalers = {}
        self.target_names = ['target1_pIC50', 'target2_pIC50', 'target3_pIC50', 'target4_pIC50', 'target5_pIC50']
        
        # 初始化基础模型
        self._initialize_base_models()
        
        # 初始化元模型
        self._initialize_meta_models()
        
    def _initialize_base_models(self):
        """
        初始化基础模型
        """
        # 为每个目标初始化基础模型
        for target_name in self.target_names:
            self.base_models[target_name] = {
                'rf': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgb': XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                ),
                'lgb': LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
            }

    def _initialize_meta_models(self):
        """
        初始化元模型
        """
        # 使用岭回归作为元模型
        for target_name in self.target_names:
            self.meta_models[target_name] = Ridge(alpha=1.0)

    def fit(self, X, y):
        """
        训练堆叠模型

        Args:
            X: 输入特征
            y: 目标值 (n_samples, n_targets)
        """
        print("开始训练堆叠集成模型...")
        
        # 准备特征
        X_processed = StandardScaler().fit_transform(X)
        
        # 为每个目标训练堆叠模型
        for i, target_name in enumerate(self.target_names):
            print(f"训练目标 {target_name} 的堆叠模型...")
            
            # 获取当前目标的标签
            y_target = y[:, i]
            
            # 特征选择
            selector = SelectKBest(score_func=f_regression, k=min(self.n_features, X.shape[1]))
            X_selected = selector.fit_transform(X_processed, y_target)
            self.feature_selectors[target_name] = selector
            
            # 保存标准化器
            self.scalers[target_name] = StandardScaler()
            self.scalers[target_name].fit(X)
            
            # 存储选择后的特征
            X_train = selector.transform(self.scalers[target_name].transform(X))
            
            # 生成基础模型的预测（使用交叉验证）
            base_predictions = np.zeros((X_train.shape[0], len(self.base_models[target_name])))
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            for j, (model_name, model) in enumerate(self.base_models[target_name].items()):
                # 使用交叉验证生成基础模型预测
                cv_predictions = np.zeros(X_train.shape[0])
                for train_idx, val_idx in cv.split(X_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_target[train_idx], y_target[val_idx]
                    
                    model.fit(X_cv_train, y_cv_train)
                    cv_predictions[val_idx] = model.predict(X_cv_val)
                
                base_predictions[:, j] = cv_predictions
                # 最后用全部数据训练模型
                model.fit(X_train, y_target)
            
            # 训练元模型
            self.meta_models[target_name].fit(base_predictions, y_target)
        
        print("堆叠集成模型训练完成!")

    def predict(self, X):
        """
        预测

        Args:
            X: 输入特征

        Returns:
            预测值 (n_samples, n_targets)
        """
        predictions = np.zeros((X.shape[0], len(self.target_names)))
        
        # 为每个目标进行预测
        for i, target_name in enumerate(self.target_names):
            # 特征预处理
            X_scaled = self.scalers[target_name].transform(X)
            X_selected = self.feature_selectors[target_name].transform(X_scaled)
            
            # 生成基础模型的预测
            base_predictions = np.zeros((X_selected.shape[0], len(self.base_models[target_name])))
            for j, (model_name, model) in enumerate(self.base_models[target_name].items()):
                base_predictions[:, j] = model.predict(X_selected)
            
            # 元模型预测
            predictions[:, i] = self.meta_models[target_name].predict(base_predictions)
        
        return predictions

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 输入特征
            y: 真实标签

        Returns:
            评估指标
        """
        # 预测
        y_pred = self.predict(X)
        
        # 计算每个目标的评估指标
        metrics = {}
        for i, target_name in enumerate(self.target_names):
            mse = mean_squared_error(y[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y[:, i], y_pred[:, i])
            
            metrics[target_name] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        
        # 计算总体指标
        total_mse = mean_squared_error(y, y_pred)
        total_rmse = np.sqrt(total_mse)
        total_r2 = r2_score(y, y_pred)
        
        metrics['overall'] = {
            'mse': total_mse,
            'rmse': total_rmse,
            'r2': total_r2
        }
        
        return metrics

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        import pickle
        import os
        
        # 创建目录
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 保存模型
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"模型已保存到: {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            加载的模型
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"模型已从 {filepath} 加载")
        return model