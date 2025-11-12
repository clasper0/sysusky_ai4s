"""
指纹数据加载器
==============

该模块用于加载和处理基于指纹的分子数据，支持整合多种分子特征。
针对小样本数据进行了优化，提供稳定的数据处理管道。
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from fingerprint_extractor import FingerprintExtractor


class FingerprintDataLoader:
    """
    指纹数据加载器，用于处理分子数据并生成特征
    """

    def __init__(self, fingerprint_type: str = 'morgan', n_bits: int = 1024):
        """
        初始化数据加载器

        Args:
            fingerprint_type: 指纹类型
            n_bits: 指纹位数
        """
        self.fingerprint_type = fingerprint_type
        self.n_bits = n_bits
        self.scaler = StandardScaler()
        self.fingerprint_extractor = FingerprintExtractor(fingerprint_type, n_bits)
        self.is_fitted = False

    def load_data(self, activity_file: str, smiles_file: str, property_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载数据文件

        Args:
            activity_file: 活性数据文件路径
            smiles_file: SMILES数据文件路径
            property_file: 分子属性数据文件路径（可选）

        Returns:
            (活性数据, SMILES数据, 属性数据)
        """
        # 加载活性数据
        activity_df = pd.read_csv(activity_file)
        
        # 加载SMILES数据
        smiles_df = pd.read_csv(smiles_file, header=None, names=['molecule_id', 'smiles'])
        
        # 加载属性数据（如果提供）
        property_df = None
        if property_file and os.path.exists(property_file):
            property_df = pd.read_csv(property_file)
            
        return activity_df, smiles_df, property_df

    def prepare_dataset(self, activity_df: pd.DataFrame, smiles_df: pd.DataFrame, 
                       property_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        准备数据集

        Args:
            activity_df: 活性数据
            smiles_df: SMILES数据
            property_df: 分子属性数据（可选）

        Returns:
            (特征矩阵, 目标值矩阵, 分子ID列表, 目标列名列表)
        """
        print("准备数据集...")
        
        # 合并活性数据和SMILES数据
        merged_df = pd.merge(activity_df, smiles_df, on='molecule_id')
        
        # 如果提供了属性数据，也合并进来
        if property_df is not None:
            merged_df = pd.merge(merged_df, property_df, on='molecule_id')
        
        # 将活性数据转换为宽格式 (每个target_id一列)
        pivot_df = merged_df.pivot(index='molecule_id', columns='target_id', values='pIC50')
        target_columns = list(pivot_df.columns)
        
        # 重置索引，准备合并
        pivot_df = pivot_df.reset_index()
        
        # 合并所有数据
        final_df = pd.merge(pivot_df, smiles_df, on='molecule_id')
        if property_df is not None:
            final_df = pd.merge(final_df, property_df, on='molecule_id')
        
        # 提取分子ID
        molecule_ids = final_df['molecule_id'].tolist()
        
        # 提取SMILES
        smiles_list = final_df['smiles'].tolist()
        
        # 提取目标值
        targets = final_df[target_columns].values.astype(np.float32)
        
        # 提取分子指纹特征
        print("提取分子指纹特征...")
        features_list = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            # 使用指纹提取器提取所有特征（指纹+描述符）
            features = self.fingerprint_extractor.extract_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)
        
        # 检查是否有有效特征
        if len(features_list) == 0:
            raise ValueError("没有有效的分子特征被提取，请检查SMILES数据")
        
        # 过滤掉无效的分子
        features = np.vstack(features_list)
        targets = targets[valid_indices]
        molecule_ids = [molecule_ids[i] for i in valid_indices]
        
        print(f"最终数据集大小: {len(features)} 个分子, {features.shape[1]} 个特征, {targets.shape[1]} 个目标")
        print(f"目标列: {target_columns}")
        
        return features, targets, molecule_ids, target_columns

    def split_and_scale_data(self, 
                            features: np.ndarray, 
                            targets: np.ndarray, 
                            test_size: float = 0.2,
                            val_size: float = 0.1) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        划分和标准化数据

        Args:
            features: 特征矩阵
            targets: 目标值矩阵
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）

        Returns:
            ((X_train, X_val, X_test), (y_train, y_val, y_test))
        """
        # 划分训练集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )
        
        # 划分训练集和验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.is_fitted = True
        
        return (X_train_scaled, X_val_scaled, X_test_scaled), (y_train, y_val, y_test)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        转换特征（用于预测新数据）

        Args:
            features: 原始特征矩阵

        Returns:
            标准化后的特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("数据加载器尚未拟合，请先调用split_and_scale_data方法")
        
        return self.scaler.transform(features)
    
    def save_scaler(self, filepath: str):
        """
        保存标准化器

        Args:
            filepath: 标准化器保存路径
        """
        joblib.dump(self.scaler, filepath)
        
    def load_scaler(self, filepath: str):
        """
        加载标准化器

        Args:
            filepath: 标准化器文件路径
        """
        self.scaler = joblib.load(filepath)
        self.is_fitted = True