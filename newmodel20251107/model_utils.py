"""
模型工具
========

该模块提供了模型保存和加载功能。
"""

import os
import joblib
import pickle
from typing import Any


def save_model(model: Any, filepath: str):
    """
    保存模型

    Args:
        model: 要保存的模型
        filepath: 保存路径
    """
    # 创建目录
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存模型
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {filepath}")


def load_model(filepath: str) -> Any:
    """
    加载模型

    Args:
        filepath: 模型文件路径

    Returns:
        加载的模型
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    # 加载模型
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"模型已从 {filepath} 加载")
    return model