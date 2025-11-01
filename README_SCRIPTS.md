# 分子图卷积网络项目脚本说明

本项目基于图卷积网络(GCN)实现分子pIC50值预测，包含多个核心脚本，每个脚本承担不同的功能。

## 项目结构概览

```
sysusky_ai4s/
├── smiles_to_graph.py      # SMILES到图结构的转换
├── gcn_model.py            # GCN模型架构定义
├── data_loader.py          # 数据加载和预处理
├── trainer.py              # 训练循环和评估
├── model_utils.py          # 模型保存、加载和管理
├── inference.py            # 预测和推理接口
├── train.py               # 主训练脚本
├── visualize_molecules.py  # 分子可视化工具
├── predict.py             # 命令行预测脚本（由inference.py生成）
└── README_SCRIPTS.md      # 本说明文件
```

## 各脚本详细说明

### 1. smiles_to_graph.py - SMILES到图结构转换器

该脚本负责将SMILES字符串转换为图结构数据，供GCN模型使用。

主要功能：
- 使用RDKit解析SMILES字符串
- 提取原子特征（原子类型、电荷、度数、价电子数等）
- 提取键特征（键类型、立体化学、环状态等）
- 构建PyTorch Geometric兼容的图数据结构

核心类：
- `SMILESToGraph`：主要转换类，包含原子和键的特征提取方法

### 2. gcn_model.py - GCN模型定义

该脚本定义了图卷积神经网络模型架构。

主要功能：
- 实现基础单任务GCN模型
- 实现多任务学习版本
- 包含注意力机制和残差连接
- 提供多种全局池化策略（平均、最大、加和、注意力）

核心类：
- `MolecularGCN`：基础分子GCN模型
- `MultiTaskMolecularGCN`：多任务学习模型
- `AttentionPooling`：注意力池化层

### 3. data_loader.py - 数据加载和预处理

该脚本处理数据加载、验证和预处理过程。

主要功能：
- 加载包含SMILES和pIC50值的CSV数据
- 将SMILES转换为图结构
- 数据集验证和清洗
- 训练/验证/测试数据分割
- 批处理和缓存优化

核心类：
- `MolecularDataset`：PyTorch数据集类
- `MolecularDataLoader`：数据加载器类

### 4. trainer.py - 模型训练器

该脚本提供完整的训练流程和评估机制。

主要功能：
- 完整的训练循环实现
- 支持多种优化器和学习率调度器
- 实现早停机制和模型检查点
- 计算评估指标（RMSE, MAE, R²）
- 训练曲线可视化

核心类：
- `MolecularTrainer`：模型训练器

### 5. model_utils.py - 模型管理工具

该脚本提供模型的保存、加载和管理功能。

主要功能：
- 模型版本控制和注册
- 完整的模型保存/加载机制
- 模型元数据管理
- 预测接口

核心类：
- `ModelManager`：模型管理器
- `ModelPredictor`：模型预测器

### 6. inference.py - 推理引擎

该脚本提供高级预测接口和推理功能。

主要功能：
- 高级预测接口
- 批处理支持
- 模型集成
- 不确定性估计（MC Dropout）
- 结果可视化和分析

核心类：
- `MolecularInferenceEngine`：推理引擎

### 7. train.py - 主训练脚本

该脚本是模型训练的入口点。

主要功能：
- 解析命令行参数
- 设置训练环境
- 加载数据
- 创建模型
- 启动训练过程
- 保存训练结果

使用方法：
```bash
python train.py --data-path data.csv --epochs 100 --batch-size 32
```

### 8. visualize_molecules.py - 分子可视化工具

该脚本提供分子结构可视化功能。

主要功能：
- 将SMILES转换为2D分子结构图
- 批量可视化分子
- 保存可视化结果

### 9. predict.py - 命令行预测脚本

该脚本由[inference.py](file:///d:/Codes/sysusky_ai4s/inference.py)生成，提供命令行预测接口。

使用方法：
```bash
python predict.py --input molecules.csv --model molecular_gcn_v1 --uncertainty --visualize
```

## 使用流程

1. 准备数据：准备包含SMILES和pIC50值的CSV文件
2. 训练模型：运行[train.py](file:///d:/Codes/sysusky_ai4s/train.py)进行模型训练
3. 模型推理：使用[inference.py](file:///d:/Codes/sysusky_ai4s/inference.py)或[predict.py](file:///d:/Codes/sysusky_ai4s/predict.py)进行预测
4. 结果可视化：使用[visualize_molecules.py](file:///d:/Codes/sysusky_ai4s/visualize_molecules.py)可视化分子结构

## 依赖库

- PyTorch
- PyTorch Geometric
- RDKit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib