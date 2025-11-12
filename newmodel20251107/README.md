# 高级集成学习模型 (Advanced Ensemble Learning Model)

该目录包含一个具有强泛化能力的高级集成学习模型，用于提高小样本场景下的分子活性预测能力。

## 模型特点

1. **多种集成策略**：
   - 加权平均集成：基于交叉验证性能为不同模型分配权重
   - 堆叠集成：使用元学习器组合多个基础模型的预测结果

2. **丰富的基础模型**：
   - 随机森林 (Random Forest)
   - XGBoost
   - LightGBM
   - 梯度提升回归器 (Gradient Boosting Regressor)
   - Extra Trees
   - 岭回归 (Ridge Regression)
   - Lasso回归
   - 弹性网络 (Elastic Net)
   - 支持向量回归 (SVR)
   - 多层感知机 (MLP)

3. **特征工程**：
   - Morgan指纹 (2048位)
   - 25个RDKit分子描述符
   - 基于F值的特征选择

4. **模型优化**：
   - 交叉验证评估模型性能
   - 为每个目标独立训练模型
   - 自动计算模型权重

## 文件结构

```
newmodel20251107/
├── advanced_ensemble_model.py  # 高级集成模型实现
├── data_loader.py             # 数据加载和预处理
├── train_model.py             # 模型训练脚本
├── predict.py                 # 模型预测脚本
├── model_utils.py             # 模型工具（保存/加载）
└── README.md                  # 说明文档
```

## 使用方法

### 训练模型

```bash
python train_model.py --data-path ../data/candidate_hybrid.csv --model-type advanced
```

参数说明：
- `--data-path`: 数据文件路径
- `--output-dir`: 输出目录
- `--fingerprint-type`: 指纹类型 (morgan/rdkit)
- `--n-bits`: 指纹位数
- `--n-features`: 最大特征数量
- `--model-type`: 模型类型 (advanced/stacking)

### 进行预测

```bash
python predict.py --model-dir experiments --input ../data/candidate.csv --output predictions.csv
```

参数说明：
- `--model-dir`: 模型目录路径
- `--input`: 输入SMILES文件路径
- `--output`: 输出预测结果文件路径
- `--fingerprint-type`: 指纹类型 (morgan/rdkit)
- `--n-bits`: 指纹位数

## 模型性能

该模型在小样本数据集上具有良好的泛化能力，通过以下方式防止过拟合：

1. 使用交叉验证评估模型性能
2. 对表现差的模型赋予零权重
3. 限制最大特征数量
4. 使用正则化技术

## 输出文件

训练完成后会生成以下文件：

- `config.json`: 训练配置
- `model.pkl`: 保存的模型
- `results.json`: 评估结果
- `test_predictions.csv`: 测试集预测结果

## 依赖

- scikit-learn
- xgboost
- lightgbm
- rdkit
- numpy
- pandas