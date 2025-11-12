# 基于分子指纹的pIC50预测模型

该模块实现了基于分子指纹的pIC50预测模型，特别针对小样本数据进行了优化。

## 特点

1. **小样本优化**：使用随机森林和XGBoost等在小样本场景下表现稳定的模型
2. **多种指纹类型**：支持Morgan指纹和RDKit指纹
3. **多任务学习**：可以同时预测多个靶点的pIC50值
4. **易于使用**：提供完整的训练和预测脚本

## 文件说明

- [fingerprint_extractor.py](file:///d:/Codes/sysusky_ai4s/fingerprint_extractor.py)：分子指纹提取器，用于从SMILES生成分子指纹和描述符
- [fingerprint_model.py](file:///d:/Codes/sysusky_ai4s/fingerprint_model.py)：基于指纹的预测模型实现
- [fingerprint_data_loader.py](file:///d:/Codes/sysusky_ai4s/fingerprint_data_loader.py)：数据加载和预处理模块
- [train_fingerprint.py](file:///d:/Codes/sysusky_ai4s/train_fingerprint.py)：训练脚本
- [predict_fingerprint.py](file:///d:/Codes/sysusky_ai4s/predict_fingerprint.py)：预测脚本

## 使用方法

### 训练模型

```bash
python train_fingerprint.py --data-dir data --model-type rf --fingerprint-type morgan
```

参数说明：
- `--data-dir`：数据目录路径，默认为"data"
- `--model-type`：模型类型，可选"rf"（随机森林）或"xgb"（XGBoost），默认为"rf"
- `--fingerprint-type`：指纹类型，可选"morgan"或"rdkit"，默认为"morgan"
- `--n-bits`：指纹位数，默认为2048
- `--test-size`：测试集比例，默认为0.2
- `--val-size`：验证集比例，默认为0.1
- `--output-dir`：输出目录，默认为"fingerprint_experiments"
- `--experiment-name`：实验名称，默认为当前时间戳

### 进行预测

```bash
python predict_fingerprint.py --model-path fingerprint_experiments/experiment_20230101_120000/fingerprint_model.pkl --input data/molecule.smi --output predictions.csv
```

参数说明：
- `--model-path`：训练好的模型文件路径（必须）
- `--input`：输入文件路径，包含SMILES的CSV文件（必须）
- `--output`：输出文件路径（必须）
- `--smiles-column`：SMILES列名，默认为"smiles"
- `--id-column`：分子ID列名，如果没有则自动生成

## 模型选择建议

### 随机森林 (Random Forest)
- 对小样本数据更加稳定
- 不容易过拟合
- 提供特征重要性信息
- 训练速度较快

### XGBoost
- 在某些情况下可能有更好的预测性能
- 有正则化项防止过拟合
- 可以处理复杂的非线性关系

## 指纹类型选择

### Morgan指纹
- 应用最广泛的分子指纹类型
- 对分子的局部环境敏感
- 适合大多数场景

### RDKit指纹
- RDKit库原生支持的指纹
- 计算速度较快
- 在某些场景下可能有不同的表现

## 输出文件

训练完成后，会在指定的输出目录中生成以下文件：
- `config.json`：训练配置参数
- `fingerprint_model.pkl`：训练好的模型
- `scaler.pkl`：特征标准化器
- `results.json`：评估结果
- `test_predictions.csv`：测试集预测结果

## 注意事项

1. 确保输入数据格式正确，特别是SMILES和活性数据的对应关系
2. 对于小样本数据，建议使用随机森林模型
3. 可以通过调整模型参数进一步优化性能
4. 保存的标准化器在预测时需要与模型一起使用