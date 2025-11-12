# 集成学习模型用于多靶点pIC50预测

本项目实现了一个集成学习模型，可以从SMILES字符串预测化合物对五个靶点的pIC50值。

## 数据格式

输入数据应为CSV格式，至少包含以下列：
- `SMILES`: 化合物的SMILES表示
- `target1_pIC50`, `target2_pIC50`, `target3_pIC50`, `target4_pIC50`, `target5_pIC50`: 五个靶点的pIC50值

## 特征工程

模型会自动从SMILES字符串中提取以下特征：
1. Morgan指纹 (2048位)
2. 25个RDKit分子描述符，包括：
   - 分子量 (MolWt)
   - LogP (MolLogP)
   - 氢键供体数 (NumHDonors)
   - 氢键受体数 (NumHAcceptors)
   - 可旋转键数 (NumRotatableBonds)
   - 环数 (NumRings)
   - 极性表面积 (TPSA)
   - 原子数 (NumAtoms)
   - CSP3比例 (FractionCSP3)
   - 重原子数 (HeavyAtomCount)
   - 等等...

## 模型类型

提供两种集成学习模型：

### 1. 加权集成模型 (weighted)
结合多种机器学习算法，根据交叉验证性能为每个模型分配权重：
- 随机森林 (Random Forest)
- XGBoost
- LightGBM
- 梯度提升回归器 (Gradient Boosting Regressor)
- Ridge回归
- Lasso回归
- 弹性网络 (Elastic Net)
- 支持向量回归 (SVR)
- 多层感知机 (MLP)

### 2. 堆叠集成模型 (stacking)
使用两层结构：
- 基础模型层：随机森林、XGBoost、LightGBM
- 元模型层：Ridge回归

## 使用方法

### 训练模型

```bash
python train.py --data-path ../data/candidate_hybrid.csv --model-type weighted
```

参数说明：
- `--data-path`: 训练数据路径
- `--model-type`: 模型类型 (weighted 或 stacking)
- `--test-size`: 测试集比例 (默认: 0.2)
- `--random-state`: 随机种子 (默认: 42)

### 进行预测

```bash
python predict.py --model-dir experiments_weighted --input ../data/candidate.csv --output predictions.csv
```

参数说明：
- `--model-dir`: 模型目录路径
- `--input`: 待预测数据路径 (CSV格式，需包含SMILES列)
- `--output`: 预测结果输出路径

## 输出文件

训练完成后会生成以下文件：
- `model.pkl`: 训练好的模型
- `selector.pkl`: 特征选择器（如果使用了特征选择）
- `results.json`: 模型评估结果
- `test_predictions.csv`: 测试集预测结果

## 依赖

请查看 [requirements.txt](file:///d:/Codes/sysusky_ai4s/ensemble_model/requirements.txt) 文件获取依赖列表。