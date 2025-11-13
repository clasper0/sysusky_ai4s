# pIC50 预测模型项目

这个项目旨在使用机器学习方法预测小分子在多个靶点上的pIC50值。项目包含了多个模型脚本和评估工具。

## 项目文件说明

### 1. train_and_evaluate_model.py（基础模型脚本）
这是项目的初始版本脚本，用于：
- 加载训练数据和测试数据
- 准备特征数据
- 训练集成学习模型（随机森林、梯度提升、XGBoost和岭回归）
- 评估模型性能（包含MAE、RMSE、R² Score、Pearson和Spearman相关系数）
- 对测试集进行预测
- 保存预测结果和评估指标

**运行方式：**
```bash
python train_and_evaluate_model.py
```

### 2. improved_extended_rdkit_model.py（改进模型脚本）
这是一个改进的模型脚本，相比基础版本增加了以下功能：
- 使用更全面的集成学习方法（随机森林、梯度提升、XGBoost和岭回归）
- 特征选择和标准化处理
- 网格搜索超参数优化
- 交叉验证评估
- 更详细的模型性能分析

### 3. final_optimized_rdkit_model.py（当前最优脚本）
这是最终优化的模型脚本，也是当前推荐使用的最优版本，包含：
- 经过精细调优的集成学习模型（随机森林、梯度提升、XGBoost和岭回归）
- 投票回归器集成多个模型以提高预测准确性
- 特征选择分析
- 对候选分子的完整预测流程
- 最好的预测性能和稳定性

**推荐运行命令：**
```bash
python final_optimized_rdkit_model.py
```

### 4. feature_importance_analysis.py（特征重要性分析工具）
专门用于分析特征重要性的脚本：
- 分析每个靶点的特征重要性
- 生成特征重要性可视化图表
- 输出各靶点Top重要特征列表
- 提供高活性分子特征分析

## 数据文件

### 训练数据
- `data/activity_train.csv`：包含分子ID、靶点ID和pIC50值
- `data/molecular_features_extended.csv`：包含分子的RDKit特征（1074个特征）

### 测试数据
- `data/candidate_features_extended.csv`：候选分子的RDKit特征

### 输出文件
运行脚本后会在`output`目录下生成：
- `test_set_predictions.csv`：测试集预测结果
- `model_evaluation_metrics.csv`：模型评估指标
- `cross_validation_results.csv`：交叉验证结果
- `model_comparison_results.csv`：模型比较结果
- `best_hyperparameters.json`：最佳超参数配置
- 各靶点的特征重要性分析文件和高活性分子列表

## 模型评估指标

模型评估包含以下指标：
1. Mean Absolute Error (MAE)
2. Root Mean Square Error (RMSE)
3. R² Score
4. Pearson Correlation Coefficient
5. Spearman Correlation Coefficient

## 使用说明

要运行模型训练和评估过程，请执行以下任一命令：

基础模型：
```bash
python train_and_evaluate_model.py
```

改进模型：
```bash
python improved_extended_rdkit_model.py
```

**推荐使用最终优化模型（当前最优）：**
```bash
python final_optimized_rdkit_model.py
```

特征重要性分析：
```bash
python feature_importance_analysis.py
```

这将完成完整的训练、评估和预测流程，并生成相应的输出文件。