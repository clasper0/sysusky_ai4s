# 模型输出结果说明

本目录包含了机器学习模型训练、评估和分析过程中生成的所有输出文件。

## 模型性能评估文件

### [model_evaluation_metrics.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/model_evaluation_metrics.csv)
- **描述**: 模型在测试集上的评估指标
- **包含指标**: 
  - MAE (Mean Absolute Error): 平均绝对误差
  - RMSE (Root Mean Square Error): 均方根误差
  - R² Score: 决定系数
  - Pearson Correlation: 皮尔逊相关系数
  - Spearman Correlation: 斯皮尔曼相关系数

### [cross_validation_results.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/cross_validation_results.csv)
- **描述**: 模型的交叉验证结果
- **包含指标**:
  - mae_mean/std: MAE的均值和标准差
  - rmse_mean/std: RMSE的均值和标准差
  - r2_mean/std: R² Score的均值和标准差

### [model_comparison_results.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/model_comparison_results.csv)
- **描述**: 不同模型的性能比较结果
- **包含**: 各种基模型和集成模型在多个指标上的性能对比

## 超参数优化文件

### [best_hyperparameters.json](file:///Users/zitham_nie/Desktop/Easy_model/output/best_hyperparameters.json)
- **描述**: Optuna优化得到的最佳超参数
- **包含**: 随机森林、梯度提升、XGBoost和岭回归模型的最佳参数配置

## 预测结果文件

### [test_set_predictions.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/test_set_predictions.csv)
- **描述**: 模型对测试集的预测结果
- **包含**: 分子ID、靶点ID和预测的pIC50值

### 候选分子预测文件
- **[deep_learning_candidate_predictions.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/deep_learning_candidate_predictions.csv)**: 深度学习模型对候选分子的预测
- **[final_optimized_candidate_predictions.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/final_optimized_candidate_predictions.csv)**: 最终优化模型对候选分子的预测
- **[improved_candidate_extended_rdkit_predictions.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/improved_candidate_extended_rdkit_predictions.csv)**: 改进的RDKit模型对候选分子的预测

## 特征重要性分析文件

### 各靶点Top 10重要特征
- **[top10_features_TAR_001.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/top10_features_TAR_001.csv)**: 靶点TAR_001的Top 10重要特征
- **[top10_features_TAR_002.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/top10_features_TAR_002.csv)**: 靶点TAR_002的Top 10重要特征
- **[top10_features_TAR_003.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/top10_features_TAR_003.csv)**: 靶点TAR_003的Top 10重要特征
- **[top10_features_TAR_004.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/top10_features_TAR_004.csv)**: 靶点TAR_004的Top 10重要特征
- **[top10_features_TAR_005.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/top10_features_TAR_005.csv)**: 靶点TAR_005的Top 10重要特征

### 高活性分子列表
- **[high_activity_molecules_TAR_001.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecules_TAR_001.csv)**: 靶点TAR_001的高活性分子
- **[high_activity_molecules_TAR_002.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecules_TAR_002.csv)**: 靶点TAR_002的高活性分子
- **[high_activity_molecules_TAR_003.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecules_TAR_003.csv)**: 靶点TAR_003的高活性分子
- **[high_activity_molecules_TAR_004.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecules_TAR_004.csv)**: 靶点TAR_004的高活性分子
- **[high_activity_molecules_TAR_005.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecules_TAR_005.csv)**: 靶点TAR_005的高活性分子

### 高活性分子特征分析
- **[high_activity_molecule_features_TAR_001.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecule_features_TAR_001.csv)**: 靶点TAR_001高活性分子的特征分析
- **[high_activity_molecule_features_TAR_002.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecule_features_TAR_002.csv)**: 靶点TAR_002高活性分子的特征分析
- **[high_activity_molecule_features_TAR_003.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecule_features_TAR_003.csv)**: 靶点TAR_003高活性分子的特征分析
- **[high_activity_molecule_features_TAR_004.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecule_features_TAR_004.csv)**: 靶点TAR_004高活性分子的特征分析
- **[high_activity_molecule_features_TAR_005.csv](file:///Users/zitham_nie/Desktop/Easy_model/output/high_activity_molecule_features_TAR_005.csv)**: 靶点TAR_005高活性分子的特征分析

### 可视化图表
- **[feature_importance_heatmap.png](file:///Users/zitham_nie/Desktop/Easy_model/output/feature_importance_heatmap.png)**: 各靶点特征重要性热力图

## 文件命名规范

- 所有CSV文件均采用UTF-8编码
- 日期格式统一为YYYY-MM-DD
- 靶点ID格式为TAR_001至TAR_005
- 分子ID格式为MOL_XXXX或CAND_XXXX

## 更新说明

文件会随着模型训练和分析过程的进行而更新。建议在使用前检查文件的修改时间以确保使用的是最新结果。