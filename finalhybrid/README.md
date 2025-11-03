# Hybrid Molecular Property Prediction Model

## 项目概述

本项目实现了一个基于图注意力网络(GAT)的多任务分子属性预测模型，用于同时预测小分子化合物对多个药物靶点的pIC50值。该模型结合了分子结构信息和分子描述符特征，采用端到端的学习方式进行训练。

## 项目结构

```
.
├── data/                    # 数据文件目录
│   ├── candidate_hybrid.csv # 训练数据文件
│   └── ...
├── real_data/              # 真实数据源文件
│   ├── activity_train.csv  # 分子-靶点活性数据
│   ├── molecule.smi        # 分子SMILES数据
│   ├── property.csv        # 分子描述符数据
│   └── candidate.csv       # 待预测候选分子
├── model/                  # 模型文件保存目录
├── src/                    # 源代码目录
│   ├── hybrid_model_final.py  # 模型定义
│   ├── smiles_to_graph.py     # SMILES转图工具（包含GAT层）
│   ├── dataset_hybrid.py      # 数据集定义
│   └── data_loader.py         # 数据加载器
├── process_real_data.py    # 真实数据处理脚本
├── train_hybrid_final.py   # 模型训练脚本
├── predict_candidates.py   # 候选分子预测脚本
├── evaluate_model.py       # 模型评估脚本
└── README.md               # 项目说明文档
```

## 模型原理

### 1. 多任务学习框架
模型采用了多任务学习(multi-task learning)策略，同时预测小分子对5个不同药物靶点的pIC50值。这种设计能够利用任务间的共享信息，提高模型泛化能力和预测准确性。

### 2. 图神经网络表示
将每个分子表示为图结构：
- 节点：代表原子，包含原子类型、杂化状态等特征
- 边：代表化学键，包含键类型等信息
- 图神经网络逐层更新节点表示，捕获分子的局部和全局结构信息

### 3. 注意力机制
模型在网络中引入了图注意力网络(GAT)层，使得模型能够自适应地关注对预测更重要的原子和化学键。

### 4. 分子描述符融合
除了结构信息外，模型还融合了分子的理化性质描述符（分子量MolWt、脂水分配系数LogP、氢键受体数HBA、氢键供体数HBD），增强了模型对分子特性的理解。

## 模型架构

### 1. 多种图神经网络层融合
```
Input SMILES → Graph Representation → 
[GCN + GAT + GraphSAGE + GIN] Layers → 
Graph-level Embedding
```

- 使用多种图神经网络层（GCN、GAT、GraphSAGE、GIN）并行提取分子结构特征
- 通过加权平均融合不同网络的输出
- 通过注意力机制聚合节点信息得到图级别的表示

### 2. 分子描述符分支
```
Molecular Descriptors → Linear Layers → Descriptor Embedding
```

- 对输入的4维分子描述符进行线性变换和激活

### 3. 特征融合与预测头
```
[Graph Embedding, Descriptor Embedding] → Concatenation → 
Shared Hidden Layers → Task-specific Heads → 
[Target1_pred, Target2_pred, ..., Target5_pred]
```

- 将图嵌入和描述符嵌入拼接后送入共享隐藏层
- 通过5个任务特定的输出头分别预测各靶点的pIC50值

### 4. 损失函数
采用均方误差(MSE)损失函数，对5个任务的预测误差求和：
```
Loss = Σ(MSE(target_i_pred, target_i_true)) for i in 1..5
```

## 算法流程

### 1. 数据预处理
1. 解析SMILES字符串生成分子图表示
2. 标准化分子描述符特征
3. 将数据划分为训练集(65%)、验证集(15%)和测试集(20%)

### 2. 模型训练
1. 前向传播：通过多种GNN主干和描述符分支提取特征
2. 特征融合：拼接两种特征并经过共享隐藏层
3. 多任务预测：通过5个任务头输出预测值
4. 损失计算：计算总MSE损失
5. 反向传播：更新模型参数
6. 早停机制：监控验证集性能防止过拟合

### 3. 模型评估
使用以下指标评估模型性能：
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Pearson相关系数
- Spearman相关系数

## 项目脚本详解

### 1. 数据处理脚本

#### generate_example_data.py
生成示例训练数据，用于演示和测试模型功能。
- 创建包含5个靶点pIC50值和4个分子描述符的模拟数据
- 生成1000个分子样本
- 输出文件：[data/candidate_hybrid.csv](file:///d:/Codes/sysusky_ai4s/finalhybrid/data/candidate_hybrid.csv)

#### process_real_data.py
处理真实数据，整合三个数据源生成训练数据文件。
- 输入文件：
  - [real_data/activity_train.csv](file:///d:/Codes/sysusky_ai4s/finalhybrid/real_data/activity_train.csv)：分子对各靶点的pIC50值
  - [real_data/molecule.smi](file:///d:/Codes/sysusky_ai4s/finalhybrid/real_data/molecule.smi)：分子ID对应的SMILES
  - [real_data/property.csv](file:///d:/Codes/sysusky_ai4s/finalhybrid/real_data/property.csv)：分子的理化性质
- 输出文件：[data/candidate_hybrid.csv](file:///d:/Codes/sysusky_ai4s/finalhybrid/data/candidate_hybrid.csv)

### 2. 模型训练脚本

#### train.py
训练基础GCN模型。
- 支持单任务和多任务模式
- 包含完整的训练、验证和测试流程
- 支持早停机制和模型保存

#### train_hybrid_final.py
训练最终版混合模型（推荐使用）。
- 结合图注意力网络和分子描述符特征
- 多任务学习框架
- 包含完整的训练、验证和测试流程
- 支持早停机制和模型保存

### 3. 模型评估脚本

#### evaluate_model.py
评估模型性能并生成详细报告。
- 在测试集上评估模型
- 计算MAE、RMSE、Pearson相关系数、Spearman相关系数等指标
- 保存评估结果到CSV文件
- 保存测试集真实值和预测值用于后续分析

### 4. 预测脚本

#### predict_candidates.py
对候选分子进行pIC50值预测。
- 加载训练好的模型
- 读取候选分子文件
- 预测5个靶点的pIC50值
- 保存预测结果到CSV文件

## 使用方法

### 1. 数据准备
```bash
# 处理真实数据生成训练集
python process_real_data.py
```

### 2. 模型训练
```bash
# 训练混合模型
python train_hybrid_final.py
```

### 3. 模型评估
```bash
# 评估模型性能
python evaluate_model.py
```

### 4. 候选分子预测
```bash
# 预测候选分子活性
python predict_candidates.py
```

## 主要特点

1. **端到端学习**：从SMILES直接预测pIC50值，无需手工特征工程
2. **多任务预测**：同时预测多个靶点活性，提高数据利用效率
3. **多种图神经网络融合**：结合GCN、GAT、GraphSAGE和GIN的优势
4. **注意力机制**：使用注意力网络突出重要原子和键的作用
5. **特征融合**：结合图结构特征和分子描述符，增强表征能力
6. **可扩展性**：易于增加新的靶点或分子特征

## 依赖库

- PyTorch
- RDKit
- NetworkX
- Pandas
- NumPy
- Scikit-learn

## 注意力机制详解

模型中使用的图注意力网络(GAT)具有以下优势：

1. **自适应权重分配**：根据不同邻居节点的重要性自动分配注意力权重
2. **可解释性**：注意力权重可以帮助理解模型决策过程
3. **表达能力**：相比传统GCN，GAT具有更强的节点表示学习能力

## 当前局限性

1. 数据量较小（仅约160个样本）限制了模型性能
2. 图神经网络在小数据集上容易过拟合
3. 缺乏独立的测试集验证模型泛化能力

## 后续改进方向

1. 收集更多训练数据
2. 尝试更深的GNN架构或多头注意力机制
3. 引入残差连接和批归一化提升训练稳定性
4. 使用数据增强技术扩充训练集
5. 实现注意力权重可视化以增强模型可解释性