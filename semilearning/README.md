# 直推学习模型 (Transductive Learning Model)

本目录包含基于图神经网络的直推学习模型实现，用于提高小样本场景下的分子活性预测能力。

## 模型介绍

直推学习（Transductive Learning）是半监督学习的一种形式，与传统的归纳学习不同，它在训练时就能"看到"测试数据的特征信息，只是没有测试数据的标签。通过这种方式，模型可以利用测试数据的结构信息来提高预测性能，特别适用于小样本场景。

### 核心思想

1. **全局图结构利用**：将训练集、验证集和测试集的所有分子构建成一个大的图结构
2. **信息传播**：通过图神经网络在整张图上进行信息传播，测试集节点可以从训练集节点获取有用信息
3. **标签传播**：利用已知标签节点的信息来帮助预测无标签节点

### 模型架构

- **图卷积网络**：使用多层GCN层提取节点（原子）特征
- **全局池化**：结合平均池化和最大池化获取分子级别的表示
- **多任务预测**：同时预测5个靶点的pIC50值

## 文件结构

```
semilearning/
├── transductive_model.py     # 直推学习模型定义
├── train_transductive.py     # 训练脚本
├── predict_transductive.py   # 预测脚本
├── README.md                 # 说明文档
└── experiments/              # 实验结果目录
```

## 使用方法

### 训练模型

```bash
python train_transductive.py --data-dir ../data --epochs 200 --hidden-dim 128
```

主要参数：
- `--data-dir`：数据目录路径
- `--epochs`：训练轮数
- `--hidden-dim`：隐藏层维度
- `--num-layers`：GCN层数
- `--dropout`：Dropout概率
- `--lr`：学习率
- `--device`：计算设备（cpu/cuda）

### 进行预测

```bash
python predict_transductive.py --model-path experiments/experiment_XXXXXX/transductive_model.pth --input ../data/molecule.smi --output predictions.csv
```

## 直推学习的优势

1. **更好的泛化能力**：通过利用测试集的结构信息，模型可以更好地理解整个化学空间的分布
2. **缓解小样本问题**：在训练样本较少的情况下，测试样本的结构信息可以帮助模型学习更好的表示
3. **信息共享**：相似分子之间的信息可以更好地传播，提高预测准确性

## 注意事项

1. **计算资源**：直推学习需要将所有数据加载到内存中，对内存要求较高
2. **隐私问题**：在实际应用中，如果测试数据涉及隐私，则不能使用直推学习
3. **适用场景**：主要适用于离线预测任务，在线预测任务无法使用直推学习

## 实验结果

通过直推学习方法，我们期望能够在小样本场景下获得更好的预测性能，特别是在测试集R2分数上有显著提升。