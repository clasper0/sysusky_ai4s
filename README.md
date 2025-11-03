# Molecular GCN for pIC50 Prediction

基于图卷积神经网络(GCN)的分子pIC50预测系统。该系统将分子的SMILES表示转换为图结构，并使用深度学习模型预测其生物活性(pIC50值)。


## 项目结构

```
sysusky_ai4s/
├── smiles_to_graph.py      # SMILES到图结构的转换
├── gcn_model.py            # GCN模型架构定义
├── data_loader.py          # 数据加载和预处理
├── trainer.py              # 训练循环和评估
├── model_utils.py          # 模型保存、加载和管理
├── inference.py            # 预测和推理接口
├── train.py               # 主训练脚本
├── predict.py             # 命令行预测脚本
├── visualize_molecules.py  # 分子可视化工具
├── README.md              # 项目说明文档
└── data/                  # 数据目录
    └── candidate.csv      # 候选分子数据
```

## 主要功能

### 1. 分子图结构转换 (`smiles_to_graph.py`)

- 将SMILES字符串转换为图结构
- 提取原子特征(原子类型、电荷、度数、价电子数等)
- 提取键特征(键类型、立体化学、环状态等)
- 优化性能，解决RDKit警告问题

### 2. GCN模型架构 (`gcn_model.py`)

- 基础单任务GCN模型
- 多任务学习版本
- 注意力机制和残差连接
- 全局池化策略(平均、最大、加和、注意力)

### 3. 数据处理管道 (`data_loader.py`)

- 分子数据集类
- 数据验证和预处理
- 训练/验证/测试数据分割
- 批处理和缓存优化

### 4. 训练系统 (`trainer.py`)

- 完整的训练循环
- 多种优化器和学习率调度器
- 早停和模型检查点
- 评估指标(RMSE, MAE, R²)
- 训练曲线可视化

### 5. 模型管理 (`model_utils.py`)

- 模型版本控制和注册
- 完整的模型保存/加载
- 模型元数据管理
- 预测接口

### 6. 推理引擎 (`inference.py`)

- 高级预测接口
- 批处理支持
- 模型集成
- 不确定性估计(MC Dropout)
- 结果可视化和分析

## 安装依赖

```bash
pip install torch torch-geometric
pip install rdkit-pypi
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 快速开始

### 1. 准备数据

确保在`data/`目录下有候选分子数据文件`candidate.csv`，包含以下列：

- `SMILES`: 分子的SMILES字符串
- `pIC50`: 目标活性值(或`IC50`，系统会自动转换)

### 2. 训练模型

```bash
# 使用默认设置训练
python train.py

# 自定义参数训练
python train.py \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-dims 256 512 1024 \
    --experiment-name my_experiment
```

### 3. 进行预测

```python
from inference import MolecularInferenceEngine

# 加载训练好的模型
engine = MolecularInferenceEngine(model_name="molecular_gcn_v1")

# 单分子预测
result = engine.predict("CCO")  # 乙醇
print(f"预测pIC50: {result['prediction']:.3f}")

# 批量预测
smiles_list = ["CCO", "c1ccccc1", "CC(=O)c1ccc2nc(-c3ccccc3)n(O)c2c1"]
results = engine.predict(smiles_list)
for result in results:
    if result['success']:
        print(f"{result['smiles']}: {result['prediction']:.3f}")
```

### 4. 命令行预测

```bash
# 从文件预测
python predict.py \
    --input molecules.csv \
    --output predictions.csv \
    --model molecular_gcn_v1 \
    --uncertainty \
    --visualize
```

## 模型架构

### 基础GCN模型

```
输入 (原子特征) → GCN层 → GAT注意力层 → 批归一化 → Dropout → ...
→ 全局池化 (注意力+平均+最大+加和) → 全连接层 → pIC50预测
```

### 特征工程

- **原子特征 (40维)**: 原子类型(23维) + 形式电荷 + 度数 + 氢键数 + 价电子数 + 芳香性 + 杂化类型(6维) + 环状态 + 原子质量
- **键特征 (10维)**: 键类型(5维) + 立体化学(6维) + 环状态 + 共轭性

## 训练配置

### 默认超参数

- 模型: MolecularGCN
- 隐藏层维度: [128, 256, 512]
- Dropout率: 0.2
- 优化器: Adam
- 学习率: 0.001
- 权重衰减: 1e-5
- 调度器: ReduceLROnPlateau
- 批大小: 32
- 早停耐心值: 50

### 评估指标

- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)

## 高级功能

### 1. 多任务学习

支持同时预测多个分子性质：

```python
from gcn_model import MultiTaskMolecularGCN

model = MultiTaskMolecularGCN(
    input_dim=40,
    hidden_dims=[128, 256, 512],
    output_dims={"pic50": 1, "logp": 1, "solubility": 1}
)
```

### 2. 不确定性估计

使用蒙特卡洛Dropout进行预测不确定性估计：

```python
results = engine.predict(
    smiles_list,
    return_uncertainty=True,
    n_samples=100
)
```

### 3. 模型集成

使用多个模型的集成进行更鲁棒的预测：

```python
engine = MolecularInferenceEngine(
    ensemble_models=["model_v1", "model_v2", "model_v3"]
)
```

### 4. 模型版本管理

完整的模型版本控制和元数据管理：

```python
from model_utils import ModelManager

manager = ModelManager()
manager.save_model(model, "my_model", metadata={"performance": {"rmse": 0.85}})
loaded_model = manager.load_model("my_model")
```

## 实验追踪

训练过程中会自动保存：

- 模型检查点
- 训练历史和指标
- 配置参数
- 训练曲线图
- 最佳模型

实验结果保存在`experiments/`目录下，按实验名称组织。

## 性能优化

### 1. 数据加载优化

- 图结构缓存
- 多进程数据加载
- 内存优化

### 2. 训练优化

- 梯度裁剪
- 学习率调度
- 早停机制
- 混合精度训练(可选)

### 3. 推理优化

- 批处理推理
- GPU加速
- 模型量化(可选)

## 故障排除

### 常见问题

1. **RDKit警告**: 已在`smiles_to_graph.py`中处理
2. **CUDA内存不足**: 减小batch_size或使用CPU训练
3. **数据加载失败**: 检查CSV文件格式和SMILES有效性
4. **模型加载错误**: 确保模型版本和配置匹配

### 调试模式

```bash
# 启用详细日志
python train.py --debug

# 小数据集测试
python train.py --batch-size 4 --epochs 5 --test-size 0.5
```

## 扩展开发

### 添加新模型

```python
# 在gcn_model.py中继承基类
class NewGCNModel(MolecularGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加新的层和功能
```

### 自定义数据预处理

```python
# 在data_loader.py中扩展数据集类
class CustomMolecularDataset(MolecularDataset):
    def __getitem__(self, idx):
        # 自定义数据加载逻辑
```

## 引用

如果使用本系统，请引用相关论文：

- Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
- Veličković et al. (2018): Graph Attention Networks
- Wu et al. (2018): A Comprehensive Survey on Graph Neural Networks

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 邮件联系项目维护者
