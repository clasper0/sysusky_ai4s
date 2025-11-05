"""
分子GCN训练的数据加载和预处理模块
====================================

该模块负责处理分子GCN模型在pIC50预测任务中的数据加载、预处理和批次创建。

主要组件:
1. SMILES预处理和验证
2. 用于PyTorch集成的数据集类
3. 数据增强策略
4. 训练/验证/测试数据分割
5. 数据整理工具
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
import os
import json
from smiles_to_graph import SMILESToGraph
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class MolecularDataset(Dataset):
    """
    分子SMILES和pIC50值的数据集类
    """

    def __init__(
        self,
        data_path: str,
        smiles_col: str = "SMILES",
        target_col: str = "pIC50",
        descriptor_cols: List[str] = None,
        transform=None,
        target_transform=None,
        cache_graphs: bool = True,
        validate_smiles: bool = True
    ):
        """
        初始化分子数据集

        参数:
            data_path: 包含分子数据的CSV文件路径
            smiles_col: SMILES字符串的列名
            target_col: 目标值(pIC50)的列名，多个列以逗号分隔
            descriptor_cols: 分子描述符列名列表
            transform: 图的可选变换
            target_transform: 目标值的可选变换
            cache_graphs: 是否缓存转换后的图
            validate_smiles: 是否验证SMILES字符串
        """
        self.data_path = data_path
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.descriptor_cols = descriptor_cols if descriptor_cols is not None else []
        self.transform = transform
        self.target_transform = target_transform
        self.cache_graphs = cache_graphs
        self.validate_smiles = validate_smiles

        # 加载数据
        self.df = pd.read_csv(data_path)
        print(f"从 {data_path} 加载了包含 {len(self.df)} 个分子的数据集")

        # 验证SMILES列
        if smiles_col not in self.df.columns:
            raise ValueError(f"数据中未找到SMILES列 '{smiles_col}'")
            
        # 处理目标列
        if target_col is None:
            # 查找所有pIC50相关的列
            target_cols = [col for col in self.df.columns if 'pic50' in col.lower()]
            if not target_cols:
                # 使用前5个数值列作为目标
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                target_cols = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        else:
            target_cols = [col.strip() for col in target_col.split(',')] if isinstance(target_col, str) else [target_col]
        
        self.target_cols = target_cols
        for col in self.target_cols:
            if col not in self.df.columns:
                raise ValueError(f"数据中未找到目标列 '{col}'")

        # 验证描述符列
        invalid_descriptors = [col for col in self.descriptor_cols if col not in self.df.columns]
        if invalid_descriptors:
            print(f"警告: 以下描述符列不存在，将被忽略: {invalid_descriptors}")
            self.descriptor_cols = [col for col in self.descriptor_cols if col in self.df.columns]

        # 初始化SMILES到图的转换器
        self.converter = SMILESToGraph()

        # 处理数据
        self._process_data()


    def _process_data(self):
        """处理原始数据并提取有效的分子"""
        valid_indices = []
        smiles_list = []
        targets = []
        descriptors = []

        print("正在验证SMILES并提取特征...")

        for idx, row in self.df.iterrows():
            smiles = str(row[self.smiles_col])
            
            # 检查目标值是否有效
            target_values = []
            skip_row = False
            for col in self.target_cols:
                target_val = row[col]
                if pd.isna(target_val):
                    skip_row = True
                    break
                target_values.append(float(target_val))
            
            if skip_row:
                continue

            # 验证SMILES
            if self.validate_smiles:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                except Exception as e:
                    print(f"SMILES验证失败 '{smiles}': {e}")
                    continue

            # 提取分子描述符
            desc_values = []
            if self.descriptor_cols:
                for col in self.descriptor_cols:
                    desc_val = row[col]
                    if pd.isna(desc_val):
                        skip_row = True
                        break
                    desc_values.append(float(desc_val))
                
                if skip_row:
                    continue

            valid_indices.append(idx)
            smiles_list.append(smiles)
            targets.append(target_values)
            descriptors.append(desc_values)

        self.valid_indices = valid_indices
        self.smiles_list = smiles_list
        self.targets = np.array(targets, dtype=np.float32)
        self.descriptors = np.array(descriptors, dtype=np.float32) if self.descriptor_cols else np.array([]).reshape(len(smiles_list), 0)

        # 如果需要，缓存图
        if self.cache_graphs:
            self._cache_graphs()

        print(f"成功处理了 {len(self.smiles_list)} 个有效分子")
        target_means = np.mean(self.targets, axis=0)
        target_stds = np.std(self.targets, axis=0)
        print(f"目标值统计: 均值={target_means}, 标准差={target_stds}")
        if self.descriptor_cols:
            desc_means = np.mean(self.descriptors, axis=0)
            desc_stds = np.std(self.descriptors, axis=0)
            print(f"描述符统计: 均值={desc_means}, 标准差={desc_stds}")

    def _cache_graphs(self):
        """预转换并缓存所有SMILES到图的转换"""
        print("正在缓存图表示...")
        self.graph_cache = {}
        failed_count = 0

        for i, smiles in enumerate(self.smiles_list):
            graph = self.converter.smiles_to_graph(smiles)
            if graph is not None:
                self.graph_cache[i] = graph
            else:
                failed_count += 1

        print(f"图缓存完成。失败: {failed_count}/{len(self.smiles_list)}")

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Data:
        """获取单个数据样本"""
        # 获取SMILES和目标值
        smiles = self.smiles_list[idx]
        target = self.targets[idx]

        # 获取图（从缓存或实时转换）
        if self.cache_graphs and idx in self.graph_cache:
            graph = self.graph_cache[idx].clone()
        else:
            graph = self.converter.smiles_to_graph(smiles)
            if graph is None:
                # 如果转换失败，返回一个虚拟图
                graph = self._create_dummy_graph()

        # 添加目标到图中
        graph.y = torch.tensor(target, dtype=torch.float32)
        
        # 添加分子描述符到图中（如果有）
        if len(self.descriptor_cols) > 0 and self.descriptors.size > 0:
            graph.descriptors = torch.tensor(self.descriptors[idx], dtype=torch.float32)
        else:
            # 确保即使没有描述符也创建一个空的张量
            graph.descriptors = torch.empty(0, dtype=torch.float32)

        # 应用变换
        if self.transform:
            graph = self.transform(graph)
        if self.target_transform:
            graph.y = self.target_transform(graph.y)

        return graph

    def _create_dummy_graph(self) -> Data:
        """为转换失败创建虚拟图"""
        dummy_x = torch.zeros(1, 36)  # 匹配特征维度
        dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
        dummy_edge_attr = torch.zeros((0, 10))  # 匹配边特征维度
        dummy_y = torch.zeros(len(self.target_cols), dtype=torch.float32)  # 匹配多任务输出维度
        dummy_descriptors = torch.empty(0, dtype=torch.float32)  # 空描述符张量

        return Data(
            x=dummy_x,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr,
            y=dummy_y,
            descriptors=dummy_descriptors,
            num_nodes=1
        )

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        target_means = np.mean(self.targets, axis=0)
        target_stds = np.std(self.targets, axis=0)
        
        stats = {
            "分子数量": len(self.smiles_list),
            "目标数量": len(self.target_cols),
            "SMILES长度均值": np.mean([len(s) for s in self.smiles_list])
        }
        
        # 为每个目标添加统计信息
        for i, col in enumerate(self.target_cols):
            stats[f"目标_{col}_均值"] = float(target_means[i])
            stats[f"目标_{col}_标准差"] = float(target_stds[i])
            stats[f"目标_{col}_最小值"] = float(np.min(self.targets[:, i]))
            stats[f"目标_{col}_最大值"] = float(np.max(self.targets[:, i]))
            
        return stats


class MolecularDataLoader:
    """
    分子数据的加载器类，带有额外的实用工具
    """

    def __init__(
        self,
        dataset: MolecularDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        follow_batch: List[str] = None
    ):
        """
        初始化分子数据加载器

        参数:
            dataset: MolecularDataset实例
            batch_size: 训练的批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数量
            pin_memory: 是否为GPU传输锁定内存
            drop_last: 是否丢弃最后一个不完整的批次
            follow_batch: 需要跟踪的批次信息列表
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.follow_batch = follow_batch or []

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """分子图的自定义整理函数"""
        from torch_geometric.data import Batch as GeomBatch
        return GeomBatch.from_data_list(batch, follow_batch=self.follow_batch)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def create_data_splits(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    batch_size: int = 32,
    descriptor_cols: List[str] = None,
    **dataset_kwargs
) -> Tuple[MolecularDataLoader, MolecularDataLoader, MolecularDataLoader, MolecularDataset]:
    """
    创建训练/验证/测试数据分割

    参数:
        data_path: CSV数据文件路径
        test_size: 用于测试的数据比例
        val_size: 用于验证的训练数据比例
        random_state: 用于可重现性的随机种子
        batch_size: 数据加载器的批次大小
        descriptor_cols: 分子描述符列名列表
        **dataset_kwargs: MolecularDataset的其他参数

    返回:
        (train_loader, val_loader, test_loader, full_dataset)元组
    """
    # 加载完整数据集
    full_dataset = MolecularDataset(
        data_path, 
        descriptor_cols=descriptor_cols,
        **dataset_kwargs
    )

    # 创建训练/测试分割
    train_val_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        random_state=random_state
    )

    # 创建训练/验证分割
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    # 创建子集数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # 创建数据加载器
    train_loader = MolecularDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = MolecularDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = MolecularDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"数据分割创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本 ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  验证集: {len(val_dataset)} 样本 ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  测试集: {len(test_dataset)} 样本 ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
    
    # 打印描述符信息
    if descriptor_cols:
        print(f"  使用的分子描述符: {descriptor_cols}")

    return train_loader, val_loader, test_loader, full_dataset


def load_example_data(data_dir: str = "data") -> Tuple[MolecularDataLoader, MolecularDataLoader, MolecularDataLoader]:
    """
    加载示例分子数据集

    参数:
        data_dir: 包含数据文件的目录

    返回:
        数据加载器元组 (train, val, test)
    """
    # 尝试找到候选数据文件
    candidate_path = os.path.join(data_dir, "candidate.csv")

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"未找到数据文件: {candidate_path}")

    # 检查是否存在pIC50列，如果不存在则尝试从IC50创建
    df = pd.read_csv(candidate_path)

    # 查找所有pIC50相关的列
    pic50_cols = [col for col in df.columns if 'pic50' in col.lower()]
    
    if not pic50_cols:
        # 为演示创建合成的pIC50值
        np.random.seed(42)
        for i in range(5):  # 创建5个靶点的pIC50值
            df[f'target{i+1}_pIC50'] = np.random.normal(5.0, 1.5, len(df))
        pic50_cols = [f'target{i+1}_pIC50' for i in range(5)]
    else:
        # 确保只有5个靶点的pIC50值
        pic50_cols = pic50_cols[:5]
    
    # 定义更多分子描述符列
    descriptor_features = [
        'MolWt', 'LogP', 'HBA', 'HBD', 'TPSA', 
        'RotatableBonds', 'MolLogP', 'HeavyAtomCount',
        'RingCount', 'HydrogenBondDonorCount', 'HydrogenBondAcceptorCount'
    ]
    
    # 检查哪些描述符列存在
    existing_descriptors = [col for col in descriptor_features if col in df.columns]
    if not existing_descriptors:
        # 如果没有现有描述符，则创建一些合成数据
        np.random.seed(42)
        df['MolWt'] = np.random.normal(350, 50, len(df))
        df['LogP'] = np.random.normal(3.0, 1.0, len(df))
        df['HBA'] = np.random.randint(2, 8, len(df))
        df['HBD'] = np.random.randint(1, 3, len(df))
        df['TPSA'] = np.random.normal(80, 20, len(df))
        existing_descriptors = ['MolWt', 'LogP', 'HBA', 'HBD', 'TPSA']
    
    # 选择需要的列
    feature_cols = ['SMILES'] + pic50_cols + existing_descriptors
    
    # 更新数据框，只保留需要的列
    df = df[feature_cols]
    
    # 保存修改后的数据
    df.to_csv(candidate_path, index=False)

    # 创建数据分割
    return create_data_splits(
        candidate_path,
        test_size=0.2,
        val_size=0.15,
        batch_size=32,
        smiles_col="SMILES",
        target_col=','.join(pic50_cols),
        descriptor_cols=existing_descriptors
    )


if __name__ == "__main__":
    # 示例用法
    print("正在测试分子数据加载...")

    try:
        train_loader, val_loader, test_loader, dataset = load_example_data()

        # 测试数据加载
        print(f"\n正在测试批次加载:")
        for batch in train_loader:
            print(f"批次形状: {batch.batch.size()}")
            print(f"节点特征: {batch.x.shape}")
            print(f"边索引: {batch.edge_index.shape}")
            print(f"目标值: {batch.y.shape}")  # 现在是多目标
            break

        # 显示数据集统计信息
        stats = dataset.get_statistics()
        print(f"\n数据集统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")

        print("\n数据加载测试成功完成!")

    except Exception as e:
        print(f"数据加载测试期间出错: {e}")
        print("请确保数据目录和文件存在。")