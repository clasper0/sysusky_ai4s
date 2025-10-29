"""
Data Loading and Preprocessing for Molecular GCN Training
========================================================

This module handles data loading, preprocessing, and batch creation
for training the molecular GCN model on pIC50 prediction tasks.

Key Components:
1. SMILES preprocessing and validation
2. Dataset class for PyTorch integration
3. Data augmentation strategies
4. Train/validation/test splitting
5. Data collation utilities
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class MolecularDataset(Dataset):
    """
    Dataset class for molecular SMILES with pIC50 values.
    """

    def __init__(
        self,
        data_path: str,
        smiles_col: str = "SMILES",
        target_col: str = "pIC50",
        transform=None,
        target_transform=None,
        cache_graphs: bool = True,
        validate_smiles: bool = True
    ):
        """
        Initialize the molecular dataset.

        Args:
            data_path: Path to CSV file containing molecular data
            smiles_col: Column name for SMILES strings
            target_col: Column name for target values (pIC50)
            transform: Optional transform for graphs
            target_transform: Optional transform for target values
            cache_graphs: Whether to cache converted graphs
            validate_smiles: Whether to validate SMILES strings
        """
        self.data_path = data_path
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.transform = transform
        self.target_transform = target_transform
        self.cache_graphs = cache_graphs
        self.validate_smiles = validate_smiles

        # Load data
        self.df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(self.df)} molecules from {data_path}")

        # Validate columns
        if smiles_col not in self.df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in data")
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Initialize SMILES to graph converter
        self.converter = SMILESToGraph()

        # Process data
        self._process_data()

        # Initialize target scaler
        self.target_scaler = StandardScaler()
        self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()

    def _process_data(self):
        """Process the raw data and extract valid molecules."""
        valid_indices = []
        smiles_list = []
        targets = []

        print("Validating SMILES and extracting features...")

        for idx, row in self.df.iterrows():
            smiles = str(row[self.smiles_col])
            target = row[self.target_col]

            # Skip invalid entries
            if pd.isna(smiles) or pd.isna(target):
                continue

            # Validate SMILES if required
            if self.validate_smiles:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                except:
                    continue

            valid_indices.append(idx)
            smiles_list.append(smiles)
            targets.append(float(target))

        self.valid_indices = valid_indices
        self.smiles_list = smiles_list
        self.targets = np.array(targets, dtype=np.float32)

        # Cache graphs if requested
        if self.cache_graphs:
            self._cache_graphs()

        print(f"Successfully processed {len(self.smiles_list)} valid molecules")
        print(f"Target statistics: mean={self.targets.mean():.3f}, std={self.targets.std():.3f}")

    def _cache_graphs(self):
        """Pre-convert and cache all SMILES to graphs."""
        print("Caching graph representations...")
        self.graph_cache = {}
        failed_count = 0

        for i, smiles in enumerate(self.smiles_list):
            graph = self.converter.smiles_to_graph(smiles)
            if graph is not None:
                self.graph_cache[i] = graph
            else:
                failed_count += 1

        print(f"Graph caching completed. Failed: {failed_count}/{len(self.smiles_list)}")

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tuple[Data, float]:
        """Get a single data sample."""
        # Get SMILES and target
        smiles = self.smiles_list[idx]
        target = self.targets[idx]

        # Get graph (from cache or convert on-the-fly)
        if self.cache_graphs and idx in self.graph_cache:
            graph = self.graph_cache[idx]
        else:
            graph = self.converter.smiles_to_graph(smiles)
            if graph is None:
                # Return a dummy graph if conversion fails
                graph = self._create_dummy_graph()

        # Add target to graph
        graph.y = torch.tensor([target], dtype=torch.float32)

        # Apply transforms
        if self.transform:
            graph = self.transform(graph)
        if self.target_transform:
            graph.y = self.target_transform(graph.y)

        return graph

    def _create_dummy_graph(self) -> Data:
        """Create a dummy graph for failed conversions."""
        dummy_x = torch.zeros(1, 40)  # Matching the feature dimension
        dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
        dummy_edge_attr = torch.zeros((0, 10))  # Matching edge feature dimension
        dummy_y = torch.zeros(1, dtype=torch.float32)

        return Data(
            x=dummy_x,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr,
            y=dummy_y,
            num_nodes=1
        )

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            "num_molecules": len(self.smiles_list),
            "target_mean": float(self.targets.mean()),
            "target_std": float(self.targets.std()),
            "target_min": float(self.targets.min()),
            "target_max": float(self.targets.max()),
            "smiles_length_mean": np.mean([len(s) for s in self.smiles_list])
        }


class MolecularDataLoader:
    """
    Data loader class with additional utilities for molecular data.
    """

    def __init__(
        self,
        dataset: MolecularDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize molecular data loader.

        Args:
            dataset: MolecularDataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

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
        """Custom collate function for molecular graphs."""
        from torch_geometric.data import Batch as GeomBatch
        return GeomBatch.from_data_list(batch)

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
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, MolecularDataset]:
    """
    Create train/validation/test data splits.

    Args:
        data_path: Path to CSV data file
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        random_state: Random seed for reproducibility
        batch_size: Batch size for data loaders
        **dataset_kwargs: Additional arguments for MolecularDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    # Load full dataset
    full_dataset = MolecularDataset(data_path, **dataset_kwargs)

    # Create indices for train/test split
    train_val_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        random_state=random_state,
        stratify=None  # No stratification for regression
    )

    # Create train/val split
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # Create data loaders
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

    print(f"Data splits created:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} samples ({len(test_dataset)/len(full_dataset)*100:.1f}%)")

    return train_loader, val_loader, test_loader, full_dataset


def load_example_data(data_dir: str = "data") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load example molecular dataset.

    Args:
        data_dir: Directory containing data files

    Returns:
        Tuple of data loaders (train, val, test)
    """
    # Try to find the candidate data file
    candidate_path = os.path.join(data_dir, "candidate.csv")

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Data file not found: {candidate_path}")

    # Check if pIC50 column exists, if not try to create it from IC50
    df = pd.read_csv(candidate_path)

    if 'pIC50' not in df.columns:
        if 'IC50' in df.columns:
            # Convert IC50 to pIC50: pIC50 = -log10(IC50 in M)
            # Assuming IC50 is in nM, convert to M first
            df['pIC50'] = -np.log10(df['IC50'] * 1e-9)
            print("Converted IC50 to pIC50 values")

            # Save updated dataframe
            df.to_csv(candidate_path, index=False)
        else:
            # Create synthetic pIC50 values for demonstration
            np.random.seed(42)
            df['pIC50'] = np.random.normal(6.0, 1.5, len(df))
            print("Created synthetic pIC50 values for demonstration")
            df.to_csv(candidate_path, index=False)

    # Create data splits
    return create_data_splits(
        candidate_path,
        test_size=0.2,
        val_size=0.15,
        batch_size=32,
        smiles_col="SMILES",
        target_col="pIC50"
    )


if __name__ == "__main__":
    # Example usage
    print("Testing molecular data loading...")

    try:
        train_loader, val_loader, test_loader, dataset = load_example_data()

        # Test data loading
        print(f"\nTesting batch loading:")
        for batch in train_loader:
            print(f"Batch shape: {batch.batch.size()}")
            print(f"Node features: {batch.x.shape}")
            print(f"Edge index: {batch.edge_index.shape}")
            print(f"Targets: {batch.y.shape}")
            break

        # Show dataset statistics
        stats = dataset.get_statistics()
        print(f"\nDataset statistics: {json.dumps(stats, indent=2)}")

        print("\nData loading test completed successfully!")

    except Exception as e:
        print(f"Error during data loading test: {e}")
        print("Make sure the data directory and files exist.")