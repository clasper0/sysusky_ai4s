"""
Graph Convolutional Network for pIC50 Prediction
==================================================

This module implements GCN models for predicting molecular pIC50 values
from SMILES strings converted to graph structures.

Key Components:
1. GCN layers with residual connections
2. Global pooling mechanisms
3. Attention mechanisms for interpretability
4. Multi-task learning support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import numpy as np


class MolecularGCN(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.

    Architecture:
    - Multiple GCN layers with residual connections
    - Global attention pooling
    - Fully connected layers for regression
    """

    def __init__(
        self,
        input_dim: int = 40,  # 原子特征维度
        hidden_dims: List[int] = [128, 256, 512],  # 隐藏层维度
        output_dim: int = 1,  # 输出维度 (pIC50值)
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_residual: bool = True,
        attention_heads: int = 4
    ):
        """
        Initialize the Molecular GCN model.

        Args:
            input_dim: Dimension of input atom features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (prediction target)
            dropout_rate: Dropout rate for regularization
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            attention_heads: Number of attention heads for GAT layers
        """
        super(MolecularGCN, self).__init__()    # 标准操作，调用父类 nn.Module 的构造函数

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()

        # Input layer
        self.gcn_layers.append(GCNConv(input_dim, hidden_dims[0]))
        self.gat_layers.append(GATConv(hidden_dims[0], hidden_dims[0] // attention_heads,
                                       heads=attention_heads, dropout=dropout_rate))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.gcn_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.gat_layers.append(GATConv(hidden_dims[i], hidden_dims[i] // attention_heads,
                                          heads=attention_heads, dropout=dropout_rate))

        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for dim in hidden_dims:
                self.batch_norms.append(nn.BatchNorm1d(dim))

        # Attention pooling
        self.attention_pooling = AttentionPooling(hidden_dims[-1])

        # Prediction heads
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 3, hidden_dims[-1] // 2),  # 3 pooling types concatenated
            nn.Dropout(dropout_rate),
            self._get_activation_layer(),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _get_activation_layer(self):
        """Get activation layer based on configuration."""
        if isinstance(self.activation, type(F.relu)):
            return nn.ReLU()
        elif isinstance(self.activation, type(F.gelu)):
            return nn.GELU()
        elif isinstance(self.activation, type(F.leaky_relu)):
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data: PyTorch Geometric Data object with x, edge_index, edge_attr

        Returns:
            torch.Tensor: Predicted pIC50 values
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Store initial input for residual connection
        x_in = x

        # Apply GCN and GAT layers
        for i, (gcn_layer, gat_layer) in enumerate(zip(self.gcn_layers, self.gat_layers)):
            # GCN layer
            x = gcn_layer(x, edge_index, edge_attr)

            # Apply batch norm if enabled
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            # Apply activation and dropout
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # GAT layer for attention mechanism
            x_att = gat_layer(x, edge_index)
            x = x + x_att  # Residual connection with attention

            # Residual connection
            if self.use_residual and i < len(self.gcn_layers) - 1:
                if x.size(-1) == x_in.size(-1):
                    x = x + x_in
                x_in = x

        # Global pooling with attention
        graph_embedding = self.attention_pooling(x, batch)

        # Additional pooling strategies for better representation
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)

        # Combine pooling results
        combined_pool = torch.cat([graph_embedding, mean_pool, max_pool, add_pool], dim=1)

        # Final prediction
        output = self.prediction_head(combined_pool)

        return output.squeeze(-1)  # Remove last dimension for regression


class AttentionPooling(nn.Module):
    """
    Attention-based global pooling mechanism.
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize attention pooling.

        Args:
            hidden_dim: Dimension of node features
        """
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to node features.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            torch.Tensor: Graph-level representations [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0)

        # Calculate attention weights
        attention_weights = self.attention(x)  # [num_nodes, 1]

        # Create batch mask
        batch_mask = F.one_hot(batch, num_classes=batch_size).float()  # [num_nodes, batch_size]

        # Normalize attention weights within each batch
        attention_weights = attention_weights * batch_mask
        attention_sums = attention_weights.sum(dim=0, keepdim=True)
        attention_weights = attention_weights / (attention_sums + 1e-8)

        # Weighted sum of node features
        graph_embedding = torch.matmul(attention_weights.t(), x)  # [batch_size, hidden_dim]

        return graph_embedding


class MultiTaskMolecularGCN(MolecularGCN):
    """
    Multi-task version of MolecularGCN for predicting multiple properties simultaneously.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dims: List[int] = [128, 256, 512],
        output_dims: Dict[str, int] = {"pic50": 1, "logp": 1, "solubility": 1},
        **kwargs
    ):
        """
        Initialize multi-task GCN model.

        Args:
            input_dim: Dimension of input atom features
            hidden_dims: List of hidden layer dimensions
            output_dims: Dictionary of task names to output dimensions
            **kwargs: Additional arguments passed to parent class
        """
        super(MultiTaskMolecularGCN, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] // 2,  # Shared embedding dimension
            **kwargs
        )

        self.tasks = list(output_dims.keys())
        self.output_dims = output_dims

        # Task-specific prediction heads
        self.task_heads = nn.ModuleDict()
        for task, dim in output_dims.items():
            self.task_heads[task] = nn.Sequential(
                nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4),
                nn.Dropout(self.dropout_rate),
                self._get_activation_layer(),
                nn.Linear(hidden_dims[-1] // 4, dim)
            )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Dict[str, torch.Tensor]: Dictionary of task predictions
        """
        # Get shared graph embedding (reusing parent class logic)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_in = x

        # Apply GCN and GAT layers (same as parent class)
        for i, (gcn_layer, gat_layer) in enumerate(zip(self.gcn_layers, self.gat_layers)):
            x = gcn_layer(x, edge_index, edge_attr)

            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            x_att = gat_layer(x, edge_index)
            x = x + x_att

            if self.use_residual and i < len(self.gcn_layers) - 1:
                if x.size(-1) == x_in.size(-1):
                    x = x + x_in
                x_in = x

        # Global pooling
        graph_embedding = self.attention_pooling(x, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)

        combined_pool = torch.cat([graph_embedding, mean_pool, max_pool, add_pool], dim=1)

        # Get shared embedding
        shared_embedding = self.prediction_head[0](combined_pool)

        # Task-specific predictions
        predictions = {}
        for task in self.tasks:
            task_output = self.task_heads[task](shared_embedding)
            predictions[task] = task_output.squeeze(-1)

        return predictions


def create_model(model_type: str = "single_task", **kwargs) -> nn.Module:
    """
    Factory function to create GCN models.

    Args:
        model_type: Type of model ("single_task" or "multi_task")
        **kwargs: Additional model parameters

    Returns:
        nn.Module: Initialized model
    """
    if model_type == "single_task":
        return MolecularGCN(**kwargs)
    elif model_type == "multi_task":
        return MultiTaskMolecularGCN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    model = MolecularGCN(
        input_dim=40,
        hidden_dims=[128, 256, 512],
        output_dim=1,
        dropout_rate=0.2
    )

    print("Model architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")