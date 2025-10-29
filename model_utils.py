"""
Model Saving, Loading, and Utility Functions
===========================================

This module provides utilities for model management,
including saving, loading, versioning, and deployment utilities.

Key Components:
1. Model checkpointing and versioning
2. Model loading for inference
3. Model analysis and interpretation
4. Export utilities for deployment
5. Model comparison and benchmarking
"""

import torch
import torch.nn as nn
import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from gcn_model import MolecularGCN, MultiTaskMolecularGCN, create_model
from smiles_to_graph import SMILESToGraph


class ModelManager:
    """
    Comprehensive model management utility for saving, loading,
    and versioning molecular GCN models.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            model_dir: Directory to store models and metadata
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Metadata file
        self.metadata_file = self.model_dir / "model_registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load or create model registry."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": {}, "latest_versions": {}}
            self._save_registry()

    def _save_registry(self):
        """Save model registry."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _generate_model_hash(self, model: nn.Module) -> str:
        """Generate hash for model architecture."""
        model_str = str(model)
        return hashlib.md5(model_str.encode()).hexdigest()[:8]

    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        version: str = None,
        metadata: Dict = None,
        include_optimizer: bool = False,
        optimizer: torch.optim.Optimizer = None
    ) -> str:
        """
        Save model with comprehensive metadata.

        Args:
            model: PyTorch model to save
            model_name: Name of the model
            version: Model version (auto-generated if None)
            metadata: Additional metadata to save
            include_optimizer: Whether to include optimizer state
            optimizer: Optimizer to save if include_optimizer=True

        Returns:
            Version string of saved model
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_hash = self._generate_model_hash(model)
            version = f"v{timestamp}_{model_hash}"

        # Create model directory
        model_path = self.model_dir / model_name / version
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        model_file = model_path / "model.pth"
        torch.save(model.state_dict(), model_file)

        # Save optimizer if requested
        if include_optimizer and optimizer is not None:
            optimizer_file = model_path / "optimizer.pth"
            torch.save(optimizer.state_dict(), optimizer_file)

        # Save full model checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_class": type(model).__name__,
            "model_config": self._extract_model_config(model),
        }

        if include_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        checkpoint_file = model_path / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_file)

        # Create metadata
        model_metadata = {
            "model_name": model_name,
            "version": version,
            "model_class": type(model).__name__,
            "model_config": self._extract_model_config(model),
            "model_size_bytes": self._get_model_size(model),
            "created_at": datetime.now().isoformat(),
            "path": str(model_path.relative_to(self.model_dir)),
            "saved_by": "ModelManager"
        }

        # Add custom metadata
        if metadata:
            model_metadata.update(metadata)

        # Save metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {}

        self.registry["models"][model_name][version] = {
            "path": str(model_path.relative_to(self.model_dir)),
            "metadata": model_metadata
        }

        self.registry["latest_versions"][model_name] = version
        self._save_registry()

        print(f"Model saved: {model_name}:{version}")
        print(f"Location: {model_path}")
        print(f"Size: {model_metadata['model_size_bytes'] / 1024 / 1024:.2f} MB")

        return version

    def _extract_model_config(self, model: nn.Module) -> Dict:
        """Extract model configuration."""
        config = {}

        if hasattr(model, 'input_dim'):
            config['input_dim'] = model.input_dim
        if hasattr(model, 'hidden_dims'):
            config['hidden_dims'] = model.hidden_dims
        if hasattr(model, 'output_dim'):
            config['output_dim'] = model.output_dim
        if hasattr(model, 'dropout_rate'):
            config['dropout_rate'] = model.dropout_rate
        if hasattr(model, 'use_batch_norm'):
            config['use_batch_norm'] = model.use_batch_norm
        if hasattr(model, 'use_residual'):
            config['use_residual'] = model.use_residual

        return config

    def load_model(
        self,
        model_name: str,
        version: str = None,
        device: str = "cpu",
        load_optimizer: bool = False
    ) -> Union[nn.Module, Tuple[nn.Module, Optional[torch.optim.Optimizer]]]:
        """
        Load model from registry.

        Args:
            model_name: Name of the model to load
            version: Version to load (latest if None)
            device: Device to load model on
            load_optimizer: Whether to load optimizer state

        Returns:
            Loaded model (and optimizer if load_optimizer=True)
        """
        # Get version
        if version is None:
            version = self.registry["latest_versions"].get(model_name)
            if version is None:
                raise ValueError(f"No versions found for model: {model_name}")

        # Check if model exists in registry
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_name}")

        if version not in self.registry["models"][model_name]:
            raise ValueError(f"Version {version} not found for model: {model_name}")

        # Load metadata
        model_info = self.registry["models"][model_name][version]
        metadata = model_info["metadata"]

        # Create model instance
        model_class_name = metadata["model_class"]
        model_config = metadata["model_config"]

        if model_class_name == "MolecularGCN":
            model = MolecularGCN(**model_config)
        elif model_class_name == "MultiTaskMolecularGCN":
            model = MultiTaskMolecularGCN(**model_config)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

        # Load model weights
        model_path = self.model_dir / model_info["path"]
        checkpoint_file = model_path / "checkpoint.pth"

        if checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model_file = model_path / "model.pth"
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        # Load optimizer if requested
        optimizer = None
        if load_optimizer:
            optimizer_file = model_path / "optimizer.pth"
            if optimizer_file.exists():
                # Create dummy optimizer (would need to know original optimizer config)
                optimizer = torch.optim.Adam(model.parameters())
                optimizer_state = torch.load(optimizer_file, map_location=device)
                optimizer.load_state_dict(optimizer_state)

        print(f"Model loaded: {model_name}:{version}")
        print(f"Device: {device}")
        print(f"Model class: {model_class_name}")

        if load_optimizer and optimizer is not None:
            return model, optimizer
        else:
            return model

    def list_models(self) -> Dict:
        """List all registered models."""
        models = {}
        for model_name, versions in self.registry["models"].items():
            models[model_name] = {
                "versions": list(versions.keys()),
                "latest": self.registry["latest_versions"].get(model_name),
                "count": len(versions)
            }
        return models

    def delete_model(self, model_name: str, version: str = None):
        """Delete model version or entire model."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_name}")

        if version is None:
            # Delete entire model
            model_path = self.model_dir / model_name
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)

            del self.registry["models"][model_name]
            if model_name in self.registry["latest_versions"]:
                del self.registry["latest_versions"][model_name]

            print(f"Deleted entire model: {model_name}")
        else:
            # Delete specific version
            if version not in self.registry["models"][model_name]:
                raise ValueError(f"Version {version} not found for model: {model_name}")

            version_path = self.model_dir / self.registry["models"][model_name][version]["path"]
            if version_path.exists():
                import shutil
                shutil.rmtree(version_path)

            del self.registry["models"][model_name][version]

            # Update latest version if necessary
            if self.registry["latest_versions"].get(model_name) == version:
                remaining_versions = list(self.registry["models"][model_name].keys())
                if remaining_versions:
                    self.registry["latest_versions"][model_name] = sorted(remaining_versions)[-1]
                else:
                    del self.registry["latest_versions"][model_name]

            print(f"Deleted version {version} of model: {model_name}")

        self._save_registry()


class ModelPredictor:
    """
    Prediction utility for trained molecular GCN models.
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        model_manager: ModelManager = None,
        device: str = "cpu"
    ):
        """
        Initialize predictor.

        Args:
            model: Model instance or model name (for loading)
            model_manager: ModelManager instance for loading models
            device: Device for inference
        """
        self.device = torch.device(device)

        if isinstance(model, str):
            if model_manager is None:
                model_manager = ModelManager()
            self.model = model_manager.load_model(model, device=device)
        else:
            self.model = model
            self.model.to(self.device)

        self.model.eval()
        self.converter = SMILESToGraph()

    def predict_single(self, smiles: str) -> Dict:
        """
        Predict pIC50 for a single SMILES string.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary with prediction and metadata
        """
        # Convert SMILES to graph
        graph = self.converter.smiles_to_graph(smiles)

        if graph is None:
            return {
                "smiles": smiles,
                "prediction": None,
                "error": "Invalid SMILES string",
                "success": False
            }

        # Prepare batch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph])
        batch = batch.to(self.device)

        # Make prediction
        with torch.no_grad():
            prediction = self.model(batch).cpu().item()

        return {
            "smiles": smiles,
            "prediction": float(prediction),
            "error": None,
            "success": True,
            "graph_info": {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.edge_index.shape[1] // 2
            }
        }

    def predict_batch(self, smiles_list: List[str]) -> List[Dict]:
        """
        Predict pIC50 for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of prediction dictionaries
        """
        results = []
        graphs = []
        valid_smiles = []

        # Convert SMILES to graphs
        for smiles in smiles_list:
            graph = self.converter.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_smiles.append(smiles)
            else:
                results.append({
                    "smiles": smiles,
                    "prediction": None,
                    "error": "Invalid SMILES string",
                    "success": False
                })

        # Batch prediction for valid molecules
        if graphs:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(graphs)
            batch = batch.to(self.device)

            with torch.no_grad():
                predictions = self.model(batch).cpu().numpy()

            for smiles, pred, graph in zip(valid_smiles, predictions, graphs):
                results.append({
                    "smiles": smiles,
                    "prediction": float(pred),
                    "error": None,
                    "success": True,
                    "graph_info": {
                        "num_nodes": graph.num_nodes,
                        "num_edges": graph.edge_index.shape[1] // 2
                    }
                })

        return results

    def predict_from_file(
        self,
        input_file: str,
        smiles_col: str = "SMILES",
        output_file: str = None
    ) -> pd.DataFrame:
        """
        Predict pIC50 from CSV file.

        Args:
            input_file: Input CSV file with SMILES
            smiles_col: Column name for SMILES
            output_file: Output CSV file (auto-generated if None)

        Returns:
            DataFrame with predictions
        """
        import pandas as pd

        # Load data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} molecules from {input_file}")

        # Get SMILES list
        smiles_list = df[smiles_col].tolist()

        # Make predictions
        print("Making predictions...")
        results = self.predict_batch(smiles_list)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Merge with original data
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.csv"

        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

        return output_df


def create_example_models():
    """Create and save example models for testing."""
    from gcn_model import MolecularGCN, MultiTaskMolecularGCN

    manager = ModelManager()

    # Single task model
    print("Creating single task model...")
    single_task_model = MolecularGCN(
        input_dim=40,
        hidden_dims=[128, 256, 512],
        output_dim=1,
        dropout_rate=0.2
    )

    manager.save_model(
        single_task_model,
        "molecular_gcn_v1",
        metadata={
            "task": "pIC50_prediction",
            "description": "Single task molecular GCN for pIC50 prediction",
            "performance": {"val_rmse": 0.85, "val_r2": 0.72}
        }
    )

    # Multi-task model
    print("Creating multi-task model...")
    multi_task_model = MultiTaskMolecularGCN(
        input_dim=40,
        hidden_dims=[128, 256, 512],
        output_dims={"pic50": 1, "logp": 1, "solubility": 1},
        dropout_rate=0.2
    )

    manager.save_model(
        multi_task_model,
        "multi_task_gcn_v1",
        metadata={
            "tasks": ["pIC50", "LogP", "Solubility"],
            "description": "Multi-task molecular GCN for multiple properties",
            "performance": {
                "pic50": {"val_rmse": 0.87, "val_r2": 0.71},
                "logp": {"val_rmse": 0.45, "val_r2": 0.82},
                "solubility": {"val_rmse": 0.62, "val_r2": 0.68}
            }
        }
    )

    print("Example models created successfully!")


if __name__ == "__main__":
    # Example usage
    print("Model Management System")
    print("=" * 50)

    # Create example models
    create_example_models()

    # Test model loading and prediction
    manager = ModelManager()
    models = manager.list_models()

    print("\nRegistered models:")
    for model_name, info in models.items():
        print(f"  {model_name}: {info['count']} versions (latest: {info['latest']})")

    # Test predictor
    if models:
        model_name = list(models.keys())[0]
        predictor = ModelPredictor(model_name, manager)

        # Test prediction
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)c1ccc2nc(-c3ccccc3)n(O)c2c1"]
        results = predictor.predict_batch(test_smiles)

        print(f"\nTest predictions for {model_name}:")
        for result in results:
            if result["success"]:
                print(f"  {result['smiles']}: {result['prediction']:.3f}")
            else:
                print(f"  {result['smiles']}: Error - {result['error']}")