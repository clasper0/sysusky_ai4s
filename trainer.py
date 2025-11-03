"""
Training Loop and Evaluation Metrics for Molecular GCN
======================================================

This module implements the training loop, evaluation metrics,
and model checkpointing for the molecular GCN model.

Key Components:
1. Training and validation loops
2. Evaluation metrics (RMSE, MAE, R²)
3. Early stopping and model checkpointing
4. Learning rate scheduling
5. Experiment tracking utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
# warnings.filterwarnings("ignore")

from gcn_model import MolecularGCN, MultiTaskMolecularGCN
from data_loader import MolecularDataLoader


class MolecularTrainer:
    """
    Trainer class for molecular GCN models with comprehensive
    training utilities and evaluation metrics.
    """

    def __init__(
        self,
        model: Union[MolecularGCN, MultiTaskMolecularGCN],
        train_loader: MolecularDataLoader,
        val_loader: MolecularDataLoader,
        test_loader: Optional[MolecularDataLoader] = None,
        device: str = "auto",
        experiment_dir: str = "experiments",
        experiment_name: str = None
    ):
        """
        Initialize the trainer.

        Args:
            model: GCN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to use for training ("cpu", "cuda", or "auto")
            experiment_dir: Directory to save experiment results
            experiment_name: Name for this experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        print(f"Training on device: {self.device}")

        # Experiment setup
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_path = os.path.join(experiment_dir, self.experiment_name)
        os.makedirs(self.experiment_path, exist_ok=True)

        # Training history
        self.train_history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_mae": [],
            "val_mae": [],
            "train_r2": [],
            "val_r2": [],
            "learning_rate": [],
            "epoch_times": []
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        # Save initial config
        self._save_config()

    def _save_config(self):
        """Save experiment configuration."""
        config = {
            "model_type": type(self.model).__name__,
            "model_params": {
                "input_dim": getattr(self.model, "input_dim", None),
                "hidden_dims": getattr(self.model, "hidden_dims", None),
                "output_dim": getattr(self.model, "output_dim", None),
                "dropout_rate": getattr(self.model, "dropout_rate", None)
            },
            "dataset_sizes": {
                "train": len(self.train_loader.dataset),
                "val": len(self.val_loader.dataset),
                "test": len(self.test_loader.dataset) if self.test_loader else 0
            },
            "device": str(self.device),
            "experiment_path": self.experiment_path
        }

        config_path = os.path.join(self.experiment_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def setup_optimizer(
        self,
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        **optimizer_kwargs
    ) -> optim.Optimizer:
        """
        Setup optimizer for training.

        Args:
            optimizer_name: Name of optimizer ("adam", "sgd", "adamw")
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            **optimizer_kwargs: Additional optimizer parameters

        Returns:
            Configured optimizer
        """
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.optimizer = optimizer
        return optimizer

    def setup_scheduler(
        self,
        scheduler_name: str = "plateau",
        **scheduler_kwargs
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Setup learning rate scheduler.

        Args:
            scheduler_name: Name of scheduler ("plateau", "cosine", "step")
            **scheduler_kwargs: Additional scheduler parameters

        Returns:
            Configured scheduler
        """
        if scheduler_name.lower() == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True,
                **scheduler_kwargs
            )
        elif scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                **scheduler_kwargs
            )
        elif scheduler_name.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
                **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        self.scheduler = scheduler
        return scheduler

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy
        pred_np = predictions.cpu().detach().numpy()
        target_np = targets.cpu().detach().numpy()

        # Compute metrics
        mse = mean_squared_error(target_np, pred_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_np, pred_np)
        r2 = r2_score(target_np, pred_np)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        epoch_start_time = time.time()

        for batch in self.train_loader:
            batch = batch.to(self.device)
            targets = batch.y.view(-1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(batch)

            # Compute loss
            loss = nn.MSELoss()(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * len(targets)
            all_predictions.append(predictions)
            all_targets.append(targets)

        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.compute_metrics(all_predictions, all_targets)

        # Add average loss
        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics["loss"] = avg_loss

        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        metrics["epoch_time"] = epoch_time

        return metrics

    def validate_epoch(self, data_loader: MolecularDataLoader) -> Dict[str, float]:
        """
        Validate the model for one epoch.

        Args:
            data_loader: Data loader for validation/test

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                targets = batch.y.view(-1)

                # Forward pass
                predictions = self.model(batch)

                # Compute loss
                loss = nn.MSELoss()(predictions, targets)

                # Accumulate metrics
                total_loss += loss.item() * len(targets)
                all_predictions.append(predictions)
                all_targets.append(targets)

        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.compute_metrics(all_predictions, all_targets)

        # Add average loss
        avg_loss = total_loss / len(data_loader.dataset)
        metrics["loss"] = avg_loss

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        save_optimizer: bool = True
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            save_optimizer: Whether to save optimizer state
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_history": self.train_history
        }

        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if hasattr(self, "scheduler"):
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.experiment_path, "latest_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.experiment_path, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.train_history = checkpoint.get("train_history", {})

        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")

    def train(
        self,
        num_epochs: int,
        patience: int = 50,
        min_delta: float = 1e-4,
        save_every: int = 10
    ):
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Patience: {patience}, Min delta: {min_delta}")

        for epoch in range(1, num_epochs + 1):
            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch(self.val_loader)

            # Update learning rate
            if hasattr(self, "scheduler"):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record history
            self.train_history["train_loss"].append(train_metrics["loss"])
            self.train_history["val_loss"].append(val_metrics["loss"])
            self.train_history["train_rmse"].append(train_metrics["rmse"])
            self.train_history["val_rmse"].append(val_metrics["rmse"])
            self.train_history["train_mae"].append(train_metrics["mae"])
            self.train_history["val_mae"].append(val_metrics["mae"])
            self.train_history["train_r2"].append(train_metrics["r2"])
            self.train_history["val_r2"].append(val_metrics["r2"])
            self.train_history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            self.train_history["epoch_times"].append(train_metrics["epoch_time"])

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss - min_delta:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                is_best = False

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Print progress
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Train RMSE: {train_metrics['rmse']:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f} | "
                  f"Train R²: {train_metrics['r2']:.3f} | "
                  f"Val R²: {val_metrics['r2']:.3f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {train_metrics['epoch_time']:.1f}s "
                  f"{'*' if is_best else ''}")

            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                print(f"Best epoch: {self.best_epoch}, Best validation loss: {self.best_val_loss:.4f}")
                break

        # Final evaluation on test set if available
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            test_metrics = self.validate_epoch(self.test_loader)
            print(f"Test Results: Loss: {test_metrics['loss']:.4f}, "
                  f"RMSE: {test_metrics['rmse']:.4f}, "
                  f"MAE: {test_metrics['mae']:.4f}, "
                  f"R²: {test_metrics['r2']:.3f}")

        # Save final training history
        self._save_training_history()
        self._plot_training_curves()

    def _save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.experiment_path, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=2)

    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(self.train_history["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.train_history["val_loss"], label="Validation Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # RMSE curves
        axes[0, 1].plot(self.train_history["train_rmse"], label="Train RMSE")
        axes[0, 1].plot(self.train_history["val_rmse"], label="Validation RMSE")
        axes[0, 1].set_title("RMSE Curves")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # R² curves
        axes[1, 0].plot(self.train_history["train_r2"], label="Train R²")
        axes[1, 0].plot(self.train_history["val_r2"], label="Validation R²")
        axes[1, 0].set_title("R² Curves")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("R²")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate curve
        axes[1, 1].plot(self.train_history["learning_rate"])
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_path, "training_curves.png"), dpi=300)
        plt.close()

        print(f"Training curves saved to {self.experiment_path}/training_curves.png")


if __name__ == "__main__":
    # Example usage
    from data_loader import load_example_data
    from gcn_model import MolecularGCN

    print("Setting up example training session...")

    try:
        # Load data
        train_loader, val_loader, test_loader, dataset = load_example_data()

        # Create model
        model = MolecularGCN(
            input_dim=40,
            hidden_dims=[128, 256, 512],
            output_dim=1,
            dropout_rate=0.2
        )

        # Create trainer
        trainer = MolecularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name="example_training"
        )

        # Setup optimizer and scheduler
        trainer.setup_optimizer("adam", learning_rate=0.001, weight_decay=1e-5)
        trainer.setup_scheduler("plateau", patience=10)

        print("Setup complete. Ready to start training!")
        print("To start training, call: trainer.train(num_epochs=100)")

    except Exception as e:
        print(f"Error during setup: {e}")