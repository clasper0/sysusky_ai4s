#!/usr/bin/env python3
"""
Training script for molecular GCN model.
======================================

This script provides a complete training pipeline for the
molecular GCN model, including data loading, model training,
evaluation, and saving.
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_example_data, create_data_splits
from gcn_model import MolecularGCN, create_model
from trainer import MolecularTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train molecular GCN model")

    # Data arguments
    parser.add_argument("--data-path", default=None,
                       help="Path to CSV data file (default: uses example data)")
    parser.add_argument("--smiles-col", default="SMILES",
                       help="SMILES column name in data file")
    parser.add_argument("--target-col", default="pIC50",
                       help="Target column name in data file")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Fraction of data for testing")
    parser.add_argument("--val-size", type=float, default=0.15,
                       help="Fraction of training data for validation")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")

    # Model arguments
    parser.add_argument("--model-type", default="single_task",
                       choices=["single_task", "multi_task"],
                       help="Type of model to train")
    parser.add_argument("--input-dim", type=int, default=40,
                       help="Input feature dimension")
    parser.add_argument("--hidden-dims", nargs="+", type=int,
                       default=[128, 256, 512],
                       help="Hidden layer dimensions")
    parser.add_argument("--output-dim", type=int, default=1,
                       help="Output dimension")
    parser.add_argument("--dropout-rate", type=float, default=0.2,
                       help="Dropout rate")
    parser.add_argument("--use-batch-norm", action="store_true", default=True,
                       help="Use batch normalization")
    parser.add_argument("--use-residual", action="store_true", default=True,
                       help="Use residual connections")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--optimizer", default="adam",
                       choices=["adam", "adamw", "sgd"],
                       help="Optimizer to use")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--scheduler", default="plateau",
                       choices=["plateau", "cosine", "step"],
                       help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                       help="Minimum improvement for early stopping")

    # Experiment arguments
    parser.add_argument("--experiment-name", default=None,
                       help="Experiment name (auto-generated if None)")
    parser.add_argument("--experiment-dir", default="experiments",
                       help="Directory to save experiments")
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    return parser.parse_args()


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    print("Molecular GCN Training")
    print("=" * 50)
    print(f"Experiment: {args.experiment_name or 'auto-generated'}")
    print(f"Device: {args.device}")
    print(f"Model type: {args.model_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print()

    # Load data
    print("Loading data...")
    if args.data_path:
        # Custom data file
        train_loader, val_loader, test_loader, dataset = create_data_splits(
            args.data_path,
            test_size=args.test_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            smiles_col=args.smiles_col,
            target_col=args.target_col
        )
    else:
        # Example data
        train_loader, val_loader, test_loader, dataset = load_example_data()

    # Get data statistics
    print(f"Dataset loaded successfully!")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print()

    # Create model
    print("Creating model...")
    model_params = {
        "input_dim": args.input_dim,
        "hidden_dims": args.hidden_dims,
        "output_dim": args.output_dim,
        "dropout_rate": args.dropout_rate,
        "use_batch_norm": args.use_batch_norm,
        "use_residual": args.use_residual
    }

    model = create_model(args.model_type, **model_params)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {type(model).__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Create trainer
    trainer = MolecularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        experiment_dir=args.experiment_dir,
        experiment_name=args.experiment_name
    )

    # Setup optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    trainer.setup_optimizer(
        args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    trainer.setup_scheduler(args.scheduler)

    # Start training
    print("\nStarting training...")
    print(f"Early stopping patience: {args.patience}")
    print(f"Min delta: {args.min_delta}")
    print(f"Save checkpoint every {args.save_every} epochs")
    print()

    try:
        trainer.train(
            num_epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            save_every=args.save_every
        )

        print("\nTraining completed successfully!")
        print(f"Experiment saved to: {trainer.experiment_path}")

        # Print final results
        if test_loader is not None:
            print("\nFinal test set evaluation:")
            test_metrics = trainer.validate_epoch(test_loader)
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}")
            print(f"  Test R²: {test_metrics['r2']:.3f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current progress
        print(f"Saving current progress to {trainer.experiment_path}")
        trainer.save_checkpoint(len(trainer.train_history['train_loss']), save_optimizer=True)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nTraining script finished.")


if __name__ == "__main__":
    main()