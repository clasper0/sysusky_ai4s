#!/usr/bin/env python3
"""
Training script for molecular GCN model.
======================================

This script provides a complete training pipeline for the
molecular GCN model, including data loading, model training,
evaluation, and saving.
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_data_splits
from gcn_model import create_model
from trainer import MolecularTrainer


def main():
    """Main training function with explicit parameters."""
    print("Molecular GCN Training")
    print("=" * 50)
    
    # Explicitly defined parameters
    DATA_PATH = "data/candidate_hybrid.csv"
    SMILES_COL = "SMILES"
    TARGET_COL = None  # Let data loader auto-detect target columns
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    BATCH_SIZE = 16
    EPOCHS = 30
    
    # Model parameters
    MODEL_TYPE = "multi_task"
    INPUT_DIM = 36
    HIDDEN_DIMS = [128, 256, 512]
    DROPOUT_RATE = 0.2
    USE_BATCH_NORM = True
    USE_RESIDUAL = True
    
    # Training parameters
    OPTIMIZER = "adam"
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    SCHEDULER = "plateau"
    PATIENCE = 20
    MIN_DELTA = 1e-4
    SAVE_EVERY = 10
    
    # Experiment parameters
    DEVICE = "cpu"
    EXPERIMENT_DIR = "experiments"
    EXPERIMENT_NAME = None  # Auto-generated
    
    print(f"Data file: {DATA_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()

    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file '{DATA_PATH}' not found!")
        print("Please make sure the data file exists.")
        return

    # Load data
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader, dataset = create_data_splits(
            DATA_PATH,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            batch_size=BATCH_SIZE,
            smiles_col=SMILES_COL,
            target_col=TARGET_COL
        )
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Show data statistics
    print(f"Dataset loaded successfully!")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print()

    # Determine output dimension
    try:
        sample_batch = next(iter(train_loader))
        actual_output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
        print(f"Detected output dimension: {actual_output_dim}")
    except Exception as e:
        print(f"Failed to detect output dimension: {e}")
        actual_output_dim = 5  # Default to 5 targets
        print(f"Using default output dimension: {actual_output_dim}")

    # Create model
    print("Creating model...")
    model_params = {
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "output_dim": actual_output_dim,
        "dropout_rate": DROPOUT_RATE,
        "use_batch_norm": USE_BATCH_NORM,
        "use_residual": USE_RESIDUAL
    }

    try:
        model = create_model(MODEL_TYPE, **model_params)
        print("Model created successfully!")
    except Exception as e:
        print(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Show model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {type(model).__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Create trainer
    try:
        trainer = MolecularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            experiment_dir=EXPERIMENT_DIR,
            experiment_name=EXPERIMENT_NAME
        )
        print("Trainer created successfully!")
    except Exception as e:
        print(f"Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Setup optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    try:
        trainer.setup_optimizer(
            OPTIMIZER,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        trainer.setup_scheduler(SCHEDULER)
        print("Optimizer and scheduler set up successfully!")
    except Exception as e:
        print(f"Failed to set up optimizer/scheduler: {e}")
        import traceback
        traceback.print_exc()
        return

    # Start training
    print("\nStarting training...")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Min delta: {MIN_DELTA}")
    print(f"Save checkpoint every {SAVE_EVERY} epochs")
    print()

    try:
        trainer.train(
            num_epochs=EPOCHS,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            save_every=SAVE_EVERY
        )

        print("\nTraining completed successfully!")
        print(f"Experiment saved to: {trainer.experiment_path}")

        # Final test set evaluation
        if test_loader is not None:
            print("\nFinal test set evaluation:")
            test_metrics = trainer.validate_epoch(test_loader)
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}")
            if 'r2' in test_metrics:
                print(f"  Test R²: {test_metrics['r2']:.3f}")

        # Save final model
        final_model_path = os.path.join(trainer.experiment_path, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Saving current progress to {trainer.experiment_path}")
        trainer.save_checkpoint(len(trainer.train_history['train_loss']), save_optimizer=True)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nTraining script finished.")


if __name__ == "__main__":
    main()