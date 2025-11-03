#!/usr/bin/env python3
"""
Debug script to understand data and model output structure.
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_data_splits
from gcn_model import create_model

def main():
    print("Debugging data and model output structure...")
    
    # Load data
    DATA_PATH = "data/candidate_hybrid.csv"
    train_loader, val_loader, test_loader, dataset = create_data_splits(
        DATA_PATH,
        test_size=0.2,
        val_size=0.15,
        batch_size=16,
        target_col=None  # Let the data loader auto-detect target columns
    )
    
    print("Data loaded successfully!")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch structure:")
    print(f"  Batch type: {type(batch)}")
    print(f"  Batch attributes: {dir(batch)}")
    print(f"  Batch size: {batch.batch.size() if hasattr(batch, 'batch') else 'N/A'}")
    print(f"  Node features (x): {batch.x.shape}")
    print(f"  Edge index: {batch.edge_index.shape}")
    print(f"  Targets (y): {batch.y.shape}")
    print(f"  Targets sample: {batch.y[:5]}")
    
    # Create model
    model = create_model(
        "multi_task",
        input_dim=36,
        hidden_dims=[64, 128],
        output_dims={"target1": 1, "target2": 1, "target3": 1, "target4": 1, "target5": 1},
        dropout_rate=0.2
    )
    
    print(f"\nModel created: {type(model).__name__}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)
        print(f"\nModel output type: {type(output)}")
        if isinstance(output, dict):
            print(f"Model output keys: {list(output.keys())}")
            for key, value in output.items():
                print(f"  {key}: {value.shape}")
        else:
            print(f"Model output shape: {output.shape}")
    
    print("\nDebug completed!")

if __name__ == "__main__":
    main()