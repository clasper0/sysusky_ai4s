"""
Inference and Prediction Interface for Molecular GCN Models
============================================================

This module provides a high-level interface for making predictions
with trained molecular GCN models, including batch processing,
API endpoints, and visualization utilities.

Key Components:
1. High-level prediction interface
2. Batch processing utilities
3. Model ensemble support
4. Prediction confidence estimation
5. Result visualization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from model_utils import ModelManager, ModelPredictor
from smiles_to_graph import SMILESToGraph
from gcn_model import MolecularGCN, MultiTaskMolecularGCN
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw


class MolecularInferenceEngine:
    """
    High-level inference engine for molecular GCN predictions.
    """

    def __init__(
        self,
        model_path: str = None,
        model_name: str = None,
        version: str = None,
        device: str = "auto",
        ensemble_models: List[str] = None,
        confidence_method: str = "mc_dropout"
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model file
            model_name: Name of registered model
            version: Model version (latest if None)
            device: Device for inference ("auto", "cpu", "cuda")
            ensemble_models: List of model names for ensemble prediction
            confidence_method: Method for confidence estimation
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_method = confidence_method
        self.ensemble_models = ensemble_models

        # Load models
        self.model_manager = ModelManager()
        self.predictors = {}

        if ensemble_models:
            # Load ensemble of models
            for model_name in ensemble_models:
                predictor = ModelPredictor(model_name, self.model_manager, device=self.device)
                self.predictors[model_name] = predictor
        elif model_name:
            # Load single model by name
            predictor = ModelPredictor(model_name, self.model_manager, device=self.device)
            self.predictors[model_name] = predictor
        elif model_path:
            # Load model from path
            self._load_model_from_path(model_path)

        self.converter = SMILESToGraph()

    def _load_model_from_path(self, model_path: str):
        """Load model from file path."""
        # This would need to be implemented based on the specific model format
        raise NotImplementedError("Loading from path not yet implemented")

    def predict(
        self,
        smiles: Union[str, List[str]],
        return_uncertainty: bool = False,
        n_samples: int = 100
    ) -> Union[Dict, List[Dict]]:
        """
        Make predictions for SMILES strings.

        Args:
            smiles: Single SMILES string or list of SMILES
            return_uncertainty: Whether to return uncertainty estimates
            n_samples: Number of MC dropout samples for uncertainty

        Returns:
            Prediction results
        """
        single_input = isinstance(smiles, str)
        smiles_list = [smiles] if single_input else smiles

        results = []

        for smiles_str in smiles_list:
            # Get predictions from all models
            model_predictions = []
            model_results = []

            for model_name, predictor in self.predictors.items():
                if return_uncertainty and self.confidence_method == "mc_dropout":
                    # MC Dropout for uncertainty estimation
                    predictions = self._mc_dropout_predict(predictor, smiles_str, n_samples)
                    model_predictions.append(predictions)
                    model_results.append({
                        "model_name": model_name,
                        "prediction": float(np.mean(predictions)),
                        "uncertainty": float(np.std(predictions)),
                        "samples": predictions
                    })
                else:
                    # Single prediction
                    result = predictor.predict_single(smiles_str)
                    if result["success"]:
                        model_predictions.append(result["prediction"])
                        model_results.append({
                            "model_name": model_name,
                            "prediction": result["prediction"],
                            "uncertainty": None
                        })

            # Aggregate results
            if model_predictions:
                if len(model_predictions) > 1:
                    # Ensemble prediction
                    ensemble_pred = np.mean(model_predictions)
                    ensemble_uncertainty = np.std(model_predictions)
                else:
                    # Single model prediction
                    ensemble_pred = model_predictions[0]
                    ensemble_uncertainty = model_results[0]["uncertainty"]

                result = {
                    "smiles": smiles_str,
                    "prediction": float(ensemble_pred),
                    "uncertainty": float(ensemble_uncertainty) if ensemble_uncertainty is not None else None,
                    "success": True,
                    "model_predictions": model_results,
                    "num_models": len(model_predictions)
                }
            else:
                result = {
                    "smiles": smiles_str,
                    "prediction": None,
                    "uncertainty": None,
                    "success": False,
                    "error": "All models failed to process this SMILES"
                }

            results.append(result)

        return results[0] if single_input else results

    def _mc_dropout_predict(
        self,
        predictor: ModelPredictor,
        smiles: str,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Monte Carlo Dropout prediction for uncertainty estimation.

        Args:
            predictor: Model predictor
            smiles: SMILES string
            n_samples: Number of dropout samples

        Returns:
            Array of predictions
        """
        # Convert SMILES to graph
        graph = self.converter.smiles_to_graph(smiles)
        if graph is None:
            return np.array([np.nan] * n_samples)

        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph])
        batch = batch.to(self.device)

        # Enable dropout during evaluation
        predictor.model.train()  # This enables dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = predictor.model(batch).cpu().item()
                predictions.append(pred)

        predictor.model.eval()  # Return to eval mode
        return np.array(predictions)

    def predict_from_file(
        self,
        input_file: str,
        output_file: str = None,
        smiles_col: str = "SMILES",
        return_uncertainty: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions from CSV file.

        Args:
            input_file: Input CSV file
            output_file: Output CSV file
            smiles_col: SMILES column name
            return_uncertainty: Whether to compute uncertainty estimates

        Returns:
            DataFrame with predictions
        """
        # Load input data
        df = pd.read_csv(input_file)
        smiles_list = df[smiles_col].tolist()

        print(f"Processing {len(smiles_list)} molecules...")

        # Make predictions
        results = self.predict(smiles_list, return_uncertainty=return_uncertainty)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Merge with original data
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Save results
        if output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.csv"

        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

        return output_df

    def visualize_predictions(
        self,
        results: Union[Dict, List[Dict]],
        output_file: str = None,
        show_molecules: bool = True,
        max_molecules: int = 10
    ):
        """
        Visualize prediction results.

        Args:
            results: Prediction results
            output_file: Output file for plot
            show_molecules: Whether to show molecule structures
            max_molecules: Maximum number of molecules to show
        """
        if isinstance(results, dict):
            results = [results]

        # Filter successful predictions
        successful = [r for r in results if r["success"]]

        if not successful:
            print("No successful predictions to visualize")
            return

        # Create subplots
        n_mols = min(len(successful), max_molecules)
        cols = min(n_mols, 5)
        rows = (n_mols + cols - 1) // cols

        if show_molecules:
            fig, axes = plt.subplots(rows * 2, cols, figsize=(15, 6 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

        if rows == 1 and not show_molecules:
            axes = axes.reshape(1, -1)

        for i, result in enumerate(successful[:n_mols]):
            col = i % cols
            row = i // cols

            # Get molecule
            mol = Chem.MolFromSmiles(result["smiles"])

            if show_molecules:
                # Molecule image
                ax_mol = axes[row * 2, col] if rows > 1 else axes[0, col]
                if mol:
                    img = Draw.MolToImage(mol, size=(200, 200))
                    ax_mol.imshow(img)
                ax_mol.set_title(f"SMILES: {result['smiles'][:20]}...")
                ax_mol.axis('off')

                # Prediction info
                ax_pred = axes[row * 2 + 1, col] if rows > 1 else axes[1, col]
                pred_text = f"pIC50: {result['prediction']:.3f}"
                if result['uncertainty'] is not None:
                    pred_text += f"\n± {result['uncertainty']:.3f}"
                if 'model_predictions' in result:
                    pred_text += f"\nModels: {result['num_models']}"
                ax_pred.text(0.5, 0.5, pred_text, ha='center', va='center',
                           transform=ax_pred.transAxes, fontsize=12)
                ax_pred.axis('off')
            else:
                # Just prediction info
                ax = axes[row, col] if rows > 1 else axes[0, col]
                pred_text = f"SMILES: {result['smiles'][:15]}...\n"
                pred_text += f"pIC50: {result['prediction']:.3f}"
                if result['uncertainty'] is not None:
                    pred_text += f"\n± {result['uncertainty']:.3f}"
                ax.text(0.5, 0.5, pred_text, ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.axis('off')

        # Hide empty subplots
        for i in range(n_mols, rows * cols):
            col = i % cols
            row = i // cols
            if show_molecules:
                axes[row * 2, col].axis('off')
                axes[row * 2 + 1, col].axis('off')
            else:
                axes[row, col].axis('off')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()

    def analyze_predictions(
        self,
        results: Union[Dict, List[Dict]],
        output_dir: str = None
    ) -> Dict:
        """
        Analyze prediction results and generate statistics.

        Args:
            results: Prediction results
            output_dir: Directory to save analysis plots

        Returns:
            Analysis statistics
        """
        if isinstance(results, dict):
            results = [results]

        # Filter successful predictions
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            print("No successful predictions to analyze")
            return {}

        # Extract predictions and uncertainties
        predictions = [r["prediction"] for r in successful]
        uncertainties = [r["uncertainty"] for r in successful if r["uncertainty"] is not None]

        # Calculate statistics
        stats = {
            "total_molecules": len(results),
            "successful_predictions": len(successful),
            "failed_predictions": len(failed),
            "success_rate": len(successful) / len(results),
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "median": float(np.median(predictions))
            }
        }

        if uncertainties:
            stats["uncertainty_stats"] = {
                "mean": float(np.mean(uncertainties)),
                "std": float(np.std(uncertainties)),
                "min": float(np.min(uncertainties)),
                "max": float(np.max(uncertainties)),
                "median": float(np.median(uncertainties))
            }

        # Create plots if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Prediction distribution
            plt.figure(figsize=(10, 6))
            plt.hist(predictions, bins=30, alpha=0.7, edgecolor='black')
            plt.title("Distribution of Predictions")
            plt.xlabel("pIC50 Prediction")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "prediction_distribution.png"), dpi=300)
            plt.close()

            # Uncertainty analysis
            if uncertainties:
                plt.figure(figsize=(10, 6))
                plt.scatter(predictions, uncertainties, alpha=0.6)
                plt.title("Prediction vs Uncertainty")
                plt.xlabel("pIC50 Prediction")
                plt.ylabel("Uncertainty")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, "uncertainty_analysis.png"), dpi=300)
                plt.close()

        return stats


def create_inference_script():
    """Create a simple inference script for command line use."""
    script_content = '''#!/usr/bin/env python3
"""
Command line interface for molecular property prediction.
"""

import argparse
import pandas as pd
from inference import MolecularInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Predict molecular properties using GCN models")
    parser.add_argument("--input", required=True, help="Input CSV file with SMILES")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--model", default="molecular_gcn_v1", help="Model name to use")
    parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    parser.add_argument("--uncertainty", action="store_true", help="Include uncertainty estimates")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")

    args = parser.parse_args()

    # Initialize inference engine
    engine = MolecularInferenceEngine(
        model_name=args.model,
        device=args.device
    )

    # Make predictions
    results_df = engine.predict_from_file(
        args.input,
        args.output,
        args.smiles_col,
        args.uncertainty
    )

    print(f"Predictions completed for {len(results_df)} molecules")

    # Visualization
    if args.visualize:
        results = results_df.to_dict('records')
        engine.visualize_predictions(
            results,
            output_file="predictions_visualization.png",
            show_molecules=True,
            max_molecules=15
        )

        stats = engine.analyze_predictions(results, output_dir="prediction_analysis")
        print("Prediction analysis:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
'''

    with open("predict.py", "w") as f:
        f.write(script_content)

    print("Command line inference script created: predict.py")
    print("Usage: python predict.py --input data.csv --model molecular_gcn_v1")


if __name__ == "__main__":
    print("Molecular Inference Engine")
    print("=" * 40)

    # Create command line script
    create_inference_script()

    # Example usage (would require actual trained models)
    print("\nExample usage:")
    print("1. From Python:")
    print("   engine = MolecularInferenceEngine(model_name='molecular_gcn_v1')")
    print("   results = engine.predict(['CCO', 'c1ccccc1'])")
    print("\n2. From command line:")
    print("   python predict.py --input molecules.csv --model molecular_gcn_v1 --uncertainty --visualize")