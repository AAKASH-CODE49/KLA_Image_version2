#!/usr/bin/env python
"""
Visualization utilities for image restoration results.

Provides:
- Comparison grids (input, restored, GT, residual)
- Metric plots (training curves, per-image metrics)
- Error analysis (histograms, heatmaps)
- Degradation analysis (per-type metrics)

Usage:
    from evaluation.visualizations import ResultsVisualizer
    
    viz = ResultsVisualizer(output_dir="results/v1/visuals")
    viz.plot_training_curves(metrics_file="results/v1/metrics.json")
    viz.create_comparison_grid(restored_dir, gt_dir, output_file="comparison.png")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy import stats


class ResultsVisualizer:
    """Visualization utilities for restoration results."""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Visualizer] Output directory: {self.output_dir}")
    
    def plot_training_curves(
        self,
        metrics_file: str,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            metrics_file: Path to metrics JSON file
            output_file: Output file path (None for auto)
            figsize: Figure size
        """
        # Load metrics
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        train_losses = metrics.get("train", [])
        val_losses = metrics.get("val", [])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
        axes[0].plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot log scale
        axes[1].semilogy(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
        axes[1].semilogy(epochs, val_losses, label="Val Loss", marker="s", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss (log scale)")
        axes[1].set_title("Training and Validation Loss (Log Scale)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which="both")
        
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = self.output_dir / "training_curves.png"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[Visualizer] Training curves saved: {output_file}")
    
    def create_comparison_grid(
        self,
        images: List[Tuple[np.ndarray, str]],
        output_file: Optional[str] = None,
        titles: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 4)
    ):
        """
        Create a comparison grid of images.
        
        Args:
            images: List of (image_array, label) tuples
            output_file: Output file path
            titles: Column titles
            figsize: Figure size
        """
        n_cols = len(images)
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for idx, (img, label) in enumerate(images):
            ax = axes[idx]
            
            # Ensure proper format
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                # (C, H, W) format
                img = np.transpose(img, (1, 2, 0))
            
            if img.ndim == 3 and img.shape[2] == 1:
                # Grayscale
                img = np.squeeze(img, axis=2)
                ax.imshow(img, cmap="gray")
            else:
                # RGB
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            
            ax.set_title(label, fontsize=12)
            ax.axis("off")
        
        if titles:
            fig.suptitle(" | ".join(titles), fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = self.output_dir / f"comparison_{len(images)}col.png"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[Visualizer] Comparison grid saved: {output_file}")
    
    def plot_metric_distribution(
        self,
        metrics_dict: Dict[str, List[float]],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot distribution of metrics across images.
        
        Args:
            metrics_dict: Dictionary of metric_name -> list of values
            output_file: Output file path
            figsize: Figure size
        """
        n_metrics = len(metrics_dict)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            
            ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(values), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(values):.3f}")
            ax.axvline(np.median(values), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(values):.3f}")
            
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{metric_name} Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(metrics_dict), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = self.output_dir / "metric_distribution.png"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[Visualizer] Metric distribution saved: {output_file}")
    
    def plot_residual_heatmap(
        self,
        restored: np.ndarray,
        gt: np.ndarray,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 4)
    ):
        """
        Plot residual error heatmap.
        
        Args:
            restored: Restored image (C, H, W) or (H, W)
            gt: Ground truth image (C, H, W) or (H, W)
            output_file: Output file path
            figsize: Figure size
        """
        # Convert to grayscale if needed
        if restored.ndim == 3:
            restored = np.mean(restored, axis=0)
        if gt.ndim == 3:
            gt = np.mean(gt, axis=0)
        
        residual = np.abs(restored - gt)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Restored
        im0 = axes[0].imshow(restored, cmap="gray")
        axes[0].set_title("Restored")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # GT
        im1 = axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Residual
        im2 = axes[2].imshow(residual, cmap="hot")
        axes[2].set_title(f"Absolute Error (max={residual.max():.4f})")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = self.output_dir / "residual_heatmap.png"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[Visualizer] Residual heatmap saved: {output_file}")
    
    def plot_metric_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Compare metrics across different models.
        
        Args:
            results_dict: Dict of model_name -> metrics
            output_file: Output file path
            figsize: Figure size
        """
        metrics_to_plot = ["psnr", "ssim", "inference_time_ms"]
        model_names = list(results_dict.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            values = []
            for model_name in model_names:
                if f"{metric}_mean" in results_dict[model_name]:
                    values.append(results_dict[model_name][f"{metric}_mean"])
                elif metric in results_dict[model_name]:
                    values.append(results_dict[model_name][metric])
                else:
                    values.append(0)
            
            bars = ax.bar(model_names, values, alpha=0.7, edgecolor="black")
            
            # Color bars
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel("Value")
            ax.set_title(f"{metric.upper()}")
            ax.grid(True, alpha=0.3, axis="y")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = self.output_dir / "model_comparison.png"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[Visualizer] Model comparison saved: {output_file}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser("Visualization utilities")
    parser.add_argument("--metrics-file", type=str, help="Metrics JSON file")
    parser.add_argument("--output-dir", type=str, default="results/visualizations")
    
    args = parser.parse_args()
    
    viz = ResultsVisualizer(output_dir=args.output_dir)
    
    if args.metrics_file:
        viz.plot_training_curves(args.metrics_file)


if __name__ == "__main__":
    main()
