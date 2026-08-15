#!/usr/bin/env python
"""Diagnostic script to inspect data ranges and model behavior."""

import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_paired_images, PairedImageDataset
from models import UnifiedRestorerV2

def diagnose_data():
    """Inspect actual data ranges."""
    print("\n" + "="*70)
    print("DATA RANGE DIAGNOSIS")
    print("="*70)
    
    gt_dir = "train/GT"
    noisy_dir = "train/NoisyLR"
    
    train_files, val_files, test_files = load_paired_images(gt_dir, noisy_dir)
    
    # Create dataset
    dataset = PairedImageDataset(
        filenames=train_files[:5],  # First 5 images
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        training=False,
        scale=2
    )
    
    print("\nSample 1 (no crop/aug, training=False):")
    sample = dataset[0]
    noisy = sample["noisy"]
    gt = sample["gt"]
    
    print(f"  Noisy tensor dtype: {noisy.dtype}")
    print(f"  Noisy shape: {noisy.shape}")
    print(f"  Noisy min: {noisy.min().item():.6f}")
    print(f"  Noisy max: {noisy.max().item():.6f}")
    print(f"  Noisy mean: {noisy.mean().item():.6f}")
    print(f"  Noisy std: {noisy.std().item():.6f}")
    
    print(f"\n  GT tensor dtype: {gt.dtype}")
    print(f"  GT shape: {gt.shape}")
    print(f"  GT min: {gt.min().item():.6f}")
    print(f"  GT max: {gt.max().item():.6f}")
    print(f"  GT mean: {gt.mean().item():.6f}")
    print(f"  GT std: {gt.std().item():.6f}")
    
    # Check numpy arrays directly
    print("\nDirect .npy file inspection:")
    noisy_np = np.load(f"train/NoisyLR/{train_files[0]}.npy")
    gt_np = np.load(f"train/GT/{train_files[0]}.npy")
    
    print(f"  Noisy .npy shape: {noisy_np.shape}")
    print(f"  Noisy .npy dtype: {noisy_np.dtype}")
    print(f"  Noisy .npy min: {noisy_np.min():.6f}")
    print(f"  Noisy .npy max: {noisy_np.max():.6f}")
    print(f"  Noisy .npy mean: {noisy_np.mean():.6f}")
    
    print(f"\n  GT .npy shape: {gt_np.shape}")
    print(f"  GT .npy dtype: {gt_np.dtype}")
    print(f"  GT .npy min: {gt_np.min():.6f}")
    print(f"  GT .npy max: {gt_np.max():.6f}")
    print(f"  GT .npy mean: {gt_np.mean():.6f}")


def diagnose_model():
    """Inspect model output without training."""
    print("\n" + "="*70)
    print("MODEL OUTPUT DIAGNOSIS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedRestorerV2(num_features=64, num_blocks=16)
    model = model.to(device).eval()
    
    print(f"\nDevice: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data in different ranges
    print("\nTest 1: Input in [0, 1]")
    x_01 = torch.randn(1, 1, 128, 128).to(device)
    x_01 = torch.clamp(x_01, 0, 1)
    
    with torch.no_grad():
        y_01 = model(x_01)
    
    print(f"  Input min: {x_01.min().item():.6f}")
    print(f"  Input max: {x_01.max().item():.6f}")
    print(f"  Output min: {y_01.min().item():.6f}")
    print(f"  Output max: {y_01.max().item():.6f}")
    print(f"  Output mean: {y_01.mean().item():.6f}")
    print(f"  Output std: {y_01.std().item():.6f}")
    print(f"  Output finite: {torch.isfinite(y_01).all().item()}")
    
    print("\nTest 2: Input in [-1, 1]")
    x_neg1 = torch.randn(1, 1, 128, 128).to(device)
    
    with torch.no_grad():
        y_neg1 = model(x_neg1)
    
    print(f"  Input min: {x_neg1.min().item():.6f}")
    print(f"  Input max: {x_neg1.max().item():.6f}")
    print(f"  Output min: {y_neg1.min().item():.6f}")
    print(f"  Output max: {y_neg1.max().item():.6f}")
    print(f"  Output mean: {y_neg1.mean().item():.6f}")
    print(f"  Output std: {y_neg1.std().item():.6f}")
    print(f"  Output finite: {torch.isfinite(y_neg1).all().item()}")


if __name__ == "__main__":
    diagnose_data()
    diagnose_model()
    print("\n" + "="*70 + "\n")
