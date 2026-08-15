#!/usr/bin/env python
"""Diagnostic for GPU training issues."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataloaders
from models import UnifiedRestorerV2
from models.losses import RestorationLoss


def diagnose_training():
    """Diagnose GPU training behavior."""
    print("\n" + "="*70)
    print("GPU TRAINING DIAGNOSTICS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load config
    with open("configs/train_v2.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Build model
    print("\n[1] Building model...")
    model = UnifiedRestorerV2(num_features=64, num_blocks=16)
    model = model.to(device).train()
    params_before = [p.clone().detach() for p in model.parameters()]
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss and optimizer
    print("\n[2] Building loss and optimizer...")
    loss_fn = RestorationLoss(
        use_charbonnier=True,
        l1_weight=0.65,
        mse_weight=0.15,
        gradient_weight=0.10,
        frequency_weight=0.001,
        perceptual_weight=0.0
    )
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = GradScaler()
    print("  Loss: RestorationLoss")
    print("  Optimizer: AdamW (lr=1e-3)")
    print("  GradScaler: enabled")
    
    # Build dataloaders
    print("\n[3] Building dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            "train/GT",
            "train/NoisyLR",
            batch_size=4,  # Smaller batch for debugging
            num_workers=0
        )
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # Get one training batch
    print("\n[4] Getting first training batch...")
    batch = next(iter(train_loader))
    noisy = batch["noisy"].to(device)
    gt = batch["gt"].to(device)
    
    print(f"  Noisy shape: {noisy.shape}, range: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"  GT shape: {gt.shape}, range: [{gt.min():.4f}, {gt.max():.4f}]")
    
    # Test forward pass
    print("\n[5] Testing forward pass...")
    with torch.no_grad():
        pred_no_grad = model(noisy)
    print(f"  Output shape: {pred_no_grad.shape}")
    print(f"  Output range: [{pred_no_grad.min():.6f}, {pred_no_grad.max():.6f}]")
    print(f"  Output mean: {pred_no_grad.mean():.6f}")
    print(f"  Output std: {pred_no_grad.std():.6f}")
    print(f"  Output finite: {torch.isfinite(pred_no_grad).all().item()}")
    
    # Test with autocast (AMP)
    print("\n[6] Testing forward with AMP (autocast)...")
    with autocast():
        pred_amp = model(noisy)
    print(f"  Output dtype (inside autocast): {pred_amp.dtype}")
    print(f"  Output range: [{pred_amp.min():.6f}, {pred_amp.max():.6f}]")
    print(f"  Output finite: {torch.isfinite(pred_amp).all().item()}")
    
    # Test loss computation
    print("\n[7] Testing loss computation...")
    with autocast():
        pred = model(noisy)
        loss, loss_components = loss_fn(pred, gt, return_components=True)
    
    print(f"  Loss dtype: {loss.dtype}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss finite: {torch.isfinite(loss).item()}")
    print(f"  Loss components:")
    for key, val in loss_components.items():
        print(f"    {key}: {val:.6f}")
    
    # Test backward pass
    print("\n[8] Testing backward pass...")
    optimizer.zero_grad(set_to_none=True)
    
    with autocast():
        pred = model(noisy)
        loss = loss_fn(pred, gt)
    
    scaler.scale(loss).backward()
    
    # Check gradients
    print(f"  Gradients computed: yes")
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if len(grad_norms) <= 3:  # Print first few
                print(f"    {name}: grad_norm={grad_norm:.6e}, finite={torch.isfinite(param.grad).all().item()}")
    
    if grad_norms:
        print(f"  Gradient norm stats: min={min(grad_norms):.6e}, max={max(grad_norms):.6e}, mean={np.mean(grad_norms):.6e}")
    
    # Check if scaler would skip
    print("\n[9] Testing gradient clipping...")
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  Clipped norm: {total_norm:.6f}")
    print(f"  Max grad norm: 1.0")
    print(f"  Clipping applied: {total_norm > 1.0}")
    
    # Test optimizer step
    print("\n[10] Testing optimizer.step()...")
    step_result = scaler.step(optimizer)
    scaler.update()
    print(f"  Optimizer step completed")
    
    # Check if parameters changed
    print("\n[11] Checking parameter changes...")
    param_changes = []
    for p_before, p_after in zip(params_before, model.parameters()):
        change = (p_after - p_before.to(device)).abs().max().item()
        param_changes.append(change)
    
    print(f"  Parameter change stats:")
    print(f"    Min: {min(param_changes):.6e}")
    print(f"    Max: {max(param_changes):.6e}")
    print(f"    Mean: {np.mean(param_changes):.6e}")
    print(f"  Parameters updated: {max(param_changes) > 0}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    diagnose_training()
