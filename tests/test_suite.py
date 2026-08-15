#!/usr/bin/env python
"""
Comprehensive test suite for the restoration project.

Tests:
- Dataset loading and processing
- Model architecture and forward pass
- Loss functions
- Training pipeline
- Evaluation metrics

Usage:
    python tests/test_suite.py --verbose
    python tests/test_suite.py --quick
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import (
    load_paired_images,
    PairedImageDataset,
    DegradationPipeline
)
from models import (
    UnifiedRestorerV1,
    UnifiedRestorerV2,
)
from models.losses import (
    CharbonnierLoss,
    GradientLoss,
    FrequencyLoss,
    PerceptualLoss,
    RestorationLoss
)
from evaluation import psnr, ssim


class TestResult:
    """Simple test result tracker."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record_pass(self):
        self.passed += 1
    
    def record_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)
    
    def print_summary(self):
        total = self.passed + self.failed
        status = "✓ PASS" if self.failed == 0 else "✗ FAIL"
        print(f"\n[{status}] {self.name}")
        print(f"  Passed: {self.passed}/{total}")
        if self.errors:
            print(f"  Errors:")
            for error in self.errors[:3]:  # Print first 3 errors
                print(f"    - {error}")
            if len(self.errors) > 3:
                print(f"    ... and {len(self.errors) - 3} more")


class TestSuite:
    """Comprehensive test suite."""
    
    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results = []
    
    def test_dataset_loading(self):
        """Test dataset loading and processing."""
        result = TestResult("Dataset Loading")
        
        try:
            # Test paired image loading
            gt_dir = "train/GT"
            noisy_dir = "train/NoisyLR"
            
            if not os.path.exists(gt_dir) or not os.path.exists(noisy_dir):
                result.record_fail(f"Dataset directories not found")
                self.results.append(result)
                return
            
            train_files, val_files, test_files = load_paired_images(gt_dir, noisy_dir)
            
            assert len(train_files) > 0, "No training images found"
            assert len(val_files) > 0, "No validation images found"
            assert len(test_files) > 0, "No test images found"
            
            result.record_pass()
            
            # Test loading individual images
            gt_img = np.load(os.path.join(gt_dir, train_files[0] + ".npy"))
            noisy_img = np.load(os.path.join(noisy_dir, train_files[0] + ".npy"))
            
            assert gt_img.ndim == 2, f"GT should be 2D, got shape {gt_img.shape}"
            assert noisy_img.ndim == 2, f"Noisy should be 2D, got shape {noisy_img.shape}"
            assert gt_img.dtype == np.float32, f"GT dtype should be float32, got {gt_img.dtype}"
            assert noisy_img.dtype == np.float32, f"Noisy dtype should be float32, got {noisy_img.dtype}"
            
            result.record_pass()
            
            if self.verbose:
                print(f"  Loaded {len(train_files)} training images")
                print(f"  Noisy shape: {noisy_img.shape}, GT shape: {gt_img.shape}")
                print(f"  GT range: [{gt_img.min():.4f}, {gt_img.max():.4f}]")
                print(f"  Noisy range: [{noisy_img.min():.4f}, {noisy_img.max():.4f}]")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_dataset_class(self):
        """Test PairedImageDataset class."""
        result = TestResult("Dataset Class")
        
        try:
            gt_dir = "train/GT"
            noisy_dir = "train/NoisyLR"
            
            if not os.path.exists(gt_dir):
                result.record_fail("Dataset directories not found")
                self.results.append(result)
                return
            
            # Load filenames
            train_files, val_files, test_files = load_paired_images(gt_dir, noisy_dir)
            
            # Create dataset
            dataset = PairedImageDataset(
                filenames=train_files[:10],  # Use first 10 for testing
                gt_dir=gt_dir,
                noisy_dir=noisy_dir,
                training=True,
                scale=2,
                lr_patch_size=64
            )
            
            assert len(dataset) > 0, "Dataset is empty"
            result.record_pass()
            
            # Test sample loading
            sample = dataset[0]
            
            assert "gt" in sample, "Sample missing 'gt' key"
            assert "noisy" in sample, "Sample missing 'noisy' key"
            result.record_pass()
            
            # Test tensor shapes
            assert sample["gt"].ndim == 3, "GT should be 3D (C, H, W)"
            assert sample["noisy"].ndim == 3, "Noisy should be 3D (C, H, W)"
            assert sample["gt"].shape[0] == 1, "GT should have 1 channel"
            assert sample["noisy"].shape[0] == 1, "Noisy should have 1 channel"
            result.record_pass()
            
            if self.verbose:
                print(f"  Dataset size: {len(dataset)}")
                print(f"  GT shape: {sample['gt'].shape}")
                print(f"  Noisy shape: {sample['noisy'].shape}")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_degradation_pipeline(self):
        """Test degradation pipeline."""
        result = TestResult("Degradation Pipeline")
        
        try:
            # Create synthetic degradation
            pipeline = DegradationPipeline(
                gaussian_sigma_range=(0.01, 0.05),
                speckle_strength_range=(0.05, 0.20),
                scale=2
            )
            result.record_pass()
            
            # Test on random image
            gt_img = np.random.randn(256, 256).astype(np.float32)
            
            degraded = pipeline.add_gaussian_noise(gt_img, sigma=0.03)
            
            assert degraded.shape == gt_img.shape, "Shape mismatch after degradation"
            assert degraded.dtype == np.float32, "Output dtype should be float32"
            result.record_pass()
            
            if self.verbose:
                print(f"  Original range: [{gt_img.min():.4f}, {gt_img.max():.4f}]")
                print(f"  Degraded range: [{degraded.min():.4f}, {degraded.max():.4f}]")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_model_v1(self):
        """Test UnifiedRestorerV1 model."""
        result = TestResult("Model V1")
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UnifiedRestorerV1(num_features=64, num_blocks=12)
            model = model.to(device).eval()
            result.record_pass()
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            expected_params = 1_000_000  # Approximately 1M
            assert params > 0.5e6 and params < 2e6, f"Parameter count {params} out of expected range"
            result.record_pass()
            
            # Test forward pass
            # Note: Models do 2x upsampling, so input 128x128 -> output 256x256
            with torch.no_grad():
                x = torch.randn(1, 1, 128, 128).to(device)
                y = model(x)
            
            assert y.shape == (1, 1, 256, 256), f"Output shape {y.shape} != expected (1, 1, 256, 256)"
            assert y.dtype == torch.float32, f"Output dtype should be float32, got {y.dtype}"
            result.record_pass()
            
            if self.verbose:
                print(f"  Parameters: {params:,}")
                print(f"  Input shape: {x.shape}")
                print(f"  Output shape: {y.shape}")
                print(f"  Output range: [{y.min():.4f}, {y.max():.4f}]")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_model_v2(self):
        """Test UnifiedRestorerV2 model."""
        result = TestResult("Model V2")
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UnifiedRestorerV2(
                num_features=64,
                num_blocks=16,
                use_hf_branch=True,
                use_edge_branch=True
            )
            model = model.to(device).eval()
            result.record_pass()
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            expected_params = 2_000_000  # Approximately 2-5M
            assert params > 1e6, f"Parameter count {params} seems too low"
            result.record_pass()
            
            # Test forward pass (2x upsampling: 128x128 -> 256x256)
            with torch.no_grad():
                x = torch.randn(1, 1, 128, 128).to(device)
                y = model(x)
            
            assert y.shape == (1, 1, 256, 256), f"Output shape {y.shape} != expected (1, 1, 256, 256)"
            result.record_pass()
            
            if self.verbose:
                print(f"  Parameters: {params:,}")
                print(f"  Input shape: {x.shape}")
                print(f"  Output shape: {y.shape}")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_loss_functions(self):
        """Test loss functions."""
        result = TestResult("Loss Functions")
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            pred = torch.randn(4, 1, 128, 128).to(device)
            target = torch.randn(4, 1, 128, 128).to(device)
            
            # Test Charbonnier loss
            loss_fn = CharbonnierLoss()
            loss = loss_fn(pred, target)
            assert loss.item() > 0, "Charbonnier loss should be positive"
            result.record_pass()
            
            # Test Gradient loss
            loss_fn = GradientLoss()
            loss = loss_fn(pred, target)
            assert loss.item() >= 0, "Gradient loss should be non-negative"
            result.record_pass()
            
            # Test Frequency loss
            loss_fn = FrequencyLoss()
            loss = loss_fn(pred, target)
            assert loss.item() >= 0, "Frequency loss should be non-negative"
            result.record_pass()
            
            # Test combined loss
            loss_fn = RestorationLoss()
            loss, components = loss_fn(pred, target, return_components=True)
            assert loss.item() > 0, "Combined loss should be positive"
            assert "total" in components, "Components should have 'total' key"
            result.record_pass()
            
            if self.verbose:
                print(f"  Charbonnier loss: {loss_fn(pred, target).item():.6f}")
                print(f"  Combined loss components: {list(components.keys())}")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_metrics(self):
        """Test evaluation metrics."""
        result = TestResult("Metrics")
        
        try:
            # Create test images
            img1 = np.random.rand(256, 256).astype(np.float32)
            img2 = np.random.rand(256, 256).astype(np.float32)
            
            # Test PSNR
            psnr_val = psnr(img1, img2, max_val=1.0)
            assert isinstance(psnr_val, (int, float)), "PSNR should return a number"
            assert psnr_val > 0, "PSNR should be positive"
            result.record_pass()
            
            # Test SSIM
            ssim_val = ssim(img1, img2, max_val=1.0)
            assert isinstance(ssim_val, (int, float)), "SSIM should return a number"
            assert -1 <= ssim_val <= 1, "SSIM should be in [-1, 1]"
            result.record_pass()
            
            # Test perfect reconstruction
            psnr_perfect = psnr(img1, img1, max_val=1.0)
            assert psnr_perfect > 50, "PSNR for identical images should be very high"
            result.record_pass()
            
            if self.verbose:
                print(f"  PSNR: {psnr_val:.2f}")
                print(f"  SSIM: {ssim_val:.4f}")
                print(f"  PSNR (perfect): {psnr_perfect:.2f}")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def test_training_step(self):
        """Test a training step."""
        result = TestResult("Training Step")
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create model
            model = UnifiedRestorerV1(num_features=64, num_blocks=12)
            model = model.to(device)
            
            # Create loss and optimizer
            loss_fn = RestorationLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Create dummy batch
            # Model expects 128x128 input (which it upsamples to 256x256)
            noisy = torch.randn(2, 1, 128, 128).to(device)
            gt = torch.randn(2, 1, 256, 256).to(device)
            gt = torch.clamp(gt, 0, 1)
            
            # Training step
            model.train()
            optimizer.zero_grad()
            pred = model(noisy)
            assert pred.shape == gt.shape, f"Pred shape {pred.shape} != GT shape {gt.shape}"
            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0, "Loss should be positive"
            result.record_pass()
            
            if self.verbose:
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Training step completed successfully")
        
        except Exception as e:
            result.record_fail(str(e))
        
        self.results.append(result)
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_dataset_loading,
            self.test_dataset_class,
            self.test_degradation_pipeline,
            self.test_model_v1,
            self.test_model_v2,
            self.test_loss_functions,
            self.test_metrics,
            self.test_training_step,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"[ERROR] Test {test.__name__} failed with exception: {e}")
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total_passed = 0
        total_failed = 0
        
        for result in self.results:
            result.print_summary()
            total_passed += result.passed
            total_failed += result.failed
        
        print("\n" + "-"*70)
        total = total_passed + total_failed
        print(f"Total: {total_passed}/{total} passed")
        
        if total_failed == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print(f"✗ {total_failed} TEST(S) FAILED")
        
        print("="*70 + "\n")
        
        return total_failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick tests only")
    
    args = parser.parse_args()
    
    suite = TestSuite(verbose=args.verbose, quick=args.quick)
    success = suite.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
