# KLA Image Restoration — Complete Project

A comprehensive PyTorch-based image restoration system for the KLA semiconductor image restoration challenge, implementing progressive models from DnCNN baseline to advanced Unified Restorer V2 with detail preservation.

## Project Overview

This repository contains implementations for paired image restoration focusing on:

- **Noise Removal** (Gaussian and Speckle)
- **Detail Preservation** (edges, textures, fine structures)
- **2× Super-Resolution** (128×128 → 256×256)
- **Robustness** to extreme input values (out-of-range handling)

## Dataset

### Structure

- **Training:** 3,200 paired images (80/10/10 split for train/val/test)
- **Input (NoisyLR):** 128×128 grayscale images, float32, range: [-0.0026, 1.3258]
- **Target (GT):** 256×256 grayscale images, float32, range: [0.0, 1.0]

### Degradations

Each image contains a combination of:

- Gaussian noise (signal-dependent)
- Speckle noise (multiplicative)
- Spatial resolution reduction (2× downsampling)

### Critical Note on Data Range

⚠️ **NoisyLR values outside [0, 1] are LEGITIMATE and should NOT be clipped before feature extraction.** Clipping only occurs for:

- Metric computation (PSNR, SSIM)
- Image visualization
- Output saving

## Models

### DnCNN Baseline

- **Status:** Existing, preserved
- **Parameters:** ~850k
- **PSNR:** 27.49 dB
- **Architecture:** 10 residual blocks + PixelShuffle ×2 upsampling

### NoiseAwareDnCNN

- **Status:** Existing, preserved
- **Parameters:** ~927k
- **PSNR:** 27.97 dB
- **Key Feature:** Signal-dependent noise sigma conditioning

### Unified Restorer V1 ⭐ NEW

- **Status:** Implemented, ready for training
- **Parameters:** ~1.3M
- **Target PSNR:** 28.00 dB
- **Features:**
  - 12 Residual Channel Attention Blocks
  - Degradation-aware FiLM conditioning
  - Robust input encoder (no BatchNorm)
  - Global residual connection

### Unified Restorer V2 ⭐⭐ NEW (Advanced)

- **Status:** Implemented, ready for training
- **Parameters:** ~2-5M
- **Target PSNR:** >28.1 dB
- **Major Features:**
  - High-frequency detail branch for texture recovery
  - Edge-aware branch for structure preservation
  - Multi-scale feature extraction (3×3, 5×5, 7×7 receptive fields)
  - 16 Residual Channel Attention Blocks
  - Enhanced degradation conditioning

## Loss Functions

Configurable combined loss with components:

```
Total Loss =
  0.65 × Charbonnier(L1)
  + 0.15 × MSE
  + 0.10 × Gradient(Sobel)
  + 0.05 × Frequency(FFT)
  + 0.05 × Perceptual(LPIPS, optional)
```

- **Charbonnier:** Smooth L1 loss, robust to outliers
- **Gradient:** Sobel-based edge preservation
- **Frequency:** FFT-domain loss for spectral consistency
- **Perceptual:** LPIPS for human-perceived quality (optional)

## Project Structure

```
.
├── analysis/                      # Analysis and audit
│   ├── project_audit.md          # Comprehensive repository audit
│   └── IMPLEMENTATION_REPORT.md  # Detailed implementation status
│
├── configs/                       # YAML configuration files
│   ├── default.yaml              # Master configuration template
│   ├── train_baseline.yaml        # DnCNN configuration
│   ├── train_noise_aware.yaml     # NoiseAwareDnCNN configuration
│   ├── train_v1.yaml              # Unified V1 configuration
│   ├── train_v2.yaml              # Unified V2 configuration (with curriculum)
│   └── evaluation.yaml            # Evaluation configuration
│
├── datasets/                      # Dataset infrastructure
│   ├── __init__.py
│   ├── dataset_utils.py           # Utilities: loading, splitting, validation
│   ├── paired_dataset.py          # PairedImageDataset with augmentation
│   └── synthetic_degradation.py   # Degradation pipeline for augmentation
│
├── models/                        # Neural network models
│   ├── __init__.py
│   ├── noise_aware_dncnn.py       # Existing NoiseAwareDnCNN
│   ├── restoration_blocks.py      # Building blocks (RCAB, Attention, etc.)
│   ├── unified_restorer_v1.py     # Unified V1 implementation
│   ├── unified_restorer_v2.py     # Unified V2 implementation
│   └── losses.py                  # Loss functions
│
├── train/                         # Training directory
│   ├── GT/                        # Ground truth images (3200 × 256×256)
│   └── NoisyLR/                   # Noisy LR images (3200 × 128×128)
│
├── checkpoints/                   # Model checkpoints
│   ├── best_dncnn.pth             # Existing DnCNN checkpoint
│   ├── best_noise_aware_dncnn.pth # Existing NoiseAwareDnCNN checkpoint
│   ├── v1/                        # V1 checkpoints (to be created)
│   └── v2/                        # V2 checkpoints (to be created)
│
├── results/                       # Results and evaluation
│   ├── dncnn_test_results.csv     # Existing DnCNN results
│   ├── baseline/                  # Baseline results
│   ├── v1/                        # V1 evaluation results (to be created)
│   ├── v2/                        # V2 evaluation results (to be created)
│   ├── robustness/                # Robustness test results
│   └── ablation/                  # Ablation study results
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Installation

### 1. Clone/Setup

```bash
cd e:\COLLEGE\AI\PROJECT\KLA_Image_Restoration-main
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: Install LPIPS (for perceptual loss)

```bash
pip install lpips
```

## Usage

### Quick Start: Dataset Inspection

```python
from datasets import load_paired_images, compute_dataset_statistics

# Load and split dataset
train_files, val_files, test_files = load_paired_images(
    "train/GT", "train/NoisyLR", seed=42
)

# Compute statistics
stats = compute_dataset_statistics("train/GT", "train/NoisyLR")
print(f"Dataset: {stats['total_samples']} samples")
print(f"Noisy range: [{stats['noisy_stats']['min']:.4f}, {stats['noisy_stats']['max']:.4f}]")
```

### Quick Start: Model Testing

```python
import torch
from models import UnifiedRestorerV1, UnifiedRestorerV2

# Create V1 model
model_v1 = UnifiedRestorerV1(
    in_channels=1,
    out_channels=1,
    num_features=64,
    num_blocks=12,
    scale=2
)

# Create V2 model
model_v2 = UnifiedRestorerV2(
    in_channels=1,
    out_channels=1,
    num_features=64,
    num_blocks=16,
    scale=2,
    use_hf_branch=True,
    use_edge_branch=True
)

# Test forward pass
x = torch.randn(1, 1, 64, 64)  # (B, C, H, W)
y_v1 = model_v1(x)  # Output: (1, 1, 128, 128)
y_v2 = model_v2(x)  # Output: (1, 1, 128, 128)

print(f"V1 Parameters: {model_v1.count_parameters():,}")
print(f"V2 Parameters: {model_v2.count_parameters():,}")
```

### Quick Start: Loss Functions

```python
import torch
from models.losses import RestorationLoss

# Create loss function
loss_fn = RestorationLoss(
    use_charbonnier=True,
    l1_weight=0.65,
    mse_weight=0.15,
    gradient_weight=0.10,
    frequency_weight=0.05
)

# Compute loss
pred = torch.randn(8, 1, 128, 128)  # Batch of predictions
target = torch.randn(8, 1, 128, 128)  # Batch of targets
loss, components = loss_fn(pred, target, return_components=True)

print(f"Total Loss: {loss.item():.4f}")
for name, value in components.items():
    print(f"  {name}: {value:.4f}")
```

### Quick Start: Dataset Loading

```python
from datasets import create_dataloaders

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    gt_dir="train/GT",
    noisy_dir="train/NoisyLR",
    batch_size=16,
    num_workers=4,
    scale=2,
    lr_patch_size=64,
    seed=42
)

# Iterate through batch
for batch in train_loader:
    noisy = batch['noisy']    # (B, 1, H, W)
    gt = batch['gt']          # (B, 1, H*2, W*2)
    filename = batch['filename']

    print(f"Noisy shape: {noisy.shape}, GT shape: {gt.shape}")
    break
```

## Configuration

All hyperparameters are in YAML files under `configs/`. Example:

```yaml
# configs/train_v2.yaml
model:
  channels: 64
  blocks: 16
  use_attention: true
  use_high_frequency_branch: true
  use_edge_branch: true

training:
  epochs: 60
  batch_size: 16
  learning_rate: 1.0e-3
  optimizer: "adamw"
  scheduler: "cosine"
  use_amp: true

loss:
  l1_weight: 0.65
  mse_weight: 0.15
  gradient_weight: 0.10
  frequency_weight: 0.05

curriculum:
  enabled: true
  stage1:
    epochs: 20
    data: "real"
  stage2:
    epochs: 25
    data: "mixed"
    synthetic_ratio: 0.4
  stage3:
    epochs: 15
    data: "real"
```

## Training (Next Phase)

Training scripts will be implemented to:

1. Load configuration from YAML
2. Create model and loss function
3. Setup optimizer and scheduler
4. Implement curriculum learning (3 stages for V2)
5. Run validation at each epoch
6. Save checkpoints and results
7. Log metrics (PSNR, SSIM, LPIPS, latency)

Expected training commands:

```bash
python train/train_v1.py --config configs/train_v1.yaml
python train/train_v2.py --config configs/train_v2.yaml --stage 1  # Stage 1 only
python train/train_v2.py --config configs/train_v2.yaml --stage all  # All stages
```

## Evaluation (Next Phase)

Evaluation scripts will:

1. Load trained models
2. Evaluate on test set
3. Compute PSNR, SSIM, LPIPS
4. Measure inference latency
5. Generate visualizations
6. Perform robustness testing

Expected evaluation commands:

```bash
python evaluation/evaluate.py --checkpoint checkpoints/v2/best_v2.pth
python evaluation/robustness.py --checkpoint checkpoints/v2/best_v2.pth
python visualization/comparison.py --checkpoint checkpoints/v2/best_v2.pth
```

## Reference Results

### Existing Baselines (Preserved)

| Model           | Params | PSNR     | SSIM   | LPIPS  | Latency  |
| --------------- | ------ | -------- | ------ | ------ | -------- |
| DnCNN           | 850k   | 27.49 dB | 0.7363 | —      | 22.30 ms |
| NoiseAwareDnCNN | 927k   | 27.97 dB | 0.7509 | 0.3003 | 16.90 ms |

### Target Performance

| Model | Params | PSNR     | SSIM   | LPIPS  | Notes                        |
| ----- | ------ | -------- | ------ | ------ | ---------------------------- |
| V1    | 1.3M   | 28.00 dB | 0.7573 | 0.2825 | To be trained                |
| V2    | 2-5M   | >28.1 dB | >0.76  | <0.28  | Advanced detail preservation |

## Architecture Details

### Unified V1

```
NoisyLR (64×64)
    ↓
Input Encoder (ReLU, no BN)
    ↓
Degradation Statistics → Embedding (32-dim)
    ↓
12 × Residual Channel Attention Block
    ↓
Feature Refinement
    ↓
PixelShuffle 2×
    ↓
Output Reconstruction
    ↓
Output (128×128)
```

### Unified V2

```
NoisyLR (64×64)
    ↓
Input Encoder (ReLU, no BN)
    ↓
Degradation Statistics → Embedding
    ↓
16 × Residual Channel Attention Block
    ↓
┌─────────────────────────────────┐
│ Multi-Scale Feature Extraction  │
│ (3×3, 5×5 dilated, 7×7 dilated) │
└──────────┬──────────────────────┘
           ↓
    ┌──────┴──────┬──────────┐
    ↓             ↓          ↓
Low Freq  High Freq       Edge
Features  Branch          Branch
    │      (Blur-Subtract)  (Sobel)
    │             │         │
    └──────┬──────┴─────────┘
           ↓
        Fusion
           ↓
    Detail Refinement
           ↓
    PixelShuffle 2×
           ↓
    Output Reconstruction
           ↓
    Output (128×128)
```

## Key Features

✅ **Range Preservation** — Out-of-range values preserved during inference  
✅ **Degradation Awareness** — Models condition on input statistics  
✅ **Multi-Scale Processing** — Multiple receptive field sizes  
✅ **Detail Preservation** — High-frequency and edge branches  
✅ **Advanced Loss** — Gradient, frequency, and perceptual components  
✅ **Curriculum Learning** — Progressive training stages (V2)  
✅ **Configurable** — Everything in YAML files  
✅ **Reproducible** — Deterministic splitting, seed management

## Reproducibility

- ✅ Fixed seeds (Python, NumPy, PyTorch, CUDA)
- ✅ Deterministic dataset splitting
- ✅ Configuration saving with checkpoints
- ✅ All metrics computation documented

## Known Limitations

1. LPIPS optional (requires separate installation)
2. GPU assumed (CPU fallback available)
3. Training scripts not yet implemented
4. Evaluation scripts not yet implemented

## Next Steps

1. ✅ Infrastructure & Models (COMPLETE)
2. ⏳ Training scripts (Coming)
3. ⏳ Evaluation & Visualization (Coming)
4. ⏳ Tests & Smoke training (Coming)
5. ⏳ Full training (Manual trigger)

## Files Added in This Phase

### Configuration (5 files)

- `configs/default.yaml`
- `configs/train_baseline.yaml`
- `configs/train_noise_aware.yaml`
- `configs/train_v1.yaml`
- `configs/train_v2.yaml`
- `configs/evaluation.yaml`

### Dataset Infrastructure (4 files)

- `datasets/dataset_utils.py`
- `datasets/paired_dataset.py`
- `datasets/synthetic_degradation.py`
- `datasets/__init__.py`

### Models (6 files)

- `models/restoration_blocks.py`
- `models/unified_restorer_v1.py`
- `models/unified_restorer_v2.py`
- `models/losses.py`
- `models/__init__.py`

### Documentation (2 files)

- `analysis/project_audit.md`
- `analysis/IMPLEMENTATION_REPORT.md`

### Updated (1 file)

- `requirements.txt`

## Contact & Support

For questions about:

- **Dataset:** Check `analysis/project_audit.md`
- **Models:** See model docstrings and test functions
- **Configuration:** Refer to YAML files
- **Implementation:** See `analysis/IMPLEMENTATION_REPORT.md`

## License

[Your license here]

---

**Last Updated:** 2026-08-14  
**Status:** Infrastructure Phase Complete ✅ — Ready for Training Implementation
