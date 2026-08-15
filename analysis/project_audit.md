# KLA Image Restoration — Project Audit

**Date:** 2026-08-14  
**Objective:** Comprehensive inspection and planning for Version 2 implementation

---

## 1. EXISTING ARCHITECTURE

### 1.1 Models Implemented

#### DnCNN

- **Status:** Implemented, trained, checkpoint available
- **Architecture:**
  - Input: 1-channel image (64×64 LR)
  - Conv + ReLU
  - 10 Residual Blocks
  - Reconstruction Conv
  - PixelShuffle 2× Upsampling
  - Output: 1-channel (128×128)
- **Parameters:** ~850k (estimated)
- **Training Loss:** 0.8×L1 + 0.2×MSE
- **Results:** PSNR 27.49 dB, SSIM 0.7363, inference ~22.30 ms

#### NoiseAwareDnCNN

- **Status:** Implemented, trained, checkpoint available
- **Architecture:**
  - Input: 2-channel (image + estimated noise sigma map)
  - Head: Conv2d(2 → 64) + ReLU
  - 10 Residual Blocks
  - Reconstruction Conv
  - PixelShuffle 2× Upsampling
  - Output: Conv2d(64 → 1)
- **Parameters:** ~927k (from master prompt reference)
- **Conditioning:** Signal-dependent noise model
  - Variance: VAR_A×x² + VAR_B×x + VAR_C
  - VAR_A = 0.01846589, VAR_B = 0.00843552, VAR_C = -0.00033869
- **Training Loss:** 0.8×L1 + 0.2×MSE
- **Results:** PSNR 27.97 dB, SSIM 0.7509, LPIPS 0.3003, inference ~16.90 ms

#### Unified V1

- **Status:** NOT FOUND in codebase
- **Master Prompt Reference:** 1.286M parameters, 28.00 dB PSNR, 0.7573 SSIM, 0.2825 LPIPS
- **Note:** Reference results exist but model implementation is missing
- **Action:** Must be created before V2

#### Unified V2

- **Status:** NOT IMPLEMENTED
- **Target:** 2M–5M parameters, improved detail preservation, better edge recovery
- **Action:** To be implemented

### 1.2 Training Scripts

#### train_dncnn.py

- Uses KLADataset class with 64×64 LR crop, 128×128 GT crop
- Train/Val/Test split: 80/10/10
- Batch size: 16
- 50 epochs
- Learning rate: 1e-3
- Random flips for augmentation
- Saves best_dncnn.pth

#### train_noise_aware.py

- Similar structure to DnCNN training
- Computes noise sigma maps from noisy images
- Uses empirical noise variance model
- Saves best_noise_aware_dncnn.pth

### 1.3 Evaluation Scripts

- **evaluate_dncnn.py:** Full-image evaluation, PSNR/SSIM/MSE metrics
- **evaluate_noise_aware.py:** Similar to DnCNN evaluation
- **evaluate_combined.py:** Tests robustness to synthetic Gaussian and Speckle noise
- **evaluate_gaussian.py:** Isolated Gaussian noise robustness
- **evaluate_speckle.py:** Isolated Speckle noise robustness
- **evaluate_lpips.py:** LPIPS metric computation

### 1.4 Analysis Scripts

- **dataset_audit.py:** Comprehensive dataset statistics
- **noise_model_analysis.py:** Noise modeling and analysis
- **analyze_errors.py:** Error analysis with DnCNN architecture
- **traditional_baseline.py:** Baseline methods (bicubic, etc.)
- **check_model_shape.py, check_shapes.py:** Debug utilities

---

## 2. DATA PIPELINE

### 2.1 Dataset Structure

```
train/
  ├── GT/          (3200 files, 256×256 each)
  └── NoisyLR/     (3200 files, 128×128 each)
```

### 2.2 Dataset Statistics

**Ground Truth (GT):**

- Resolution: 256×256
- Format: .npy files, float32
- Range: [0.0, 1.0]
- Mean: 0.2182
- Values within [0, 1]: 100%

**Noisy LR:**

- Resolution: 128×128
- Format: .npy files, float32
- Range: [-0.0026, 1.3258]
- Mean: 0.2184
- Values < 0: ~0.13%
- Values > 1: ~2.3%
- **Important:** Out-of-range values are LEGITIMATE. Do NOT clip before feature extraction.

### 2.3 Degradation Types

Based on analysis scripts, dataset contains combinations of:

1. **Gaussian Noise** - Signal-dependent, variance model fitted
2. **Speckle Noise** - Multiplicative noise
3. **Downsampling** - 2× spatial resolution reduction (128×128 → 256×256 in reconstruction)

### 2.4 Data Loading

**KLADataset class:**

- Loads paired .npy files by filename matching
- Training mode: Random 64×64 LR crop with corresponding 128×128 GT crop
- Crops maintained with SCALE=2 alignment
- Random horizontal/vertical flips applied to both
- No validation/test splits saved; computed dynamically

### 2.5 Data Range Preservation

**CRITICAL:**

- NoisyLR values are NOT clipped before feature extraction
- Out-of-range values are preserved as model input
- Final predictions clipped only for PSNR/SSIM/LPIPS computation and image saving

---

## 3. TRAINING PIPELINE

### 3.1 Current Training Configuration

- **Epochs:** 50
- **Batch Size:** 16
- **Learning Rate:** 1e-3
- **Optimizer:** Adam (assumed, not explicitly shown)
- **Scheduler:** Not visible in inspection
- **Loss:** 0.8×L1 + 0.2×MSE
- **AMP:** Not used in current training
- **Gradient Clipping:** Not implemented

### 3.2 Seed and Reproducibility

- Seed: 42 (set in all training scripts)
- NumPy, PyTorch, CUDA seeds set consistently

### 3.3 Training Features

- **Augmentation:** Random crops, horizontal flip, vertical flip
- **Validation:** Deterministic validation at full image size
- **Checkpointing:** Best model saved by PSNR

---

## 4. EVALUATION PIPELINE

### 4.1 Metrics Implemented

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity)
- **MSE/MAE** (in analysis scripts)
- **LPIPS** (optional, in dedicated script)
- **Inference Latency** (timing per image)

### 4.2 Evaluation Methodology

- Full-image evaluation (no crops)
- Train/Val/Test split applied consistently
- Test set: ~320 samples
- Predictions clipped to [0, 1] for metric computation

### 4.3 Results Tracking

**Format:** CSV files with per-image metrics

- `dncnn_test_results.csv`: Columns: filename, mse, psnr, ssim, inference_time_sec
- Summary TXT files with aggregate statistics

### 4.4 Current Results

**DnCNN:**

- Evaluated: 320 test samples
- Mean PSNR: 27.49 dB
- Mean SSIM: 0.7363
- Mean Inference: 22.30 ms

**NoiseAwareDnCNN:**

- Available results in reference (27.97 dB PSNR)

---

## 5. EXISTING CHECKPOINTS

```
checkpoints/
  ├── best_dncnn.pth
  └── best_noise_aware_dncnn.pth
```

**Status:** Both exist and should be preserved
**Format:** PyTorch .pth files (exact format not inspected yet)

---

## 6. EXISTING RESULTS

```
results/
  ├── dncnn_test_results.csv
  ├── dncnn_test_summary.txt
  ├── loss_curve.png
  ├── psnr_curve.png
  ├── ssim_curve.png
  └── ablation/
```

**Status:** All preserved, not to be overwritten
**Note:** No V1 or V2 results yet

---

## 7. REUSABLE COMPONENTS

### 7.1 Code That Should Be Preserved

1. **KLADataset class** - Dataset loading logic
2. **Residual Block** - Standard residual architecture (BatchNorm version)
3. **Training loop patterns** - Epoch/batch iteration, validation
4. **Evaluation patterns** - PSNR/SSIM computation
5. **Seed/reproducibility setup** - Set_seed function
6. **Output directory management** - Path creation patterns

### 7.2 Concepts to Preserve

1. **Noise sigma conditioning** - NoiseAwareDnCNN approach is sound
2. **Loss formulation** - 0.8×L1 + 0.2×MSE baseline is reasonable
3. **Evaluation methodology** - Full-image test set, consistent split
4. **Data range handling** - Preservation of out-of-range values

---

## 8. IDENTIFIED PROBLEMS TO FIX

### 8.1 Architecture Limitations

1. **Over-smoothing:** DnCNN and NoiseAwareDnCNN lack explicit detail preservation
2. **No multi-scale features:** Single-scale convolutions limit receptive field
3. **No high-frequency branch:** No explicit edge/texture recovery
4. **No attention mechanism:** Fixed feature processing regardless of content

### 8.2 Dataset/Pipeline Gaps

1. **No synthetic degradation module:** Cannot augment with controlled combinations
2. **No configuration system:** Hyperparameters hard-coded in scripts
3. **No tests:** No automated validation of data pipeline
4. **No visualizations:** No side-by-side comparison outputs
5. **No batch inference:** Can only process single images

### 8.3 Missing Models

1. **Unified V1:** Referenced but not implemented
2. **Unified V2:** Target model not implemented

### 8.4 Infrastructure Gaps

1. **No configs/ directory:** YAML config system missing
2. **No inference/ directory:** Production inference not organized
3. **No tests/ directory:** Testing framework missing
4. **No visualization/ directory:** Comparison rendering missing
5. **requirements.txt:** Empty, needs specification
6. **README.md:** Empty, needs full documentation

### 8.5 Documentation

1. **No project README:** Cannot understand project from repository
2. **No training guide:** How to run training not documented
3. **No evaluation guide:** How to evaluate not documented

---

## 9. PROPOSED CHANGES & IMPLEMENTATION STRATEGY

### 9.1 Phase Priority

Following master prompt Part 55 execution order:

1. **PHASE 1** ✓ (Current): Repository inspection → AUDIT
2. **PHASE 2**: Create project structure and configs
3. **PHASE 3**: Implement paired dataset infrastructure
4. **PHASE 4**: Implement synthetic degradation
5. **PHASE 5**: Fix configuration system
6. **PHASE 6**: Preserve and verify existing baselines
7. **PHASE 7**: Preserve and verify Unified V1 (create if missing)
8. **PHASE 8**: Implement Unified V2
9. **PHASE 9**: Implement V2 loss (Charbonnier, Gradient, Frequency, Perceptual)
10. **PHASE 10**: Implement evaluation suite
11. **PHASE 11**: Implement visualizations
12. **PHASE 12-18**: Tests, smoke training, verification
13. **PHASE 19+**: Full training (manual, not automatic)

### 9.2 New Directories to Create

```
configs/           → YAML configuration files
datasets/          → Dataset infrastructure (.py modules)
models/            → Model definitions (V1, V2)
train/             → Training scripts
evaluation/        → Evaluation scripts
inference/         → Production inference
analysis/          → Analysis utilities
visualization/     → Visualization generation
tests/             → Unit tests
```

### 9.3 Key Design Decisions

1. **Preserve existing checkpoints** - Do NOT overwrite best_dncnn.pth or best_noise_aware_dncnn.pth
2. **Create Unified V1 first** - Before V2, establish reference implementation
3. **No automatic expensive training** - Require explicit user confirmation
4. **Data integrity** - Preserve out-of-range values, never clip unnecessarily
5. **Configuration-first** - All hyperparameters in YAML files
6. **Comprehensive testing** - Smoke tests before full training

---

## 10. TECHNICAL DEBT & NOTES

1. **NoiseAwareDnCNN hardcoding:** Noise variance coefficients hard-coded; should be configurable
2. **Dataset split strategy:** Currently computed on-the-fly; should be deterministic and saved
3. **Training infrastructure:** No AMP, no gradient clipping, no early stopping
4. **Evaluation infrastructure:** Separate scripts for each test; should consolidate
5. **No inference batching:** evaluate_dncnn.py uses batch_size=1 inefficiently

---

## 11. RISK MITIGATION

1. **Backup checkpoints:** Keep existing checkpoints in separate `checkpoints/baseline/` subdirectory
2. **Version tracking:** Each model version (V1, V2) has separate checkpoint/result directories
3. **Test preservation:** Existing result CSVs preserved in `results/baseline/` before new experiments
4. **Git management:** Use .gitignore to avoid committing checkpoints unintentionally

---

## 12. SUCCESS CRITERIA FOR PHASE 1 COMPLETION

- [x] Directory structure inspected
- [x] All Python files reviewed
- [x] Models identified and documented
- [x] Dataset structure understood
- [x] Data ranges confirmed
- [x] Evaluation methodology documented
- [x] Existing results identified
- [x] Missing components listed
- [x] Reusable components identified
- [x] Implementation plan created

**Next Action:** Proceed to PHASE 2 (Project Structure & Configuration)
