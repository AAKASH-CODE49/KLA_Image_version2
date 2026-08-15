"""Frequency-loss diagnostic using the project's real training data convention."""

import torch
from torch.utils.data import DataLoader

from datasets.dataset_utils import load_paired_images
from datasets.paired_dataset import PairedImageDataset
from models.losses import RestorationLoss
from models.unified_restorer_v2 import UnifiedRestorerV2

print("\n" + "=" * 80)
print("FREQUENCY LOSS DIAGNOSTIC")
print("=" * 80)

# Use the actual project API and a real training batch.
train_files, _, _ = load_paired_images("train/GT", "train/NoisyLR")
dataset = PairedImageDataset(
    filenames=train_files[:10],
    gt_dir="train/GT",
    noisy_dir="train/NoisyLR",
    training=True,
    scale=2,
    lr_patch_size=64,
    seed=42,
)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
noisy, gt = next(iter(loader))['noisy'], next(iter(loader))['gt']

# The above is intentionally split to avoid consuming the iterator twice; rebuild a clean batch.
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
noisy = batch["noisy"]
gt = batch["gt"]

model = UnifiedRestorerV2(1, 1, 64, 16, 16, 2)
model.eval()
with torch.no_grad():
    pred = model(noisy)

loss_fn = RestorationLoss(
    use_charbonnier=True,
    l1_weight=0.65,
    mse_weight=0.15,
    gradient_weight=0.10,
    frequency_weight=0.001,
    perceptual_weight=0.0,
)

raw_freq_loss = loss_fn.frequency_loss(pred, gt)
weighted_freq = 0.001 * raw_freq_loss
l1 = loss_fn.pixel_loss(pred, gt)
mse = loss_fn.mse_loss(pred, gt)
grad = loss_fn.gradient_loss(pred, gt)
total = loss_fn(pred, gt)

print(f"pred min/max/mean/std: {pred.min().item():.6f}, {pred.max().item():.6f}, {pred.mean().item():.6f}, {pred.std().item():.6f}")
print(f"target min/max/mean/std: {gt.min().item():.6f}, {gt.max().item():.6f}, {gt.mean().item():.6f}, {gt.std().item():.6f}")
print(f"raw frequency loss: {raw_freq_loss.item():.6f}")
print(f"weighted frequency contribution: {weighted_freq.item():.6f}")
print(f"L1 loss: {l1.item():.6f}")
print(f"MSE loss: {mse.item():.6f}")
print(f"gradient loss: {grad.item():.6f}")
print(f"total loss: {total.item():.6f}")
print(f"frequency finite: {torch.isfinite(raw_freq_loss).item()}")
print("\n" + "=" * 80)
