"""
Unified Restorer V1.

An improved restoration model with:
- Attention mechanisms
- Degradation-aware conditioning
- Enhanced residual backbone
- ×2 super-resolution

Target: ~1.3M parameters
Expected performance: 28.0 dB PSNR
"""

import torch
import torch.nn as nn
from .restoration_blocks import (
    ResidualChannelAttentionBlock,
    ChannelAttention,
    UpsampleBlock,
    DegradationEncoder
)


class UnifiedRestorerV1(nn.Module):
    """
    Unified Image Restoration Network V1.
    
    Architecture:
    1. Input encoding with noise sigma conditioning
    2. Feature extraction with attention blocks
    3. Degradation-aware feature refinement
    4. 2× upsampling
    5. Output reconstruction
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for grayscale)
        num_features: Number of feature channels
        num_blocks: Number of residual blocks
        reduction: Reduction factor for channel attention
        scale: Upsampling scale (2 or 4)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 12,
        reduction: int = 16,
        scale: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.scale = scale
        
        # ================================================
        # DEGRADATION ENCODER
        # ================================================
        # Computes embedding from image statistics
        # Uses: mean, std, min, max, out-of-range ratio
        
        self.degradation_encoder = DegradationEncoder(
            input_features=5,
            embedding_dim=32,
            hidden_dim=64
        )
        
        # ================================================
        # INPUT ENCODING
        # ================================================
        # Combines:
        # 1. Noisy image features
        # 2. Degradation-aware conditioning
        
        self.head_conv = nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # FiLM-style conditioning: scale and shift features based on degradation
        self.degradation_to_scale = nn.Linear(32, num_features)
        self.degradation_to_shift = nn.Linear(32, num_features)
        
        self.relu = nn.ReLU(inplace=True)
        
        # ================================================
        # RESIDUAL ATTENTION BACKBONE
        # ================================================
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResidualChannelAttentionBlock(
                    channels=num_features,
                    reduction=reduction,
                    kernel_size=3
                )
            )
        
        self.backbone = nn.Sequential(*blocks)
        
        # ================================================
        # FEATURE REFINEMENT
        # ================================================
        
        self.refine_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Global residual connection
        self.global_residual = True
        
        # ================================================
        # UPSAMPLING
        # ================================================
        
        if scale == 2:
            self.upsample = UpsampleBlock(num_features)
        elif scale == 4:
            self.upsample = nn.Sequential(
                UpsampleBlock(num_features),
                UpsampleBlock(num_features)
            )
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        
        # ================================================
        # OUTPUT RECONSTRUCTION
        # ================================================
        
        self.tail_conv = nn.Conv2d(
            num_features,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restore image.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Restored tensor (B, C, H*scale, W*scale)
        """
        
        # ============================================
        # DEGRADATION ENCODING
        # ============================================
        
        degradation_embedding = self.degradation_encoder(x)  # (B, 32)
        
        # ============================================
        # INPUT FEATURE EXTRACTION
        # ============================================
        
        features = self.head_conv(x)  # (B, F, H, W)
        features = self.relu(features)
        
        # ============================================
        # FEATURE REFINEMENT WITH CONDITIONING
        # ============================================
        
        # Compute FiLM parameters from degradation
        scale_params = self.degradation_to_scale(degradation_embedding)  # (B, F)
        shift_params = self.degradation_to_shift(degradation_embedding)   # (B, F)
        
        # Reshape for broadcasting: (B, F, 1, 1)
        scale_params = scale_params.view(-1, self.num_features, 1, 1)
        shift_params = shift_params.view(-1, self.num_features, 1, 1)
        
        # Apply FiLM conditioning
        features = features * (1.0 + scale_params) + shift_params
        
        # Residual backbone
        backbone_features = self.backbone(features)
        
        # Feature refinement
        refined = self.refine_conv(backbone_features)
        
        # Global residual connection (skip connection to input)
        if self.global_residual:
            refined = refined + features
        
        # ============================================
        # UPSAMPLING
        # ============================================
        
        upsampled = self.upsample(refined)
        
        # ============================================
        # OUTPUT RECONSTRUCTION
        # ============================================
        
        output = self.tail_conv(upsampled)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_unified_v1():
    """Quick test of the model."""
    
    model = UnifiedRestorerV1(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=12,
        reduction=16,
        scale=2
    )
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"Unified V1 Total Parameters: {total_params:,}")
    print(f"Expected: ~1,286,113")
    
    # Test forward pass
    x = torch.randn(2, 1, 64, 64)  # (B, C, H, W) - LR image
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {y.shape}"
    
    print("✓ Unified V1 test passed")


if __name__ == "__main__":
    test_unified_v1()

