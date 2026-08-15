"""
Unified Restorer V2.

Advanced restoration model with:
- Enhanced attention mechanisms
- High-frequency detail branch
- Edge-aware branch
- Multi-scale feature extraction
- Advanced degradation conditioning
- ×2 super-resolution

Target: 2M–5M parameters
Expected improvements: Better detail preservation, edge recovery
"""

import torch
import torch.nn as nn
from .restoration_blocks import (
    ResidualChannelAttentionBlock,
    ResidualBlockNoBN,
    ChannelAttention,
    UpsampleBlock,
    DegradationEncoder,
    HighFrequencyBranch,
    EdgeAwareBranch,
    MultiScaleFeatureExtractor
)


class UnifiedRestorerV2(nn.Module):
    """
    Unified Image Restoration Network V2.
    
    Major improvements over V1:
    1. High-frequency detail branch for texture/edge recovery
    2. Edge-aware branch for structure preservation
    3. Multi-scale feature extraction
    4. Enhanced degradation conditioning
    5. Improved loss function support
    
    Architecture overview:
    
                        NoisyLR
                           │
                           ▼
                  Robust Input Encoder
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
         Restoration Features  Degradation Encoder
                 │                   │
                 │                   ▼
                 │        Degradation Embedding
                 │                   │
                 └─────────┬─────────┘
                           ▼
                   Multi-scale Backbone
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
            RCAB       Residual    Multi-scale
            Blocks     Blocks      Features
              │            │            │
              └────────────┼────────────┘
                           ▼
                    Detail Refinement
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
           Low Frequency       High Frequency
             Features            Features
                  │                 │
                  │    ┌────────────┘
                  │    │
                  ▼    ▼
                 Fusion
                  │
        ┌─────────┤
        │         │
        ▼         ▼
      Edge     Detail Refinement
      Branch   (High-freq)
        │         │
        └────┬────┘
             ▼
          Fusion
             │
             ▼
          ×2 Upsample
             │
             ▼
       Reconstruction
             │
             ▼
           Output
    
    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (1 for grayscale)
        num_features: Base number of feature channels
        num_blocks: Number of residual attention blocks
        reduction: Channel attention reduction factor
        scale: Upsampling scale (2 or 4)
        use_hf_branch: Enable high-frequency branch
        use_edge_branch: Enable edge-aware branch
        hf_strength: Weight for high-frequency branch
        edge_strength: Weight for edge-aware branch
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        reduction: int = 16,
        scale: int = 2,
        use_hf_branch: bool = True,
        use_edge_branch: bool = True,
        hf_strength: float = 0.5,
        edge_strength: float = 0.3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.scale = scale
        self.use_hf_branch = use_hf_branch
        self.use_edge_branch = use_edge_branch
        
        # ================================================
        # DEGRADATION ENCODER
        # ================================================
        
        self.degradation_encoder = DegradationEncoder(
            input_features=5,
            embedding_dim=32,
            hidden_dim=64
        )
        
        # ================================================
        # ROBUST INPUT ENCODER
        # ================================================
        # Extracts initial features without BatchNorm
        # for better handling of out-of-range values
        
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # FiLM conditioning
        self.degradation_to_scale = nn.Linear(32, num_features)
        self.degradation_to_shift = nn.Linear(32, num_features)
        
        # ================================================
        # MULTI-SCALE BACKBONE
        # ================================================
        
        # Main residual attention blocks
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
        
        # Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(num_features)
        
        # ================================================
        # HIGH-FREQUENCY BRANCH
        # ================================================
        
        if use_hf_branch:
            self.hf_branch = HighFrequencyBranch(
                channels=num_features,
                num_blocks=4
            )
            self.hf_strength = hf_strength
        
        # ================================================
        # EDGE-AWARE BRANCH
        # ================================================
        
        if use_edge_branch:
            self.edge_branch = EdgeAwareBranch(
                channels=num_features,
                num_blocks=3
            )
            self.edge_strength = edge_strength
        
        # ================================================
        # FEATURE FUSION
        # ================================================
        
        num_input_branches = 1 + (1 if use_hf_branch else 0) + (1 if use_edge_branch else 0)
        
        if num_input_branches > 1:
            self.fusion = nn.Sequential(
                nn.Conv2d(num_features * num_input_branches, num_features, kernel_size=1, bias=True),
                nn.ReLU(inplace=True)
            )
        
        # ================================================
        # DETAIL REFINEMENT
        # ================================================
        
        self.refine_blocks = nn.Sequential(
            ResidualBlockNoBN(num_features),
            ResidualBlockNoBN(num_features)
        )
        
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
        
        self.output_refine = nn.Conv2d(
            num_features,
            num_features // 2,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        self.tail_conv = nn.Conv2d(
            num_features // 2,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restore image with advanced detail preservation.
        
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
        
        features = self.input_encoder(x)  # (B, F, H, W)
        
        # Compute FiLM conditioning parameters
        scale_params = self.degradation_to_scale(degradation_embedding)  # (B, F)
        shift_params = self.degradation_to_shift(degradation_embedding)   # (B, F)
        
        scale_params = scale_params.view(-1, self.num_features, 1, 1)
        shift_params = shift_params.view(-1, self.num_features, 1, 1)
        
        # Apply FiLM conditioning
        features = features * (1.0 + scale_params) + shift_params
        
        # ============================================
        # MULTI-SCALE BACKBONE
        # ============================================
        
        backbone_out = self.backbone(features)
        multi_scale_out = self.multi_scale(backbone_out)
        
        # ============================================
        # SPECIALIZED BRANCHES
        # ============================================
        
        branch_outputs = [multi_scale_out]
        
        if self.use_hf_branch:
            hf_features = self.hf_branch(multi_scale_out)
            branch_outputs.append(hf_features * self.hf_strength)
        
        if self.use_edge_branch:
            edge_features = self.edge_branch(multi_scale_out)
            branch_outputs.append(edge_features * self.edge_strength)
        
        # ============================================
        # FEATURE FUSION
        # ============================================
        
        if len(branch_outputs) > 1:
            fused = torch.cat(branch_outputs, dim=1)
            fused = self.fusion(fused)
        else:
            fused = branch_outputs[0]
        
        # ============================================
        # DETAIL REFINEMENT
        # ============================================
        
        refined = self.refine_blocks(fused)
        
        # Residual connection
        refined = refined + features
        
        # ============================================
        # UPSAMPLING
        # ============================================
        
        upsampled = self.upsample(refined)
        
        # ============================================
        # OUTPUT RECONSTRUCTION
        # ============================================
        
        output = self.output_refine(upsampled)
        output = self.tail_conv(output)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_unified_v2():
    """Quick test of the V2 model."""
    
    model = UnifiedRestorerV2(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=16,
        reduction=16,
        scale=2,
        use_hf_branch=True,
        use_edge_branch=True,
        hf_strength=0.5,
        edge_strength=0.3
    )
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"Unified V2 Total Parameters: {total_params:,}")
    print(f"Target range: 2,000,000 - 5,000,000")
    
    # Test forward pass
    x = torch.randn(2, 1, 64, 64)  # (B, C, H, W) - LR image
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {y.shape}"
    
    # Test backward pass
    loss = y.mean()
    loss.backward()
    print("✓ Backward pass successful")
    
    print("✓ Unified V2 test passed")


if __name__ == "__main__":
    test_unified_v2()

