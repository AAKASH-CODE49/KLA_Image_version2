"""
Building blocks for image restoration networks.

Includes:
- Residual blocks with BatchNorm
- Residual channel attention blocks (RCAB)
- Channel attention
- Spatial attention
- Upsampling blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC BLOCKS
# ============================================================

class ResidualBlock(nn.Module):
    """
    Basic residual block with BatchNorm.
    
    Conv → BN → ReLU → Conv → BN → + residual
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return self.relu(out + residual)


class ResidualBlockNoBN(nn.Module):
    """
    Residual block without BatchNorm (for V2 improved training stability).
    
    Conv → ReLU → Conv → + residual
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        return self.relu(out + residual)


# ============================================================
# ATTENTION BLOCKS
# ============================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-like).
    
    Uses global average pooling and two FC layers to compute
    per-channel attention weights.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        b, c, h, w = x.size()
        avg_out = self.avg_pool(x)
        
        # Attention computation
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        avg_out = self.sigmoid(avg_out)
        
        return x * avg_out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Computes attention weights across spatial dimensions.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        x_att = torch.cat([avg_out, max_out], dim=1)
        
        # Spatial attention
        spatial_att = self.sigmoid(self.conv(x_att))
        
        return x * spatial_att


class ResidualChannelAttentionBlock(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    
    Structure:
    Conv → ReLU → Conv → Channel Attention → Residual
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 3
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.channel_attention = ChannelAttention(channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.channel_attention(out)
        
        return out + residual


# ============================================================
# UPSAMPLING BLOCKS
# ============================================================

class UpsampleBlock(nn.Module):
    """
    2× upsampling block using PixelShuffle.
    
    Conv → PixelShuffle(2) → ReLU
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(
            channels,
            channels * 4,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x


# ============================================================
# SPECIALIZED BRANCHES
# ============================================================

class HighFrequencyBranch(nn.Module):
    """
    High-frequency feature extraction branch.
    
    Extracts high-frequency components using:
    1. Learned blur filter
    2. Subtraction from input
    3. Processing with residual blocks
    """
    
    def __init__(self, channels: int, num_blocks: int = 3):
        super().__init__()
        
        # Learned low-pass filter
        self.blur = nn.Conv2d(
            channels, channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True
        )
        
        # Initialize as gentle blur
        with torch.no_grad():
            kernel = torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.]
            ], dtype=torch.float32)
            kernel = kernel / kernel.sum()
            
            if channels > 1:
                kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
            else:
                kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            # Adapt kernel size if needed
            self.blur.weight.data[:] = 0
            self.blur.weight.data[:, :, 1:4, 1:4] = kernel[:, :, :, :]
        
        # Process high-frequency features
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlockNoBN(channels))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Extract low-frequency
        blurred = self.blur(features)
        
        # Compute high-frequency
        high_freq = features - blurred
        
        # Process high-frequency features
        processed = self.blocks(high_freq)
        
        return processed


class EdgeAwareBranch(nn.Module):
    """
    Edge-aware feature extraction branch.
    
    Uses Sobel or learned gradient filters to detect edges
    and process them specifically.
    """
    
    def __init__(self, channels: int, num_blocks: int = 3):
        super().__init__()
        
        # Sobel-like filters for gradient computation
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32) / 8.0
        
        sobel_y = sobel_x.t()
        
        # Create learnable gradient filters
        self.grad_x = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )
        self.grad_y = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )
        
        # Initialize with Sobel
        with torch.no_grad():
            if channels > 1:
                for i in range(channels):
                    self.grad_x.weight[i, 0] = sobel_x
                    self.grad_y.weight[i, 0] = sobel_y
            else:
                self.grad_x.weight[0, 0] = sobel_x
                self.grad_y.weight[0, 0] = sobel_y
        
        # Edge processing blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlockNoBN(channels))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Compute gradients
        grad_x = self.grad_x(features)
        grad_y = self.grad_y(features)
        
        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # Process edges
        processed = self.blocks(edge_magnitude)
        
        return processed


class DegradationEncoder(nn.Module):
    """
    Encodes image degradation statistics into a feature vector.
    
    Computes statistics (mean, std, min, max, out-of-range ratio)
    and projects to an embedding vector that can condition the
    restoration network.
    """
    
    def __init__(
        self,
        input_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.input_features = input_features
        self.embedding_dim = embedding_dim
        
        # MLP to embed degradation statistics
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
    
    def compute_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute degradation statistics for each image in batch.
        
        Args:
            x: Tensor of shape (B, C, H, W)
        
        Returns:
            Statistics tensor of shape (B, num_stats)
        """
        
        b = x.size(0)
        stats = []
        
        # Flatten spatial dimensions
        x_flat = x.view(b, -1)
        
        # Compute statistics
        stats.append(x_flat.mean(dim=1, keepdim=True))           # mean
        stats.append(x_flat.std(dim=1, keepdim=True))            # std
        stats.append(x_flat.min(dim=1, keepdim=True)[0])        # min
        stats.append(x_flat.max(dim=1, keepdim=True)[0])        # max
        
        # Out-of-range ratio
        out_of_range = ((x_flat < 0) | (x_flat > 1)).float().mean(dim=1, keepdim=True)
        stats.append(out_of_range)
        
        return torch.cat(stats, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode degradation statistics into embedding.
        
        Args:
            x: Image tensor (B, C, H, W)
        
        Returns:
            Embedding vector (B, embedding_dim)
        """
        
        stats = self.compute_statistics(x)
        embedding = self.relu(self.fc1(stats))
        embedding = self.fc2(embedding)
        
        return embedding


# ============================================================
# MULTI-SCALE FEATURE EXTRACTION
# ============================================================

class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features at multiple scales.
    
    Uses:
    - 3×3 convolution (local)
    - 5×5 effective (dilated 3×3)
    - 7×7 effective (dilated 3×3)
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv_3x3 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        self.conv_5x5 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=True
        )
        
        self.conv_7x7 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            bias=True
        )
        
        self.fusion = nn.Conv2d(
            channels * 3, channels,
            kernel_size=1,
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale features
        f3 = self.relu(self.conv_3x3(x))
        f5 = self.relu(self.conv_5x5(x))
        f7 = self.relu(self.conv_7x7(x))
        
        # Concatenate and fuse
        fused = torch.cat([f3, f5, f7], dim=1)
        out = self.fusion(fused)
        
        return out

