"""
Loss functions for image restoration.

Includes:
- Charbonnier loss
- Gradient loss (Sobel)
- Frequency loss (FFT)
- Perceptual loss (LPIPS-compatible)
- Combined loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================
# BASIC LOSS COMPONENTS
# ============================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier (smooth L1) loss.
    
    Loss = sqrt((y - y_true)^2 + epsilon^2)
    
    Smoother than L1, less sensitive to outliers than L2.
    
    Args:
        epsilon: Regularization parameter
    """
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))


class GradientLoss(nn.Module):
    """
    Gradient-based loss using Sobel filters.
    
    Encourages the model to preserve edges and fine structures.
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
        # Sobel filters
        self.register_buffer("sobel_x", self._create_sobel_x())
        self.register_buffer("sobel_y", self._create_sobel_y())
    
    @staticmethod
    def _create_sobel_x() -> torch.Tensor:
        """Create Sobel X kernel."""
        kernel = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32) / 8.0
        return kernel.unsqueeze(0).unsqueeze(0)
    
    @staticmethod
    def _create_sobel_y() -> torch.Tensor:
        """Create Sobel Y kernel."""
        kernel = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ], dtype=torch.float32) / 8.0
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute gradient-domain loss using Sobel filters.

        The Sobel kernels are converted to the same device and dtype
        as the input tensors so this works correctly with CUDA AMP.
        """

        # Make Sobel kernels compatible with AMP and device.
        sobel_x = self.sobel_x.to(
            device=pred.device,
            dtype=pred.dtype
        )

        sobel_y = self.sobel_y.to(
            device=pred.device,
            dtype=pred.dtype
        )

        # Compute prediction gradients.
        pred_grad_x = F.conv2d(
            pred,
            sobel_x,
            padding=1
        )

        pred_grad_y = F.conv2d(
            pred,
            sobel_y,
            padding=1
        )

        # Compute target gradients.
        target_grad_x = F.conv2d(
            target,
            sobel_x,
            padding=1
        )

        target_grad_y = F.conv2d(
            target,
            sobel_y,
            padding=1
        )

        # Gradient magnitude.
        pred_grad = torch.sqrt(
            pred_grad_x ** 2 +
            pred_grad_y ** 2 +
            1e-6
        )

        target_grad = torch.sqrt(
            target_grad_x ** 2 +
            target_grad_y ** 2 +
            1e-6
        )

        # Compare gradients.
        return F.l1_loss(
            pred_grad,
            target_grad,
            reduction=self.reduction
        )


class FrequencyLoss(nn.Module):
    """
    Normalized frequency-domain loss using FFT.

    The raw FFT magnitude can become very large for 256x256
    images. Direct MSE on raw FFT magnitudes can therefore
    dominate the pixel reconstruction losses.

    This implementation normalizes the frequency magnitude
    before calculating the loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized frequency-domain loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Ground-truth image (B, C, H, W)

        Returns:
            Scalar frequency loss
        """

        # ----------------------------------------------------
        # Convert to frequency domain in float32
        # ----------------------------------------------------

        pred = pred.float()
        target = target.float()

        pred_fft = torch.fft.rfft2(
            pred,
            norm="ortho"
        )

        target_fft = torch.fft.rfft2(
            target,
            norm="ortho"
        )

        # ----------------------------------------------------
        # Stable log-magnitude comparison
        # ----------------------------------------------------

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        pred_log = torch.log1p(pred_mag)
        target_log = torch.log1p(target_mag)

        # ----------------------------------------------------
        # Frequency-domain reconstruction loss
        # ----------------------------------------------------

        loss = F.mse_loss(
            pred_log,
            target_log,
            reduction=self.reduction
        )

        return loss
# ============================================================
# PERCEPTUAL LOSS (LPIPS wrapper)
# ============================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using LPIPS.
    
    NOTE: Requires lpips package and pre-trained models.
    For grayscale images, we convert to RGB for compatibility.
    
    Args:
        net: Network type ('vgg', 'alex', 'squeeze')
    """
    
    def __init__(self, net: str = "vgg"):
        super().__init__()
        
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net, verbose=False)
            self.loss_fn.eval()
            for param in self.loss_fn.parameters():
                param.requires_grad = False
            self.available = True
        except ImportError:
            print("WARNING: lpips not installed. Perceptual loss will return 0.")
            self.available = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, C, H, W) or (B, 1, H, W)
            target: Target image (B, C, H, W) or (B, 1, H, W)
        
        Returns:
            Scalar loss
        """
        
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert grayscale to RGB by repeating channels
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1] if needed
        pred_norm = 2 * pred - 1
        target_norm = 2 * target - 1
        
        with torch.no_grad():
            loss = self.loss_fn(pred_norm, target_norm)
        
        return loss.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class RestorationLoss(nn.Module):
    """
    Combined loss for image restoration.
    
    Combines multiple loss components with configurable weights:
    - L1 (Charbonnier)
    - MSE
    - Gradient
    - Frequency
    - Perceptual
    
    Args:
        use_charbonnier: Use Charbonnier instead of L1
        charbonnier_eps: Epsilon for Charbonnier
        l1_weight: Weight for L1/Charbonnier loss
        mse_weight: Weight for MSE loss
        gradient_weight: Weight for gradient loss
        frequency_weight: Weight for frequency loss
        perceptual_weight: Weight for perceptual loss
        perceptual_net: Network for perceptual loss ('vgg', 'alex', 'squeeze')
    """
    
    def __init__(
        self,
        use_charbonnier: bool = True,
        charbonnier_eps: float = 1e-3,
        l1_weight: float = 0.65,
        mse_weight: float = 0.15,
        gradient_weight: float = 0.10,
        frequency_weight: float = 0.05,
        perceptual_weight: float = 0.05,
        perceptual_net: str = "vgg"
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        self.frequency_weight = frequency_weight
        self.perceptual_weight = perceptual_weight
        
        # Pixel loss
        if use_charbonnier:
            self.pixel_loss = CharbonnierLoss(epsilon=charbonnier_eps)
        else:
            self.pixel_loss = nn.L1Loss()
        
        # MSE loss
        self.mse_loss = nn.MSELoss()
        
        # Gradient loss (optional)
        if gradient_weight > 0:
            self.gradient_loss = GradientLoss(reduction="mean")
        else:
            self.gradient_loss = None
        
        # Frequency loss (optional)
        if frequency_weight > 0:
            self.frequency_loss = FrequencyLoss(reduction="mean")
        else:
            self.frequency_loss = None
        
        # Perceptual loss (optional)
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(net=perceptual_net)
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image
            target: Target image
            return_components: If True, return loss components dict
        
        Returns:
            Total loss or dict of loss components
        """
        
        losses = {}
        total_loss = torch.tensor(0.0, device=pred.device)
        
        # Pixel loss (L1 + MSE)
        if self.l1_weight > 0:
            l1 = self.pixel_loss(pred, target)
            losses['l1'] = l1.detach().item()
            total_loss = total_loss + self.l1_weight * l1
        
        if self.mse_weight > 0:
            mse = self.mse_loss(pred, target)
            losses['mse'] = mse.detach().item()
            total_loss = total_loss + self.mse_weight * mse
        
        # Gradient loss
        if self.gradient_loss is not None and self.gradient_weight > 0:
            grad_loss = self.gradient_loss(pred, target)
            losses['gradient'] = grad_loss.detach().item()
            total_loss = total_loss + self.gradient_weight * grad_loss
        
        # Frequency loss
        if self.frequency_loss is not None and self.frequency_weight > 0:
            freq_loss = self.frequency_loss(pred, target)
            losses['frequency'] = freq_loss.detach().item()
            total_loss = total_loss + self.frequency_weight * freq_loss
        
        # Perceptual loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc_loss.detach().item()
            total_loss = total_loss + self.perceptual_weight * perc_loss
        
        losses['total'] = total_loss.detach().item()
        
        if return_components:
            return total_loss, losses
        else:
            return total_loss


# ============================================================
# UTILITY FUNCTION
# ============================================================

def create_loss(config: Dict) -> RestorationLoss:
    """
    Create loss function from configuration.
    
    Args:
        config: Configuration dict with loss parameters
    
    Returns:
        RestorationLoss instance
    """
    
    return RestorationLoss(
        use_charbonnier=config.get('use_charbonnier', True),
        charbonnier_eps=config.get('charbonnier_epsilon', 1e-3),
        l1_weight=config.get('l1_weight', 0.65),
        mse_weight=config.get('mse_weight', 0.15),
        gradient_weight=config.get('gradient_weight', 0.10),
        frequency_weight=config.get('frequency_weight', 0.05),
        perceptual_weight=config.get('perceptual_weight', 0.05),
        perceptual_net=config.get('perceptual_net', 'vgg')
    )

