"""
Image restoration models.

Includes:
- DnCNN (baseline)
- NoiseAwareDnCNN (baseline with noise conditioning)
- Unified V1 (improved restoration)
- Unified V2 (advanced restoration with detail preservation)
"""

from .noise_aware_dncnn import NoiseAwareDnCNN
from .unified_restorer_v1 import UnifiedRestorerV1
from .unified_restorer_v2 import UnifiedRestorerV2
from .restoration_blocks import (
    ResidualBlock,
    ResidualBlockNoBN,
    ChannelAttention,
    SpatialAttention,
    ResidualChannelAttentionBlock,
    HighFrequencyBranch,
    EdgeAwareBranch,
    DegradationEncoder,
    MultiScaleFeatureExtractor,
    UpsampleBlock
)

__all__ = [
    'NoiseAwareDnCNN',
    'UnifiedRestorerV1',
    'UnifiedRestorerV2',
    'ResidualBlock',
    'ResidualBlockNoBN',
    'ChannelAttention',
    'SpatialAttention',
    'ResidualChannelAttentionBlock',
    'HighFrequencyBranch',
    'EdgeAwareBranch',
    'DegradationEncoder',
    'MultiScaleFeatureExtractor',
    'UpsampleBlock'
]


def load_model(model_name: str, **kwargs):
    """
    Load a model by name.
    
    Args:
        model_name: One of "dncnn", "noise_aware", "v1", "v2"
        **kwargs: Model arguments
    
    Returns:
        Model instance
    """
    
    if model_name.lower() in ["dncnn", "baseline"]:
        # Import DnCNN from original implementation
        raise NotImplementedError("DnCNN loading not yet implemented")
    
    elif model_name.lower() in ["noise_aware", "noisaware"]:
        return NoiseAwareDnCNN(**kwargs)
    
    elif model_name.lower() in ["v1", "unified_v1"]:
        return UnifiedRestorerV1(**kwargs)
    
    elif model_name.lower() in ["v2", "unified_v2"]:
        return UnifiedRestorerV2(**kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

