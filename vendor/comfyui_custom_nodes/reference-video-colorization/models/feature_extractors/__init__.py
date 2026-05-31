"""
Modern Feature Extractors for Video Colorization

This package provides drop-in replacements for VGG19/ResNet50 feature extractors
with modern vision models (DINOv2, CLIP, etc.) that offer superior semantic understanding.

IMPORTANT LIMITATION:
DeepExemplar's NonlocalNet and ColorNet were trained specifically with VGG19 features.
While this package provides dimension-compatible projections from DINOv2/CLIP to VGG19,
these projections are NOT trained and will produce suboptimal results (color artifacts).

For production use, stick with VGG19. Alternative encoders are EXPERIMENTAL and would
require fine-tuning the entire colorization network to work properly.

Quick usage:
    from models.feature_extractors import get_feature_encoder

    # Get encoder by name
    encoder = get_feature_encoder('vgg19')  # Recommended

    # Use exactly like VGG19
    features = encoder(image, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
"""

from typing import Literal, Optional
import torch.nn as nn


def get_feature_encoder(
    encoder_type: Literal['vgg19', 'dinov2_vits', 'dinov2_vitb', 'dinov2_vitl', 'dinov2_vitg',
                          'clip_vitb', 'clip_vitl', 'resnet50'] = 'vgg19',
    device: str = 'cuda',
    **kwargs
) -> nn.Module:
    """Get a feature encoder by name.

    Args:
        encoder_type: Type of encoder to load
            - 'vgg19': Original VGG19 (baseline)
            - 'dinov2_vitb': DINOv2 ViT-Base (recommended, best quality/speed)
            - 'dinov2_vitl': DINOv2 ViT-Large (best quality, slower)
            - 'dinov2_vitg': DINOv2 ViT-Giant (highest quality, very slow)
            - 'clip_vitb': CLIP ViT-B/16 (good for text-guided)
            - 'clip_vitl': CLIP ViT-L/14 (best text understanding)
            - 'resnet50': ResNet50 (ColorMNet baseline)
        device: Device to load model on ('cuda' or 'cpu')
        **kwargs: Additional arguments passed to encoder constructor

    Returns:
        Feature encoder module compatible with VGG19 interface

    Example:
        >>> encoder = get_feature_encoder('dinov2_vitb')
        >>> features = encoder(image, ["r12", "r22", "r32", "r42", "r52"])
        >>> print([f.shape for f in features])
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'vgg19':
        try:
            from ..vgg19_gray import VGG19_pytorch
            return VGG19_pytorch().to(device).eval()
        except ImportError:
            print("Warning: Could not import VGG19_pytorch, trying alternative...")
            # Fallback to standard VGG19
            import torchvision.models as models
            from .vgg_wrapper import VGG19Wrapper
            vgg = models.vgg19(pretrained=True)
            return VGG19Wrapper(vgg).to(device).eval()

    elif encoder_type in ['dinov2_vits', 'dinov2_vitb', 'dinov2_vitl', 'dinov2_vitg']:
        from .dinov2_encoder import DINOv2Encoder
        model_map = {
            'dinov2_vits': 'dinov2_vits14',
            'dinov2_vitb': 'dinov2_vitb14',
            'dinov2_vitl': 'dinov2_vitl14',
            'dinov2_vitg': 'dinov2_vitg14',
        }
        model_name = model_map[encoder_type]
        return DINOv2Encoder(model_name=model_name, **kwargs).to(device).eval()

    elif encoder_type in ['clip_vitb', 'clip_vitl']:
        from .clip_encoder import CLIPEncoder
        model_map = {
            'clip_vitb': 'ViT-B/16',
            'clip_vitl': 'ViT-L/14',
        }
        model_name = model_map[encoder_type]
        return CLIPEncoder(model_name=model_name, device=device, **kwargs).eval()

    elif encoder_type == 'resnet50':
        import torchvision.models as models
        from .resnet_wrapper import ResNet50Wrapper
        resnet = models.resnet50(pretrained=True)
        return ResNet50Wrapper(resnet).to(device).eval()

    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Valid options: vgg19, dinov2_vits, dinov2_vitb, dinov2_vitl, dinov2_vitg, "
            f"clip_vitb, clip_vitl, resnet50"
        )


def get_encoder_info(encoder_type: str) -> dict:
    """Get information about an encoder.

    Args:
        encoder_type: Name of encoder

    Returns:
        Dictionary with encoder metadata

    Example:
        >>> info = get_encoder_info('dinov2_vitb')
        >>> print(info['params'], info['speed_rating'])
    """
    encoder_info = {
        'vgg19': {
            'name': 'VGG19',
            'year': 2014,
            'params': '144M',
            'speed_rating': 5,
            'quality_rating': 3,
            'description': 'Original baseline, ImageNet pre-trained CNN'
        },
        'resnet50': {
            'name': 'ResNet50',
            'year': 2015,
            'params': '25M',
            'speed_rating': 5,
            'quality_rating': 3,
            'description': 'ColorMNet baseline, faster than VGG19'
        },
        'dinov2_vits': {
            'name': 'DINOv2 ViT-Small',
            'year': 2023,
            'params': '21M',
            'speed_rating': 5,
            'quality_rating': 4,
            'description': 'Smallest DINOv2, faster than vitb with good quality'
        },
        'dinov2_vitb': {
            'name': 'DINOv2 ViT-Base',
            'year': 2023,
            'params': '86M',
            'speed_rating': 4,
            'quality_rating': 5,
            'description': 'Best overall: superior semantic features, self-supervised'
        },
        'dinov2_vitl': {
            'name': 'DINOv2 ViT-Large',
            'year': 2023,
            'params': '304M',
            'speed_rating': 3,
            'quality_rating': 5,
            'description': 'Highest quality semantic matching'
        },
        'dinov2_vitg': {
            'name': 'DINOv2 ViT-Giant',
            'year': 2023,
            'params': '1.1B',
            'speed_rating': 2,
            'quality_rating': 5,
            'description': 'Maximum quality, very high VRAM usage'
        },
        'clip_vitb': {
            'name': 'CLIP ViT-B/16',
            'year': 2021,
            'params': '86M',
            'speed_rating': 4,
            'quality_rating': 4,
            'description': 'Good semantic features, enables text guidance'
        },
        'clip_vitl': {
            'name': 'CLIP ViT-L/14',
            'year': 2021,
            'params': '304M',
            'speed_rating': 3,
            'quality_rating': 5,
            'description': 'Best text-guided semantic understanding'
        },
    }

    if encoder_type not in encoder_info:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    return encoder_info[encoder_type]


def list_available_encoders() -> list:
    """List all available encoder types.

    Returns:
        List of encoder type strings
    """
    return [
        'vgg19',
        'resnet50',
        'dinov2_vits',
        'dinov2_vitb',
        'dinov2_vitl',
        'dinov2_vitg',
        'clip_vitb',
        'clip_vitl',
    ]


def compare_encoders(verbose: bool = True) -> dict:
    """Compare all available encoders.

    Args:
        verbose: Print comparison table

    Returns:
        Dictionary with encoder comparisons
    """
    encoders = list_available_encoders()
    comparison = {}

    for encoder in encoders:
        comparison[encoder] = get_encoder_info(encoder)

    if verbose:
        print("\n" + "="*80)
        print("Feature Encoder Comparison")
        print("="*80)
        print(f"{'Encoder':<20} {'Year':<6} {'Params':<8} {'Speed':<7} {'Quality':<7}")
        print("-"*80)

        for encoder, info in comparison.items():
            speed = "⚡" * info['speed_rating']
            quality = "⭐" * info['quality_rating']
            print(f"{info['name']:<20} {info['year']:<6} {info['params']:<8} {speed:<7} {quality:<7}")

        print("-"*80)
        print("\nRecommendations:")
        print("  • Best overall: dinov2_vitb (great quality, good speed)")
        print("  • Fastest: vgg19 or resnet50 (original baseline)")
        print("  • Best quality: dinov2_vitl or dinov2_vitg")
        print("  • Text-guided: clip_vitb or clip_vitl")
        print("="*80 + "\n")

    return comparison


# Convenience exports
__all__ = [
    'get_feature_encoder',
    'get_encoder_info',
    'list_available_encoders',
    'compare_encoders',
]


if __name__ == "__main__":
    # Demo
    print("Feature Extractors Demo")
    print("="*80)

    # Show comparison
    compare_encoders()

    # Test loading
    print("\nTesting encoder loading...")
    try:
        encoder = get_feature_encoder('dinov2_vitb')
        print(f"✓ Successfully loaded DINOv2 ViT-Base")
        print(f"  Model: {encoder.__class__.__name__}")
    except Exception as e:
        print(f"✗ Failed to load DINOv2: {e}")

    print("\n" + "="*80)
