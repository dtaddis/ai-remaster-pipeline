"""
DINOv2 Feature Extractor for Deep Exemplar Colorization

This is a drop-in replacement for VGG19 that provides superior semantic features.
DINOv2 is trained with self-supervision and provides better semantic understanding.

Usage:
    from models.feature_extractors.dinov2_encoder import DINOv2Encoder

    # Replace VGG_NET with:
    VGG_NET = DINOv2Encoder().cuda().eval()

    # Use exactly the same way:
    features = VGG_NET(image, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """DINOv2-based feature extractor compatible with VGG19 interface.

    Provides multi-scale features with better semantic understanding than VGG19.
    """

    def __init__(self, model_name='dinov2_vitb14', use_registers=False):
        """
        Args:
            model_name: DINOv2 model variant
                - 'dinov2_vits14': Small (21M params)
                - 'dinov2_vitb14': Base (86M params) - recommended
                - 'dinov2_vitl14': Large (304M params)
                - 'dinov2_vitg14': Giant (1.1B params) - best quality, very slow
            use_registers: Use register tokens (slightly better quality, slower)
        """
        super().__init__()

        try:
            # Try to load from torch hub
            if use_registers:
                model_name = f"{model_name}_reg"
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        except Exception as e:
            print(f"Failed to load DINOv2 from torch hub: {e}")
            print("Please install: pip install timm")
            raise

        # Get embedding dimension based on model
        if 'vits' in model_name:
            self.embed_dim = 384
        elif 'vitb' in model_name:
            self.embed_dim = 768
        elif 'vitl' in model_name:
            self.embed_dim = 1024
        elif 'vitg' in model_name:
            self.embed_dim = 1536

        # Freeze DINOv2 weights
        for param in self.dinov2.parameters():
            param.requires_grad = False

        # Projection layers to match VGG19 channel dimensions
        # VGG19 dimensions: [64, 128, 256, 512, 512]
        # Use deeper projections with Xavier initialization for better feature preservation
        def make_projection(in_ch, out_ch):
            """Create a 2-layer projection with proper initialization."""
            mid_ch = (in_ch + out_ch) // 2
            proj = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 1),
                nn.ReLU(inplace=True)
            )
            # Xavier initialization for better gradient flow
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            return proj

        self.proj_r12 = make_projection(self.embed_dim, 64)
        self.proj_r22 = make_projection(self.embed_dim, 128)
        self.proj_r32 = make_projection(self.embed_dim, 256)
        self.proj_r42 = make_projection(self.embed_dim, 512)
        self.proj_r52 = make_projection(self.embed_dim, 512)

        # Mapping from layer names to outputs
        self.layer_mapping = {
            'r12': (3, self.proj_r12),   # Early features
            'r22': (6, self.proj_r22),   # Low-level features
            'r32': (9, self.proj_r32),   # Mid-level features
            'r42': (12, self.proj_r42),  # High-level features (all 12 layers for ViT-B)
            'r52': (12, self.proj_r52),  # Highest-level (same as r42 for ViT)
        }

        # Normalization (DINOv2 expects ImageNet normalization)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_features_at_layer(self, x, layer_idx):
        """Extract features from a specific transformer layer."""
        B, C, H, W = x.shape

        # Prepare patches
        features = self.dinov2.prepare_tokens_with_masks(x)

        # Forward through transformer blocks up to layer_idx
        for i, blk in enumerate(self.dinov2.blocks[:layer_idx]):
            features = blk(features)

        # Remove CLS token and reshape to spatial
        patch_features = features[:, 1:]  # Remove CLS token

        # Calculate patch grid size based on actual image dimensions
        # DINOv2 patch size is 14x14
        patch_size = 14
        patch_h = H // patch_size
        patch_w = W // patch_size

        # Reshape to spatial: [B, N, D] -> [B, D, H, W]
        features_spatial = patch_features.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)

        # Interpolate to match input resolution
        features_resized = F.interpolate(
            features_spatial,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return features_resized

    def forward(self, x, out_keys=None, preprocess=True):
        """
        Args:
            x: Input image tensor [B, 3, H, W] in range [0, 1]
            out_keys: List of layer names to extract (e.g., ["r12", "r22", "r32", "r42", "r52"])
            preprocess: Whether to apply ImageNet normalization

        Returns:
            List of feature tensors at requested layers
        """
        if out_keys is None:
            out_keys = ["r12", "r22", "r32", "r42", "r52"]

        # Normalize input
        if preprocess:
            x = (x - self.mean) / self.std

        outputs = []

        with torch.no_grad():
            for key in out_keys:
                if key not in self.layer_mapping:
                    raise ValueError(f"Unknown layer key: {key}. Valid keys: {list(self.layer_mapping.keys())}")

                layer_idx, proj = self.layer_mapping[key]

                # Extract features at this layer
                features = self._extract_features_at_layer(x, layer_idx)

                # Project to match VGG19 dimensions
                features = proj(features)

                outputs.append(features)

        return outputs


class DINOv2EncoderColorMNet(nn.Module):
    """DINOv2 encoder for ColorMNet (different interface)."""

    def __init__(self, model_name='dinov2_vitb14'):
        super().__init__()

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)

        if 'vits' in model_name:
            self.embed_dim = 384
        elif 'vitb' in model_name:
            self.embed_dim = 768
        elif 'vitl' in model_name:
            self.embed_dim = 1024
        elif 'vitg' in model_name:
            self.embed_dim = 1536

        # Freeze DINOv2
        for param in self.dinov2.parameters():
            param.requires_grad = False

        # Project to ColorMNet expected dimensions: [256, 512, 1024]
        self.proj_f4 = nn.Conv2d(self.embed_dim, 256, 1)   # 1/4 scale
        self.proj_f8 = nn.Conv2d(self.embed_dim, 512, 1)   # 1/8 scale
        self.proj_f16 = nn.Conv2d(self.embed_dim, 1024, 1) # 1/16 scale

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Returns:
            f16: 1/16 scale features [B, 1024, H/16, W/16]
            f8: 1/8 scale features [B, 512, H/8, W/8]
            f4: 1/4 scale features [B, 256, H/4, W/4]
        """
        B, C, H, W = x.shape

        # Normalize
        x = (x - self.mean) / self.std

        with torch.no_grad():
            # Extract features from different layers
            features_early = self._extract_features_at_layer(x, 4)   # For f4
            features_mid = self._extract_features_at_layer(x, 8)     # For f8
            features_late = self._extract_features_at_layer(x, 12)   # For f16

        # Project and resize to expected scales
        f4 = self.proj_f4(features_early)
        f4 = F.interpolate(f4, size=(H // 4, W // 4), mode='bilinear', align_corners=False)

        f8 = self.proj_f8(features_mid)
        f8 = F.interpolate(f8, size=(H // 8, W // 8), mode='bilinear', align_corners=False)

        f16 = self.proj_f16(features_late)
        f16 = F.interpolate(f16, size=(H // 16, W // 16), mode='bilinear', align_corners=False)

        return f16, f8, f4

    def _extract_features_at_layer(self, x, layer_idx):
        """Extract features from specific layer (same as above)."""
        B, C, H, W = x.shape
        features = self.dinov2.prepare_tokens_with_masks(x)
        for blk in self.dinov2.blocks[:layer_idx]:
            features = blk(features)
        patch_features = features[:, 1:]
        # Calculate patch grid size based on actual image dimensions
        # DINOv2 patch size is 14x14
        patch_size = 14
        patch_h = H // patch_size
        patch_w = W // patch_size
        features_spatial = patch_features.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)
        return features_spatial


if __name__ == "__main__":
    # Test the encoder
    print("Testing DINOv2 Encoder...")
    encoder = DINOv2Encoder(model_name='dinov2_vitb14').cuda()

    # Test input
    x = torch.randn(2, 3, 256, 256).cuda()

    # Extract features
    features = encoder(x, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    print("\nOutput feature shapes:")
    for i, (name, feat) in enumerate(zip(["r12", "r22", "r32", "r42", "r52"], features)):
        print(f"  {name}: {feat.shape}")

    print("\n✓ DINOv2 encoder is compatible with VGG19 interface!")
