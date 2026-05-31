"""Configuration management for ColorMNet."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ColorMNetConfig:
    """Configuration for ColorMNet model and inference.

    Attributes:
        model_path: Path to model checkpoint
        device: Device to use ("cuda", "mps", "cpu", or None for auto)
        use_fp16: Whether to use FP16 mixed precision

        # Memory management
        max_mid_term_frames: Maximum frames in mid-term memory (T_max)
        min_mid_term_frames: Minimum frames in mid-term memory (T_min)
        max_long_term_elements: Maximum elements in long-term memory (LT_max)
        num_prototypes: Number of prototypes (P)
        mem_every: Memory update frequency (r)

        # Processing options
        size: Resize shorter side to this (-1 for original)
        enable_long_term: Enable long-term memory
        top_k: Top-k matching
        deep_update_every: Deep update frequency (-1 to sync with mem_every)

        # Model architecture (from ColorMNet paper)
        key_dim: Key dimension
        value_dim: Value dimension
        hidden_dim: Hidden dimension
        single_object: Single object mode
    """

    # Model and device
    model_path: Optional[Path] = None
    device: Optional[str] = None
    use_fp16: bool = True

    # Memory management parameters
    max_mid_term_frames: int = 10
    min_mid_term_frames: int = 5
    max_long_term_elements: int = 10000
    num_prototypes: int = 128
    mem_every: int = 5
    deep_update_every: int = -1

    # Processing
    size: int = -1
    enable_long_term: bool = True
    top_k: int = 30

    # Model architecture
    key_dim: int = 64
    value_dim: int = 512
    hidden_dim: int = 64
    single_object: bool = False

    # Performance optimization
    enable_corr: bool = True  # Use CUDA correlation sampler (2-5x faster if available)

    # Internal flags
    enable_long_term_count_usage: bool = False
    benchmark: bool = False

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Dictionary of configuration values
        """
        return {
            'device': self.device,
            'max_mid_term_frames': self.max_mid_term_frames,
            'min_mid_term_frames': self.min_mid_term_frames,
            'max_long_term_elements': self.max_long_term_elements,
            'num_prototypes': self.num_prototypes,
            'mem_every': self.mem_every,
            'deep_update_every': self.deep_update_every,
            'size': self.size,
            'enable_long_term': self.enable_long_term,
            'enable_long_term_count_usage': self.enable_long_term_count_usage,
            'top_k': self.top_k,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'hidden_dim': self.hidden_dim,
            'single_object': self.single_object,
            'enable_corr': self.enable_corr,
            'benchmark': self.benchmark,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ColorMNetConfig':
        """Create config from dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            ColorMNetConfig instance
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default(cls, model_path: Optional[Path] = None) -> 'ColorMNetConfig':
        """Create default configuration.

        Args:
            model_path: Path to model checkpoint

        Returns:
            ColorMNetConfig with default values
        """
        return cls(model_path=model_path)

    @classmethod
    def low_memory(cls, model_path: Optional[Path] = None) -> 'ColorMNetConfig':
        """Create low-memory configuration for GPUs with limited VRAM.

        Args:
            model_path: Path to model checkpoint

        Returns:
            ColorMNetConfig optimized for low memory
        """
        return cls(
            model_path=model_path,
            use_fp16=True,
            max_mid_term_frames=5,
            min_mid_term_frames=3,
            max_long_term_elements=5000,
            num_prototypes=64,
            mem_every=10,
        )

    @classmethod
    def high_quality(cls, model_path: Optional[Path] = None) -> 'ColorMNetConfig':
        """Create high-quality configuration for GPUs with ample VRAM.

        Args:
            model_path: Path to model checkpoint

        Returns:
            ColorMNetConfig optimized for quality
        """
        return cls(
            model_path=model_path,
            use_fp16=False,
            max_mid_term_frames=15,
            min_mid_term_frames=7,
            max_long_term_elements=15000,
            num_prototypes=256,
            mem_every=3,
        )
