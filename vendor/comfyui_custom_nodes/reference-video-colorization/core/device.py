"""Device management for ColorMNet."""

import torch
from typing import Literal, Optional, Tuple
from .logger import get_logger
from .exceptions import InsufficientVRAMError


DeviceType = Literal["cuda", "mps", "cpu"]


class DeviceManager:
    """Manages device selection and memory monitoring."""

    def __init__(self, device: Optional[str] = None, use_fp16: bool = True):
        """Initialize device manager.

        Args:
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
            use_fp16: Whether to use FP16 mixed precision
        """
        self.logger = get_logger()
        self.device = self._select_device(device)
        self.use_fp16 = use_fp16 and self._supports_fp16()

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"FP16: {'enabled' if self.use_fp16 else 'disabled'}")

        if self.device == "cuda":
            self._log_gpu_info()

    def _select_device(self, device: Optional[str]) -> str:
        """Select the best available device.

        Args:
            device: Requested device or None for auto

        Returns:
            Selected device string
        """
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                self.logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
            return device

        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _supports_fp16(self) -> bool:
        """Check if device supports FP16.

        Returns:
            True if FP16 is supported
        """
        if self.device == "cuda":
            # Check GPU compute capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                # FP16 well-supported on compute capability >= 7.0 (Volta+)
                return capability[0] >= 7
        elif self.device == "mps":
            # MPS supports FP16
            return True
        return False

    def _log_gpu_info(self):
        """Log GPU information."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"GPU: {gpu_name}")
            self.logger.info(f"Total VRAM: {total_memory:.2f} GB")

    def get_available_memory_mb(self) -> int:
        """Get available GPU memory in MB.

        Returns:
            Available memory in MB, or -1 if not CUDA
        """
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            free_memory = torch.cuda.mem_get_info()[0]
            return int(free_memory / (1024**2))
        return -1

    def get_used_memory_mb(self) -> int:
        """Get used GPU memory in MB.

        Returns:
            Used memory in MB, or -1 if not CUDA
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024**2))
        return -1

    def estimate_memory_requirement(
        self,
        num_frames: int,
        height: int,
        width: int,
        use_fp16: bool = None,
    ) -> int:
        """Estimate memory requirement for processing.

        Args:
            num_frames: Number of video frames
            height: Frame height
            width: Frame width
            use_fp16: Use FP16 (uses instance setting if None)

        Returns:
            Estimated memory in MB
        """
        if use_fp16 is None:
            use_fp16 = self.use_fp16

        # Rough estimation based on ColorMNet architecture
        # Model: ~2GB
        # Features: ~4 bytes per pixel per channel (or 2 bytes with FP16)
        bytes_per_element = 2 if use_fp16 else 4

        model_memory = 2000  # MB
        feature_memory_per_frame = (height * width * 512 * bytes_per_element) / (1024**2)
        # Keep ~5 frames in memory for temporal processing
        frame_memory = feature_memory_per_frame * min(num_frames, 5)

        # Add 30% overhead for intermediate activations
        total_mb = int((model_memory + frame_memory) * 1.3)

        return total_mb

    def check_memory_available(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tuple[bool, int, int]:
        """Check if enough memory is available.

        Args:
            num_frames: Number of frames
            height: Frame height
            width: Frame width

        Returns:
            Tuple of (is_available, required_mb, available_mb)
        """
        required_mb = self.estimate_memory_requirement(num_frames, height, width)

        if self.device == "cuda":
            available_mb = self.get_available_memory_mb()
            is_available = available_mb >= required_mb
            return is_available, required_mb, available_mb
        else:
            # For CPU/MPS, assume memory is available
            return True, required_mb, -1

    def ensure_memory_available(
        self,
        num_frames: int,
        height: int,
        width: int,
    ):
        """Ensure sufficient memory is available, raise exception if not.

        Args:
            num_frames: Number of frames
            height: Frame height
            width: Frame width

        Raises:
            InsufficientVRAMError: If not enough memory
        """
        is_available, required_mb, available_mb = self.check_memory_available(
            num_frames, height, width
        )

        if not is_available:
            raise InsufficientVRAMError(required_mb, available_mb)

        self.logger.debug(
            f"Memory check passed: {required_mb}MB required, "
            f"{available_mb}MB available"
        )

    def empty_cache(self):
        """Clear GPU memory cache."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")

    def get_autocast_context(self):
        """Get autocast context manager for mixed precision.

        Returns:
            Context manager for automatic mixed precision
        """
        if self.device == "cuda":
            return torch.cuda.amp.autocast(enabled=self.use_fp16)
        else:
            # No-op context for non-CUDA devices
            from contextlib import nullcontext
            return nullcontext()

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to managed device.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on target device
        """
        return tensor.to(self.device)

    def __repr__(self) -> str:
        return f"DeviceManager(device='{self.device}', use_fp16={self.use_fp16})"
