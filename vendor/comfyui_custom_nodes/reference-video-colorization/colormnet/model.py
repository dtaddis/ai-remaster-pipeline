"""ColorMNet model wrapper with modern error handling."""

import torch
from pathlib import Path
from typing import Optional

from core.logger import get_logger
from core.exceptions import ModelNotFoundError
from core.device import DeviceManager
from .config import ColorMNetConfig

# Import ColorMNet network from embedded source
# Prefer the vendored package name but keep a fallback for older paths.
try:
    from colormnet_model.network import ColorMNet
except ImportError:
    # Older revisions referenced `model.network`; alias is registered in __init__, but double-protect here.
    from model.network import ColorMNet


class ColorMNetModel:
    """Wrapper around ColorMNet network with modern error handling and device management.

    This class provides a clean interface to the ColorMNet network with:
    - Automatic device management
    - Model lazy loading
    - Error handling
    - Memory management
    """

    def __init__(
        self,
        config: ColorMNetConfig,
        device_manager: Optional[DeviceManager] = None,
    ):
        """Initialize ColorMNet model wrapper.

        Args:
            config: ColorMNet configuration
            device_manager: Device manager (creates one if None)
        """
        self.logger = get_logger()
        self.config = config
        self.device_manager = device_manager or DeviceManager(
            device=config.device,
            use_fp16=config.use_fp16
        )

        self._model: Optional[ColorMNet] = None
        self._model_loaded = False

        self.logger.info("ColorMNetModel initialized")

    def load_model(self, model_path: Optional[Path] = None) -> ColorMNet:
        """Load the ColorMNet model.

        Args:
            model_path: Path to model checkpoint (uses config if None)

        Returns:
            Loaded ColorMNet network

        Raises:
            ModelNotFoundError: If checkpoint file doesn't exist
        """
        if self._model_loaded and self._model is not None:
            return self._model

        # Use provided path or config path
        checkpoint_path = model_path or self.config.model_path

        if checkpoint_path is None:
            raise ModelNotFoundError("No model path specified")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ModelNotFoundError(str(checkpoint_path))

        self.logger.info(f"Loading ColorMNet from: {checkpoint_path}")

        try:
            # Create model with config
            config_dict = self.config.to_dict()
            self._model = ColorMNet(config_dict, str(checkpoint_path))

            # Move to device
            self._model = self._model.to(self.device_manager.device)
            self._model.eval()

            # Load weights
            model_weights = torch.load(
                checkpoint_path,
                map_location=self.device_manager.device
            )
            self._model.load_weights(model_weights, init_as_zero_if_needed=True)

            self._model_loaded = True
            self.logger.info("ColorMNet loaded successfully")

            # Log memory usage
            if self.device_manager.device == "cuda":
                mem_mb = self.device_manager.get_used_memory_mb()
                self.logger.info(f"Model memory: {mem_mb}MB")

            return self._model

        except Exception as e:
            self.logger.error(f"Failed to load ColorMNet: {e}")
            raise

    @property
    def model(self) -> ColorMNet:
        """Get the loaded model (loads if not already loaded).

        Returns:
            ColorMNet network

        Raises:
            ModelNotFoundError: If model can't be loaded
        """
        if not self._model_loaded or self._model is None:
            self.load_model()
        return self._model

    def encode_key(self, frame: torch.Tensor, need_sk: bool = True, need_ek: bool = True):
        """Encode frame keys using DINOv2 features.

        Args:
            frame: Input frame tensor
            need_sk: Whether to compute shrinkage
            need_ek: Whether to compute selection

        Returns:
            Tuple of (key, shrinkage, selection, f16, f8, f4)
        """
        with torch.no_grad():
            return self.model.encode_key(frame, need_sk, need_ek)

    def encode_value(
        self,
        frame: torch.Tensor,
        image_feat_f16: torch.Tensor,
        h16: torch.Tensor,
        masks: torch.Tensor,
        is_deep_update: bool = True,
    ):
        """Encode frame values.

        Args:
            frame: Input frame
            image_feat_f16: Image features at scale 16
            h16: Hidden state at scale 16
            masks: Color masks (ab channels)
            is_deep_update: Whether to perform deep update

        Returns:
            Tuple of (g16, h16)
        """
        with torch.no_grad():
            return self.model.encode_value(frame, image_feat_f16, h16, masks, is_deep_update)

    def segment(
        self,
        multi_scale_features,
        memory_readout,
        hidden_state,
        selector=None,
        h_out=True,
        strip_bg=True,
    ):
        """Segment and decode to color predictions.

        Args:
            multi_scale_features: Features at multiple scales
            memory_readout: Memory bank readout
            hidden_state: Current hidden state
            selector: Optional selector
            h_out: Whether to output hidden state
            strip_bg: Whether to strip background

        Returns:
            Decoded color prediction
        """
        with torch.no_grad():
            return self.model.segment(
                multi_scale_features,
                memory_readout,
                hidden_state,
                selector,
                h_out,
                strip_bg,
            )

    def to(self, device: str):
        """Move model to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        if self._model is not None:
            self._model = self._model.to(device)
            self.logger.info(f"Model moved to {device}")
        return self

    def eval(self):
        """Set model to evaluation mode.

        Returns:
            Self for chaining
        """
        if self._model is not None:
            self._model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"ColorMNetModel(loaded={self._model_loaded}, "
            f"device={self.device_manager.device}, "
            f"fp16={self.device_manager.use_fp16})"
        )
