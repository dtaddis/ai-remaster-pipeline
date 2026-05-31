"""ColorMNet inference pipeline."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Callable, Optional
from pathlib import Path

from core.logger import get_logger
from core.device import DeviceManager
from core.validation import validate_input, adjust_resolution
from core.transforms import rgb_to_lab, lab_to_rgb
from core.exceptions import ValidationError

from .model import ColorMNetModel
from .config import ColorMNetConfig

# Import ColorMNet inference core from embedded source
from colormnet_inference.inference_core import InferenceCore


class ColorMNetInference:
    """High-level inference pipeline for video colorization.

    This class orchestrates the full colorization process:
    1. Input validation
    2. Reference image encoding
    3. Frame-by-frame processing with memory management
    4. Color space conversions
    5. Progress tracking
    """

    def __init__(
        self,
        model_path: Path,
        device_manager: Optional[DeviceManager] = None,
        config: Optional[ColorMNetConfig] = None,
    ):
        """Initialize inference pipeline.

        Args:
            model_path: Path to model checkpoint
            device_manager: Device manager (creates one if None)
            config: ColorMNet configuration (creates default if None)
        """
        self.logger = get_logger()
        self.device_manager = device_manager or DeviceManager()

        # Create config
        if config is None:
            config = ColorMNetConfig.default(model_path=model_path)
        else:
            config.model_path = model_path

        # Apply device to config
        config.device = self.device_manager.device

        # Create model wrapper
        self.model_wrapper = ColorMNetModel(config, self.device_manager)
        self.config = config

        # Inference processor (created per video)
        self.processor: Optional[InferenceCore] = None

    def _prepare_frame(
        self,
        frame: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Prepare frame for processing.

        Args:
            frame: RGB frame [H, W, 3] in range [0, 1]
            target_size: Optional target (H, W)

        Returns:
            Normalized LAB frame [3, H, W] on device (L ~ [-1,1], ab ~ [-1,1])
        """
        # Convert to LAB
        lab = rgb_to_lab(frame, normalize=True)

        # Resize if needed
        if target_size is not None:
            h, w = target_size
            if (lab.shape[0], lab.shape[1]) != (h, w):
                # HWC -> CHW for interpolate
                lab = lab.permute(2, 0, 1).unsqueeze(0)
                lab = F.interpolate(lab, size=(h, w), mode='bilinear', align_corners=False)
                lab = lab.squeeze(0).permute(1, 2, 0)

        # HWC -> CHW and move to device
        lab = lab.permute(2, 0, 1)
        lab = self.device_manager.to_device(lab)

        return lab

    def _prepare_reference(
        self,
        reference: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare reference image.

        Args:
            reference: RGB reference [H, W, 3] in range [0, 1]
            target_size: Target (H, W)

        Returns:
            Tuple of (L channel [1, 1, H, W], ab channels [1, 2, H, W])
        """
        # Convert and resize
        lab = self._prepare_frame(reference, target_size)

        # Split L and ab
        L = lab[0:1].unsqueeze(0)  # [1, 1, H, W]
        ab = lab[1:3].unsqueeze(0)  # [1, 2, H, W]

        return L, ab

    def colorize_video(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """Colorize video frames using reference image.

        Args:
            frames: Video frames [N, H, W, 3] in range [0, 1]
            reference: Reference image [H, W, 3] in range [0, 1]
            target_height: Target height (None to keep original)
            target_width: Target width (None to keep original)
            progress_callback: Optional callback(current, total)

        Returns:
            Colorized frames [N, H, W, 3] in range [0, 1]

        Raises:
            ValidationError: If input validation fails
        """
        self.logger.info("Starting video colorization")

        # Validate inputs
        num_frames, height, width, channels = validate_input(frames, reference)
        self.logger.info(f"Input: {num_frames} frames at {height}x{width}")

        # Adjust resolution
        if target_height is None:
            target_height = height
        if target_width is None:
            target_width = width

        proc_height, proc_width = adjust_resolution(
            target_height, target_width,
            multiple=32, min_size=64
        )

        if (proc_height, proc_width) != (target_height, target_width):
            self.logger.info(
                f"Resolution adjusted: {target_height}x{target_width} -> "
                f"{proc_height}x{proc_width}"
            )

        # Check memory
        self.device_manager.ensure_memory_available(num_frames, proc_height, proc_width)

        # Load model
        model = self.model_wrapper.model

        # Create inference processor
        self.processor = InferenceCore(model, config=self.config.to_dict())

        # Prepare reference LAB (normalized to training stats)
        ref_lab = self._prepare_frame(reference, (proc_height, proc_width))
        ref_L = ref_lab[0:1].unsqueeze(0)  # [1, 1, H, W]
        ref_ab = ref_lab[1:3].unsqueeze(0)  # [1, 2, H, W] normalized
        ref_lll = ref_L[0].repeat(3, 1, 1)  # [3, H, W] normalized L replicated

        # Process frames
        colorized_frames = []

        with self.device_manager.get_autocast_context():
            for frame_idx in range(num_frames):
                # Get frame
                frame = frames[frame_idx]

                # Prepare frame
                frame_lab = self._prepare_frame(frame, (proc_height, proc_width))
                frame_L = frame_lab[0:1].unsqueeze(0)  # [1, 1, H, W]
                frame_lll = frame_L[0].repeat(3, 1, 1)  # [3, H, W] normalized L replicated

                # First frame: initialize with reference
                if frame_idx == 0:
                    # Pass only normalized L replicated (matches training pipeline)
                    msk_lll = ref_lll  # [3, H, W] normalized L
                    msk_ab = ref_ab.squeeze(0)  # [1, 2, H, W] -> [2, H, W]
                    labels = [1, 2]  # Two color channels
                    self.processor.set_all_labels(labels)

                    # Process first frame with reference exemplar
                    prob = self.processor.step_AnyExemplar(
                        frame_lll,
                        msk_lll=msk_lll,
                        msk_ab=msk_ab,
                        valid_labels=labels,
                        end=False,
                        flag_FirstframeIsExemplar=False
                    )
                else:
                    # Subsequent frames: use memory
                    prob = self.processor.step_AnyExemplar(
                        frame_lll,
                        msk_lll=None,
                        msk_ab=None,
                        valid_labels=None,
                        end=(frame_idx == num_frames - 1),
                        flag_FirstframeIsExemplar=False
                    )

                # Prob is [2, H, W] - predicted ab channels in normalized range [-1, 1]

                # Debug: Check if prob contains color information
                if frame_idx == 0:
                    self.logger.info(f"First frame - prob (normalized) shape: {prob.shape}")
                    self.logger.info(f"First frame - prob (normalized) min: {prob.min():.3f}, max: {prob.max():.3f}, mean: {prob.mean():.3f}")
                    self.logger.info(f"Reference ab (normalized) - min: {ref_ab.min():.3f}, max: {ref_ab.max():.3f}, mean: {ref_ab.mean():.3f}")

                # Combine with L channel
                colorized_lab = torch.cat([frame_L[0], prob], dim=0)  # normalized LAB [3, H, W]

                # Convert back to RGB
                colorized_rgb = lab_to_rgb(
                    colorized_lab.permute(1, 2, 0),  # CHW -> HWC
                    normalized=True
                )

                # Resize to original resolution if needed
                if (proc_height, proc_width) != (height, width):
                    colorized_rgb = colorized_rgb.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
                    colorized_rgb = F.interpolate(
                        colorized_rgb,
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    )
                    colorized_rgb = colorized_rgb.squeeze(0).permute(1, 2, 0)  # NCHW -> HWC

                colorized_frames.append(colorized_rgb.cpu())

                # Progress callback
                if progress_callback is not None:
                    progress_callback(frame_idx + 1, num_frames)

                # Log progress
                if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                    self.logger.info(f"Processed {frame_idx + 1}/{num_frames} frames")

                    if self.device_manager.device == "cuda":
                        mem_mb = self.device_manager.get_used_memory_mb()
                        self.logger.debug(f"VRAM usage: {mem_mb}MB")

        # Stack frames
        result = torch.stack(colorized_frames, dim=0)

        self.logger.info(f"Colorization complete: {num_frames} frames")

        # Clear memory
        self.device_manager.empty_cache()

        return result

    def colorize_image(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Colorize a single image using reference.

        Args:
            image: Grayscale image [H, W, 3] in range [0, 1]
            reference: Reference image [H, W, 3] in range [0, 1]
            target_height: Target height
            target_width: Target width

        Returns:
            Colorized image [H, W, 3] in range [0, 1]
        """
        # Add batch dimension
        image_batch = image.unsqueeze(0)

        # Colorize as video
        result_batch = self.colorize_video(
            image_batch,
            reference,
            target_height,
            target_width,
        )

        # Remove batch dimension
        return result_batch[0]

    def __repr__(self) -> str:
        return (
            f"ColorMNetInference(device={self.device_manager.device}, "
            f"model_loaded={self.model_wrapper._model_loaded})"
        )
