"""Input validation utilities."""

import torch
from typing import Tuple, Optional
from .exceptions import ValidationError
from .logger import get_logger


def validate_input(
    frames: torch.Tensor,
    reference: torch.Tensor,
    min_resolution: int = 64,
    max_resolution: int = 2048,
) -> Tuple[int, int, int, int]:
    """Validate input tensors and return dimensions.

    Args:
        frames: Video frames tensor [B, H, W, C] or [H, W, C]
        reference: Reference image tensor [H, W, C]
        min_resolution: Minimum allowed resolution
        max_resolution: Maximum allowed resolution

    Returns:
        Tuple of (num_frames, height, width, channels)

    Raises:
        ValidationError: If validation fails
    """
    logger = get_logger()

    # Check reference image
    if not isinstance(reference, torch.Tensor):
        raise ValidationError(
            f"Reference must be a torch.Tensor, got {type(reference)}"
        )

    if reference.dim() not in (3, 4):
        raise ValidationError(
            f"Reference must be 3D [H,W,C] or 4D [1,H,W,C], got shape {reference.shape}"
        )

    if reference.dim() == 4:
        if reference.shape[0] != 1:
            raise ValidationError(
                f"Reference batch size must be 1, got {reference.shape[0]}"
            )
        reference = reference[0]

    ref_h, ref_w, ref_c = reference.shape
    if ref_c != 3:
        raise ValidationError(
            f"Reference must have 3 channels (RGB), got {ref_c}"
        )

    # Check frames
    if not isinstance(frames, torch.Tensor):
        raise ValidationError(
            f"Frames must be a torch.Tensor, got {type(frames)}"
        )

    if frames.dim() not in (3, 4):
        raise ValidationError(
            f"Frames must be 3D [H,W,C] or 4D [B,H,W,C], got shape {frames.shape}"
        )

    if frames.dim() == 3:
        frames = frames.unsqueeze(0)

    num_frames, height, width, channels = frames.shape

    if channels != 3:
        raise ValidationError(
            f"Frames must have 3 channels (RGB), got {channels}"
        )

    # Check resolution constraints
    if height < min_resolution or width < min_resolution:
        raise ValidationError(
            f"Resolution too small: {height}x{width}. Minimum is {min_resolution}x{min_resolution}"
        )

    if height > max_resolution or width > max_resolution:
        raise ValidationError(
            f"Resolution too large: {height}x{width}. Maximum is {max_resolution}x{max_resolution}"
        )

    # Check reference and frames match dimensions
    if (ref_h, ref_w) != (height, width):
        logger.warning(
            f"Reference size ({ref_h}x{ref_w}) doesn't match frame size ({height}x{width}). "
            f"Reference will be resized."
        )

    # Check resolution is reasonable for memory
    total_pixels = height * width * num_frames
    if total_pixels > 500_000_000:  # 500 megapixels
        logger.warning(
            f"Very large input: {num_frames} frames at {height}x{width}. "
            f"This may require significant VRAM."
        )

    logger.debug(
        f"Validation passed: {num_frames} frames, {height}x{width}, {channels} channels"
    )

    return num_frames, height, width, channels


def adjust_resolution(
    height: int,
    width: int,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    multiple: int = 32,
    min_size: int = 64,
) -> Tuple[int, int]:
    """Adjust resolution to meet constraints.

    Args:
        height: Input height
        width: Input width
        target_height: Target height (None to keep aspect ratio)
        target_width: Target width (None to keep aspect ratio)
        multiple: Resolution must be multiple of this
        min_size: Minimum dimension size

    Returns:
        Tuple of (adjusted_height, adjusted_width)
    """
    # Use target dimensions if provided
    if target_height is not None:
        height = target_height
    if target_width is not None:
        width = target_width

    # Ensure minimum size
    height = max(height, min_size)
    width = max(width, min_size)

    # Round to multiple
    def round_to_multiple(value: int, mult: int) -> int:
        remainder = value % mult
        if remainder == 0:
            return value

        # Round to nearest multiple
        down = value - remainder
        up = down + mult

        # Choose closer value, but prefer down if at minimum
        if up - value < value - down:
            return up if up >= min_size else down
        else:
            return down if down >= min_size else up

    final_height = round_to_multiple(height, multiple)
    final_width = round_to_multiple(width, multiple)

    return final_height, final_width


def validate_color_tensor(
    tensor: torch.Tensor,
    expected_range: Tuple[float, float] = (0.0, 1.0),
    name: str = "tensor",
) -> None:
    """Validate color tensor values are in expected range.

    Args:
        tensor: Color tensor to validate
        expected_range: Expected (min, max) values
        name: Tensor name for error messages

    Raises:
        ValidationError: If values are out of range
    """
    min_val, max_val = expected_range
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()

    if actual_min < min_val - 0.01 or actual_max > max_val + 0.01:
        raise ValidationError(
            f"{name} values out of range. Expected [{min_val}, {max_val}], "
            f"got [{actual_min:.3f}, {actual_max:.3f}]"
        )
