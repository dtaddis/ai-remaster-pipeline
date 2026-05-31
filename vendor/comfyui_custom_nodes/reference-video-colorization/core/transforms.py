"""Color space transformations."""

import torch
import numpy as np
from skimage import color
from typing import Union
from .logger import get_logger


def rgb_to_lab(
    rgb: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert RGB to LAB color space.

    Args:
        rgb: RGB tensor in range [0, 1], shape [..., H, W, 3]
        normalize: Whether to normalize using training-time stats:
                   L -> (L-50)/50, ab -> ab/110 (range ~[-1, 1])

    Returns:
        LAB tensor, shape [..., H, W, 3]
    """
    logger = get_logger()

    # Ensure input is in correct range
    if rgb.min() < -0.01 or rgb.max() > 1.01:
        logger.warning(
            f"RGB tensor out of expected range [0,1]: min={rgb.min():.3f}, max={rgb.max():.3f}"
        )
        rgb = torch.clamp(rgb, 0, 1)

    # Convert to numpy for skimage
    rgb_np = rgb.detach().cpu().numpy()

    # Handle batch dimension
    original_shape = rgb_np.shape
    if rgb_np.ndim > 3:
        # Flatten batch dimensions
        batch_size = np.prod(original_shape[:-3])
        rgb_np = rgb_np.reshape(batch_size, *original_shape[-3:])
    else:
        batch_size = 1
        rgb_np = rgb_np[np.newaxis, ...]

    # Convert each image
    lab_list = []
    for i in range(batch_size):
        lab = color.rgb2lab(rgb_np[i])
        lab_list.append(lab)

    lab_np = np.stack(lab_list, axis=0)

    # Reshape back
    if len(original_shape) > 3:
        lab_np = lab_np.reshape(*original_shape)
    else:
        lab_np = lab_np[0]

    # Convert back to tensor
    lab_tensor = torch.from_numpy(lab_np).to(rgb.device)

    if normalize:
        # Training-time normalization: L/50 and ab/110 to ~[-1, 1]
        lab_tensor[..., 0] = (lab_tensor[..., 0] - 50.0) / 50.0
        lab_tensor[..., 1:] = lab_tensor[..., 1:] / 110.0

    return lab_tensor


def lab_to_rgb(
    lab: torch.Tensor,
    denormalize: bool = True,
    normalized: bool = False,
) -> torch.Tensor:
    """Convert LAB to RGB color space.

    Args:
        lab: LAB tensor, shape [..., H, W, 3]
        denormalize: Legacy flag; kept for compatibility (adds 50 to L).
        normalized: If True, assumes training-time normalization (L: (x-50)/50, ab: /110).

    Returns:
        RGB tensor in range [0, 1], shape [..., H, W, 3]
    """
    lab = lab.clone()

    if normalized:
        # Inverse of training-time normalization
        lab[..., 0] = lab[..., 0] * 50.0 + 50.0
        lab[..., 1:] = lab[..., 1:] * 110.0
    elif denormalize:
        # Legacy path: L was in [-50, 50]
        lab[..., 0] = lab[..., 0] + 50.0

    # Convert to numpy
    lab_np = lab.detach().cpu().numpy()

    # Handle batch dimension
    original_shape = lab_np.shape
    if lab_np.ndim > 3:
        batch_size = np.prod(original_shape[:-3])
        lab_np = lab_np.reshape(batch_size, *original_shape[-3:])
    else:
        batch_size = 1
        lab_np = lab_np[np.newaxis, ...]

    # Convert each image
    rgb_list = []
    for i in range(batch_size):
        rgb = color.lab2rgb(lab_np[i])
        rgb_list.append(rgb)

    rgb_np = np.stack(rgb_list, axis=0)

    # Reshape back
    if len(original_shape) > 3:
        rgb_np = rgb_np.reshape(*original_shape)
    else:
        rgb_np = rgb_np[0]

    # Convert back to tensor and clamp
    rgb_tensor = torch.from_numpy(rgb_np).to(lab.device)
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)

    return rgb_tensor.float()


def tensor_to_pil(tensor: torch.Tensor):
    """Convert tensor to PIL Image.

    Args:
        tensor: Image tensor [H, W, C] or [1, H, W, C] in range [0, 1]

    Returns:
        PIL Image
    """
    from PIL import Image

    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Ensure HWC format
    if tensor.shape[0] in (1, 3):  # CHW format
        tensor = tensor.permute(1, 2, 0)

    # Convert to numpy and scale to [0, 255]
    arr = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode='L')
    else:
        return Image.fromarray(arr, mode='RGB')


def pil_to_tensor(image, device: str = 'cpu') -> torch.Tensor:
    """Convert PIL Image to tensor.

    Args:
        image: PIL Image
        device: Target device

    Returns:
        Tensor [H, W, C] in range [0, 1]
    """
    arr = np.array(image).astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    tensor = torch.from_numpy(arr).to(device)
    return tensor
