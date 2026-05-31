"""
DeepExemplarColorizationNodes.py

This module provides two ComfyUI nodes for Deep Exemplar-based Colorization:
  - DeepExColorImageNode: Colorizes a grayscale image using a reference color image.
  - DeepExColorVideoNode: Colorizes a sequence of grayscale video frames using a reference color image.
  
Both nodes follow the test.py logic from the base Deep Exemplar project.
"""

import sys
import subprocess
import importlib
import cv2

# Check for cv2.ximgproc.createFastGlobalSmootherFilter and attempt installation if needed.
if not hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter'):
    print("cv2.ximgproc.createFastGlobalSmootherFilter not found. Attempting to install opencv-contrib-python...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "opencv-contrib-python"])
        importlib.reload(cv2)
        if hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter'):
            print("Installation successful and function is now available.")
        else:
            print("Installation attempted, but function is still unavailable. Please install or upgrade opencv-contrib-python (e.g., 'pip install --upgrade opencv-contrib-python') and restart ComfyUI.")
    except Exception as e:
        print(f"Failed to install opencv-contrib-python: {e}\nPlease install opencv-contrib-python manually and restart ComfyUI.")

import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as Tlib
from tqdm import tqdm as console_tqdm

# Import ComfyUI's internal ProgressBar for GUI progress updates.
from comfy.utils import ProgressBar

from .utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor
from .utils.util import batch_lab2rgb_transpose_mc, uncenter_l, tensor_lab2rgb
from .models.FrameColor import frame_colorization
from .models import NonlocalNet  # Import NonlocalNet module for SageAttention flag

MODELS_LOADED = False
NONLOCAL_NET = None
COLOR_NET = None
VGG_NET = None

# Cache for CUDA graph compatibility check
_CUDA_GRAPHS_COMPATIBLE = None

def check_cuda_graphs_compatible():
    """
    Check if CUDA graphs are compatible with the current CUDA memory allocator.
    Returns True if compatible, False if using cudaMallocAsync or incompatible allocator.
    """
    global _CUDA_GRAPHS_COMPATIBLE

    if _CUDA_GRAPHS_COMPATIBLE is not None:
        return _CUDA_GRAPHS_COMPATIBLE

    if not torch.cuda.is_available():
        _CUDA_GRAPHS_COMPATIBLE = False
        return False

    try:
        # Check if using cudaMallocAsync allocator
        # cudaMallocAsync doesn't support checkPoolLiveAllocations needed by CUDA graphs
        allocator_backend = torch.cuda.get_allocator_backend()
        if allocator_backend == 'cudaMallocAsync':
            print("[DeepExemplar] Detected cudaMallocAsync allocator - CUDA graphs will be disabled")
            _CUDA_GRAPHS_COMPATIBLE = False
            return False
    except (AttributeError, RuntimeError):
        # If we can't check allocator, assume compatible
        pass

    # Default to compatible
    _CUDA_GRAPHS_COMPATIBLE = True
    return True

def get_torch_compile_kwargs():
    """
    Get appropriate kwargs for torch.compile based on system compatibility.
    Returns dict with mode and optionally disable_cudagraphs.
    """
    if check_cuda_graphs_compatible():
        # CUDA graphs supported - use reduce-overhead for best performance
        return {'mode': 'reduce-overhead'}
    else:
        # CUDA graphs not supported - use default mode
        # Note: disable_cudagraphs parameter was added in PyTorch 2.1+
        # For compatibility with older PyTorch, just use mode='default'
        return {'mode': 'default'}

def download_vgg19_model(vgg_path):
    """Download VGG19 conv model if not exists."""
    if os.path.exists(vgg_path):
        return True

    print("[DeepExemplar] VGG19 model not found, downloading...")

    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(vgg_path)
    os.makedirs(data_dir, exist_ok=True)

    # VGG19 conv weights URL (standard PyTorch VGG19 trained on ImageNet)
    vgg_url = "https://github.com/yyang181/colormnet/releases/download/v0.1/vgg19_conv.pth"

    try:
        import urllib.request
        print(f"[DeepExemplar] Downloading from {vgg_url}")
        print(f"[DeepExemplar] Saving to {vgg_path}")

        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if count * block_size < total_size:
                print(f"\r[DeepExemplar] Downloading: {percent}%", end='')
            else:
                print(f"\r[DeepExemplar] Download complete!")

        urllib.request.urlretrieve(vgg_url, vgg_path, reporthook)

        if os.path.exists(vgg_path):
            file_size_mb = os.path.getsize(vgg_path) / (1024 * 1024)
            print(f"[DeepExemplar] ✓ VGG19 model downloaded ({file_size_mb:.1f}MB)")
            return True
        else:
            print("[DeepExemplar] ✗ Download failed")
            return False

    except Exception as e:
        print(f"[DeepExemplar] ✗ Error downloading VGG19 model: {e}")
        print("[DeepExemplar] Please download manually from:")
        print(f"[DeepExemplar]   {vgg_url}")
        print(f"[DeepExemplar] And place at: {vgg_path}")
        return False

def download_deepexemplar_checkpoints(nonlocal_ckpt_path, color_ckpt_path):
    """Download Deep Exemplar checkpoint files if not exists."""
    # Check if both checkpoints exist
    if os.path.exists(nonlocal_ckpt_path) and os.path.exists(color_ckpt_path):
        return True

    print("[DeepExemplar] Checkpoint files not found, downloading...")

    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.dirname(nonlocal_ckpt_path)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Deep Exemplar checkpoint URL
    checkpoint_url = "https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/releases/download/v1.0/colorization_checkpoint.zip"

    try:
        import urllib.request
        import zipfile
        import tempfile

        print(f"[DeepExemplar] Downloading from {checkpoint_url}")

        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_path = tmp_file.name

            # Download with progress
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    if count * block_size < total_size:
                        print(f"\r[DeepExemplar] Downloading: {percent}%", end='')
                    else:
                        print(f"\r[DeepExemplar] Download complete!")
                else:
                    # Size unknown, just show blocks downloaded
                    mb_downloaded = (count * block_size) / (1024 * 1024)
                    print(f"\r[DeepExemplar] Downloaded: {mb_downloaded:.1f}MB", end='')

            urllib.request.urlretrieve(checkpoint_url, tmp_path, reporthook)
            print()  # New line after progress

        # Extract the zip file
        print(f"[DeepExemplar] Extracting checkpoints...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            # List contents
            zip_contents = zip_ref.namelist()
            print(f"[DeepExemplar] Zip contains {len(zip_contents)} files")

            # Extract all files to script_dir (zip already has checkpoints/ prefix)
            zip_ref.extractall(script_dir)

        # Clean up temp file
        os.unlink(tmp_path)

        # Verify the checkpoints exist
        if os.path.exists(nonlocal_ckpt_path) and os.path.exists(color_ckpt_path):
            nonlocal_size_mb = os.path.getsize(nonlocal_ckpt_path) / (1024 * 1024)
            color_size_mb = os.path.getsize(color_ckpt_path) / (1024 * 1024)
            print(f"[DeepExemplar] ✓ nonlocal_net checkpoint extracted ({nonlocal_size_mb:.1f}MB)")
            print(f"[DeepExemplar] ✓ colornet checkpoint extracted ({color_size_mb:.1f}MB)")
            return True
        else:
            print("[DeepExemplar] ✗ Extraction failed - checkpoint files not found")
            print(f"[DeepExemplar] Expected files:")
            print(f"[DeepExemplar]   {nonlocal_ckpt_path}")
            print(f"[DeepExemplar]   {color_ckpt_path}")
            return False

    except Exception as e:
        print(f"[DeepExemplar] ✗ Error downloading/extracting checkpoints: {e}")
        print("[DeepExemplar] Please download manually from:")
        print(f"[DeepExemplar]   {checkpoint_url}")
        print(f"[DeepExemplar] Extract to: {checkpoints_base}")
        return False

def load_models_if_needed():
    global MODELS_LOADED, NONLOCAL_NET, COLOR_NET, VGG_NET
    if MODELS_LOADED:
        return
    print("[DeepExemplar] Loading model checkpoints...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nonlocal_ckpt = os.path.join(script_dir, "checkpoints", "video_moredata_l1", "nonlocal_net_iter_76000.pth")
    color_ckpt = os.path.join(script_dir, "checkpoints", "video_moredata_l1", "colornet_iter_76000.pth")
    vgg_ckpt = os.path.join(script_dir, "data", "vgg19_conv.pth")

    # Auto-download Deep Exemplar checkpoints if needed
    if not download_deepexemplar_checkpoints(nonlocal_ckpt, color_ckpt):
        raise FileNotFoundError(f"Deep Exemplar checkpoints not found and download failed: {nonlocal_ckpt}, {color_ckpt}")

    # Auto-download VGG19 if needed
    if not download_vgg19_model(vgg_ckpt):
        raise FileNotFoundError(f"VGG19 model not found and download failed: {vgg_ckpt}")
    from .models.NonlocalNet import WarpNet, VGG19_pytorch
    from .models.ColorVidNet import ColorVidNet
    NONLOCAL_NET = WarpNet(1).cuda().eval()
    COLOR_NET = ColorVidNet(7).cuda().eval()
    VGG_NET = VGG19_pytorch().cuda().eval()
    NONLOCAL_NET.load_state_dict(torch.load(nonlocal_ckpt, map_location="cuda"))
    COLOR_NET.load_state_dict(torch.load(color_ckpt, map_location="cuda"))
    VGG_NET.load_state_dict(torch.load(vgg_ckpt, map_location="cuda"))
    MODELS_LOADED = True
    print("[DeepExemplar] Models loaded successfully.")

def adjust_target_size(input_h: int, input_w: int, min_h: int = 64, min_w: int = 64, downsample_multiple: int = 32) -> (int, int):
    final_h = max(input_h, min_h)
    final_w = max(input_w, min_w)
    remainder_h = final_h % downsample_multiple
    if remainder_h != 0:
        down_dist = remainder_h
        up_dist = downsample_multiple - remainder_h
        if up_dist < down_dist:
            final_h += up_dist
        else:
            if final_h - down_dist >= min_h:
                final_h -= down_dist
            else:
                final_h += up_dist
    remainder_w = final_w % downsample_multiple
    if remainder_w != 0:
        down_dist = remainder_w
        up_dist = downsample_multiple - remainder_w
        if up_dist < down_dist:
            final_w += up_dist
        else:
            if final_w - down_dist >= min_w:
                final_w -= down_dist
            else:
                final_w += up_dist
    return final_h, final_w

def build_test_py_transform(image_size=(432,768)):
    return Tlib.Compose([
        CenterPad(image_size),
        Tlib.CenterCrop(image_size),
        RGB2Lab(),
        ToTensor(),
        Normalize(),
    ])

def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
    if tensor_img.dim() == 4 and tensor_img.size(0) == 1:
        tensor_img = tensor_img.squeeze(0)
    if tensor_img.dim() == 3 and tensor_img.shape[-1] in (1, 3):
        tensor_img = tensor_img.permute(2, 0, 1)
    tensor_img = tensor_img.detach().cpu().clamp(0,1)
    np_img = (tensor_img.numpy()*255).astype(np.uint8)
    np_img = np.transpose(np_img, (1,2,0))
    if np_img.shape[2] == 1:
        return Image.fromarray(np_img[:,:,0], mode="L")
    else:
        return Image.fromarray(np_img, mode="RGB")

class DeepExColorImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_to_colorize": ("IMAGE", {"tooltip": "Grayscale or color image to be colorized"}),
                "reference_image": ("IMAGE", {"tooltip": "Color reference image that provides the color palette"}),
                "target_width": ("INT", {
                    "default": 768,
                    "min": 16,
                    "max": 4096,
                    "tooltip": "Output width (will be adjusted to nearest multiple of 32)"
                }),
                "target_height": ("INT", {
                    "default": 432,
                    "min": 16,
                    "max": 4096,
                    "tooltip": "Output height (will be adjusted to nearest multiple of 32)"
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable torch.compile optimization for 10-25% speedup (may increase first-run compilation time)"
                }),
                "use_sage_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable SageAttention for faster attention computation (requires sageattention package)"
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("colorized_image", "performance_report")
    FUNCTION = "colorize_image"
    CATEGORY = "DeepExemplar/Image"

    def colorize_image(self, image_to_colorize: torch.Tensor, reference_image: torch.Tensor,
                         target_width: int, target_height: int,
                         use_torch_compile: bool = False, use_sage_attention: bool = False) -> (list,):
        # Start timing
        start_time = time.time()

        load_models_if_needed()

        # Declare globals first
        global NONLOCAL_NET, COLOR_NET, VGG_NET

        # DeepExemplar only uses VGG19 (trained baseline)
        encoder_net = VGG_NET

        # Apply optimizations to models
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                # Get optimal compile settings based on system compatibility
                compile_kwargs = get_torch_compile_kwargs()

                if not hasattr(VGG_NET, '_compiled'):
                    VGG_NET = torch.compile(VGG_NET, **compile_kwargs)
                    VGG_NET._compiled = True
                if not hasattr(NONLOCAL_NET, '_compiled'):
                    NONLOCAL_NET = torch.compile(NONLOCAL_NET, **compile_kwargs)
                    NONLOCAL_NET._compiled = True
                if not hasattr(COLOR_NET, '_compiled'):
                    COLOR_NET = torch.compile(COLOR_NET, **compile_kwargs)
                    COLOR_NET._compiled = True

                mode_str = f"mode={compile_kwargs['mode']}"
                if 'disable_cudagraphs' in compile_kwargs:
                    mode_str += ", CUDA graphs disabled"
                print(f"[DeepExColorImageNode] ✓ torch.compile enabled ({mode_str})")
            except Exception as e:
                print(f"[DeepExColorImageNode] torch.compile failed: {e}")

        # Set SageAttention flag for NonlocalNet
        if use_sage_attention:
            NonlocalNet.USE_SAGE_ATTENTION = True
            print("[DeepExColorImageNode] ✓ SageAttention enabled")
        else:
            NonlocalNet.USE_SAGE_ATTENTION = False

        # Adjust target size (VGG19 requires dimensions divisible by 32)
        min_multiple = 32

        final_h, final_w = adjust_target_size(target_height, target_width, 64, 64, min_multiple)
        print(f"[DeepExColorImageNode] Adjusted size: ({target_width}, {target_height}) -> ({final_w}, {final_h})")
        transform = build_test_py_transform((final_h, final_w))
        src_pil = tensor_to_pil(image_to_colorize)
        ref_pil = tensor_to_pil(reference_image)
        src_lab_full = transform(src_pil).unsqueeze(0).cuda()
        ref_lab_full = transform(ref_pil).unsqueeze(0).cuda()
        src_half = F.interpolate(src_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        ref_half = F.interpolate(ref_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        I_ref_l = ref_half[:, 0:1, :, :]
        I_ref_ab = ref_half[:, 1:3, :, :]
        I_ref_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_ref_l), I_ref_ab), dim=1)).to(ref_half.device)
        with torch.no_grad():
            features_B = encoder_net(I_ref_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            I_ab_predict, _, _ = frame_colorization(
                src_half, ref_half, torch.zeros_like(src_half),
                features_B, encoder_net, NONLOCAL_NET, COLOR_NET,
                feature_noise=0, temperature=1e-10
            )
        ab_up = F.interpolate(I_ab_predict, scale_factor=2.0, mode="bilinear", align_corners=False) * 1.25

        # Convert to RGB for post-processing
        out_np = batch_lab2rgb_transpose_mc(src_lab_full[:, 0:1, :, :], ab_up)
        out_tensor = torch.from_numpy(out_np).float().div(255.0)

        # Ensure out_tensor has batch dimension
        if out_tensor.dim() == 3:
            out_tensor = out_tensor.unsqueeze(0)

        # Calculate timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Generate performance report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = (
            f"Deep Exemplar Image Colorization Report\n"
            f"{'=' * 50}\n"
            f"Date/Time: {timestamp}\n"
            f"Resolution: {final_w}x{final_h}\n"
            f"Total Time: {elapsed_time:.3f} seconds\n"
        )
        report += (
            f"Torch Compile: {use_torch_compile}\n"
            f"SageAttention: {use_sage_attention}\n"
            f"{'=' * 50}"
        )

        print(f"[DeepExColorImageNode] Colorization complete in {elapsed_time:.3f}s")

        return (out_tensor, report)

class DeepExColorVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {"tooltip": "Batch of video frames to be colorized"}),
                "reference_image": ("IMAGE", {"tooltip": "Color reference image that provides the color palette"}),
                "frame_propagate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable temporal propagation: use previous frame's colorization to inform current frame (improves consistency)"
                }),
                "use_half_resolution": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process at half resolution for speed, then upscale (2x faster with minimal quality loss)"
                }),
                "target_width": ("INT", {
                    "default": 768,
                    "min": 16,
                    "max": 4096,
                    "tooltip": "Output width (will be adjusted to nearest multiple of 32)"
                }),
                "target_height": ("INT", {
                    "default": 432,
                    "min": 16,
                    "max": 4096,
                    "tooltip": "Output height (will be adjusted to nearest multiple of 32)"
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable torch.compile optimization for 10-25% speedup (may increase first-run compilation time)"
                }),
                "use_sage_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable SageAttention for faster attention computation (requires sageattention package)"
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("colorized_frames", "performance_report")
    FUNCTION = "colorize_video"
    CATEGORY = "DeepExemplar/Video"

    def colorize_video(self, video_frames: any, reference_image: torch.Tensor,
                       frame_propagate: bool, use_half_resolution: bool,
                       target_width: int, target_height: int,
                       use_torch_compile: bool = False, use_sage_attention: bool = False) -> (list,):
        # Start timing
        start_time = time.time()

        load_models_if_needed()

        # Declare globals first
        global NONLOCAL_NET, COLOR_NET, VGG_NET

        # DeepExemplar only uses VGG19 (trained baseline)
        encoder_net = VGG_NET

        # Apply optimizations to models
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                # Get optimal compile settings based on system compatibility
                compile_kwargs = get_torch_compile_kwargs()

                if not hasattr(VGG_NET, '_compiled'):
                    VGG_NET = torch.compile(VGG_NET, **compile_kwargs)
                    VGG_NET._compiled = True
                if not hasattr(NONLOCAL_NET, '_compiled'):
                    NONLOCAL_NET = torch.compile(NONLOCAL_NET, **compile_kwargs)
                    NONLOCAL_NET._compiled = True
                if not hasattr(COLOR_NET, '_compiled'):
                    COLOR_NET = torch.compile(COLOR_NET, **compile_kwargs)
                    COLOR_NET._compiled = True

                mode_str = f"mode={compile_kwargs['mode']}"
                if 'disable_cudagraphs' in compile_kwargs:
                    mode_str += ", CUDA graphs disabled"
                print(f"[DeepExColorVideoNode] ✓ torch.compile enabled ({mode_str})")
            except Exception as e:
                print(f"[DeepExColorVideoNode] torch.compile failed: {e}")

        # Set SageAttention flag for NonlocalNet
        if use_sage_attention:
            NonlocalNet.USE_SAGE_ATTENTION = True
            print("[DeepExColorVideoNode] ✓ SageAttention enabled")
        else:
            NonlocalNet.USE_SAGE_ATTENTION = False

        # Adjust target size (VGG19 requires dimensions divisible by 32)
        min_multiple = 32

        final_h, final_w = adjust_target_size(target_height, target_width, 64, 64, min_multiple)
        print(f"[DeepExColorVideoNode] Adjusted size: ({target_width}, {target_height}) -> ({final_w}, {final_h})")

        # Warn about torch.compile warmup on first run
        if use_torch_compile:
            print(f"[DeepExColorVideoNode] Note: First frame will take 30-60s for torch.compile optimization")
            print(f"[DeepExColorVideoNode] Subsequent frames will be 15-25% faster")

        if isinstance(video_frames, list):
            frames_list = video_frames
        elif isinstance(video_frames, torch.Tensor):
            if video_frames.dim() == 4:
                frames_list = [video_frames[i] for i in range(video_frames.size(0))]
            else:
                frames_list = [video_frames]
        else:
            frames_list = [video_frames]
        print(f"[DeepExColorVideoNode] Number of frames: {len(frames_list)}")
        transform = build_test_py_transform((final_h, final_w))
        ref_pil = tensor_to_pil(reference_image)
        ref_lab_full = transform(ref_pil).unsqueeze(0).cuda()
        if use_half_resolution:
            ref_half = F.interpolate(ref_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        else:
            ref_half = ref_lab_full.clone()
        I_ref_l = ref_half[:, 0:1, :, :]
        I_ref_ab = ref_half[:, 1:3, :, :]
        I_ref_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_ref_l), I_ref_ab), dim=1)).to(ref_half.device)
        with torch.no_grad():
            features_B = encoder_net(I_ref_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        out_frames = []
        last_lab = None
        num_frames = len(frames_list)
        gui_pbar = ProgressBar(num_frames)
        console_pbar = console_tqdm(total=num_frames, desc="[DeepExColorVideoNode] Console Progress")

        if use_torch_compile:
            print(f"[DeepExColorVideoNode] Starting compilation on first frame...")

        for i, frm in enumerate(frames_list):
            frame_pil = tensor_to_pil(frm)
            frm_lab_full = transform(frame_pil).unsqueeze(0).cuda()
            if use_half_resolution:
                frm_half = F.interpolate(frm_lab_full, scale_factor=0.5, mode="bilinear", align_corners=False)
            else:
                frm_half = frm_lab_full.clone()
            if i == 0:
                with torch.no_grad():
                    I_ab, _, _ = frame_colorization(
                        frm_half, ref_half, torch.zeros_like(frm_half),
                        features_B, encoder_net, NONLOCAL_NET, COLOR_NET,
                        feature_noise=0, temperature=1e-10
                    )
                if frame_propagate:
                    last_lab = torch.cat([frm_half[:, 0:1, :, :], I_ab], dim=1)
            else:
                if frame_propagate and last_lab is not None:
                    with torch.no_grad():
                        I_ab, _, _ = frame_colorization(
                            frm_half, ref_half, last_lab,
                            features_B, encoder_net, NONLOCAL_NET, COLOR_NET,
                            feature_noise=0, temperature=1e-10
                        )
                    last_lab = torch.cat([frm_half[:, 0:1, :, :], I_ab], dim=1)
                else:
                    with torch.no_grad():
                        I_ab, _, _ = frame_colorization(
                            frm_half, ref_half, torch.zeros_like(frm_half),
                            features_B, encoder_net, NONLOCAL_NET, COLOR_NET,
                            feature_noise=0, temperature=1e-10
                        )
                    last_lab = None
            if use_half_resolution:
                ab_up = F.interpolate(I_ab, scale_factor=2.0, mode="bilinear", align_corners=False) * 1.25
            else:
                ab_up = I_ab.clone()

            # Convert to RGB
            out_np = batch_lab2rgb_transpose_mc(frm_lab_full[:, 0:1, :, :], ab_up)
            if i == 0:  # Debug first frame only
                print(f"[DeepExColorVideoNode] DEBUG frame {i}: out_np shape before squeeze: {out_np.shape}")
            # Ensure numpy array has shape (H, W, 3)
            while out_np.ndim > 3:
                out_np = out_np.squeeze(0)
            if i == 0:  # Debug first frame only
                print(f"[DeepExColorVideoNode] DEBUG frame {i}: out_np shape after squeeze: {out_np.shape}")
            # Convert to tensor with shape (H, W, C) for ComfyUI
            out_t = torch.from_numpy(out_np).float().div(255.0)
            if i == 0:  # Debug first frame only
                print(f"[DeepExColorVideoNode] DEBUG frame {i}: out_t shape: {out_t.shape}")
                if use_torch_compile:
                    print(f"[DeepExColorVideoNode] ✓ Compilation complete! Processing remaining frames...")
            out_frames.append(out_t)
            gui_pbar.update_absolute(i+1, num_frames)
            console_pbar.update(1)
            if hasattr(self, 'set_progress'):
                self.set_progress(gui_pbar.progress)
        console_pbar.close()

        # Stack frames for batch output
        out_tensor = torch.stack(out_frames, dim=0)
        print(f"[DeepExColorVideoNode] DEBUG: out_tensor shape: {out_tensor.shape}")
        print(f"[DeepExColorVideoNode] DEBUG: out_tensor dtype: {out_tensor.dtype}")
        print(f"[DeepExColorVideoNode] DEBUG: out_tensor min/max: {out_tensor.min():.3f}/{out_tensor.max():.3f}")

        # Calculate timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Generate performance report
        fps = num_frames / elapsed_time if elapsed_time > 0 else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = (
            f"Deep Exemplar Video Colorization Report\n"
            f"{'=' * 50}\n"
            f"Date/Time: {timestamp}\n"
            f"Frames Processed: {num_frames}\n"
            f"Resolution: {final_w}x{final_h}\n"
            f"Total Time: {elapsed_time:.2f} seconds\n"
            f"Average FPS: {fps:.2f}\n"
            f"Time per Frame: {elapsed_time / num_frames:.3f} seconds\n"
        )
        report += (
            f"Frame Propagation: {'Enabled' if frame_propagate else 'Disabled'}\n"
            f"Half Resolution: {'Enabled' if use_half_resolution else 'Disabled'}\n"
            f"Torch Compile: {use_torch_compile}\n"
            f"SageAttention: {use_sage_attention}\n"
            f"{'=' * 50}"
        )

        print(f"[DeepExColorVideoNode] Processed {num_frames} frames in {elapsed_time:.2f}s ({fps:.2f} FPS)")

        return (out_tensor, report)

NODE_CLASS_MAPPINGS = {
    "DeepExColorImageNode": DeepExColorImageNode,
    "DeepExColorVideoNode": DeepExColorVideoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepExColorImageNode": "Deep Exemplar Image Colorization (Original)",
    "DeepExColorVideoNode": "Deep Exemplar Video Colorization (Original)",
}
