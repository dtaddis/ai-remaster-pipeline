"""ComfyUI Deep Exemplar Video Colorization - Dual implementation package.

This package provides ComfyUI nodes for exemplar-based video colorization featuring:
- ColorMNet (2024): Modern memory-based approach with DINOv2 features
- Deep Exemplar (2019): Classic CVPR method with temporal propagation

Version: 2.0.0
Author: ComfyUI Deep Exemplar Contributors
License: MIT (Implementation) | CC BY-NC-SA 4.0 (ColorMNet model) | Original licenses (Deep Exemplar)
Repository: https://github.com/YOUR_USERNAME/ComfyUI-Deep-Exemplar-based-Video-Colorization
"""

__version__ = "2.0.0"
__author__ = "ComfyUI Deep Exemplar Contributors"
__license__ = "MIT"

import sys
import subprocess
import importlib
from pathlib import Path

# Add current directory to Python path so submodules can be imported
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import after path is set so relative modules always resolve
from .colormnet.downloader import ensure_dependencies as ensure_git_dependencies

# Backward compatibility: older revisions imported the upstream code as `model.network`
# We vendor that code under `colormnet_model`, so alias it early to avoid ModuleNotFoundError
try:
    import colormnet_model

    if "model" not in sys.modules:
        sys.modules["model"] = colormnet_model
except Exception as alias_error:
    print(f"[ColorMNet] WARNING: Couldn't register compatibility alias for 'model': {alias_error}")

print("[ColorMNet] Initializing...")

# Check and install dependencies automatically
def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required_packages = {
        'einops': 'einops',
        'progressbar': 'progressbar2',
        'gdown': 'gdown',
        'hickle': 'hickle',
        'skimage': 'scikit-image',
    }

    missing_packages = []

    for import_name, pip_name in required_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"[ColorMNet] Installing missing dependencies: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                *missing_packages,
                "--no-cache-dir"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[ColorMNet] Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ColorMNet] WARNING: Failed to auto-install some dependencies: {e}")
            print(f"[ColorMNet] Please manually install: pip install {' '.join(missing_packages)}")

# Check git-based dependencies (optional, don't block on failure)
def check_git_dependencies():
    """Check for git-installed packages (non-blocking)."""
    optional_packages = {
        'thin_plate_spline': 'py-thin-plate-spline',
        'spatial_correlation_sampler': 'Pytorch-Correlation-extension',
    }

    missing = []
    for import_name, package_name in optional_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print(f"[ColorMNet] WARNING: Optional packages not found: {', '.join(missing)}")
        print(f"[ColorMNet] These will be auto-installed on first model run if needed")

    return missing

# Auto-install dependencies
try:
    check_and_install_dependencies()
    missing_git_deps = check_git_dependencies()
except Exception as e:
    print(f"[ColorMNet] WARNING: Error checking dependencies: {e}")
    missing_git_deps = []

# Import nodes
try:
    # Import new ColorMNet nodes
    from .nodes import NODE_CLASS_MAPPINGS as ColorMNet_Mappings
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as ColorMNet_Display_Mappings

    # Import original Deep Exemplar nodes
    from .DeepExemplarColorizationNodes import NODE_CLASS_MAPPINGS as DeepEx_Mappings
    try:
        from .DeepExemplarColorizationNodes import NODE_DISPLAY_NAME_MAPPINGS as DeepEx_Display_Mappings
    except ImportError:
        # If display names not defined, use class names
        DeepEx_Display_Mappings = {k: k for k in DeepEx_Mappings.keys()}

    # Combine both sets of nodes
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    # Add ColorMNet nodes with clear names
    NODE_CLASS_MAPPINGS.update(ColorMNet_Mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(ColorMNet_Display_Mappings)

    # Add Deep Exemplar nodes with clear names
    NODE_CLASS_MAPPINGS.update(DeepEx_Mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(DeepEx_Display_Mappings)

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

    # Version info
    __version__ = "2.0.0"
    __author__ = "ColorMNet ComfyUI Integration"

    print(f"[ColorMNet] Loaded v{__version__}")
    print("[ColorMNet] License: CC BY-NC-SA 4.0 (Non-commercial use only)")
    print("[ColorMNet] Available nodes:")
    print("[ColorMNet]   NEW: ColorMNetVideo, ColorMNetImage")
    print("[ColorMNet]   ORIGINAL: DeepExColorImageNode, DeepExColorVideoNode")

    if missing_git_deps:
        print(f"[ColorMNet] Note: {len(missing_git_deps)} optional dependencies will auto-install on first use")

except Exception as e:
    print(f"[ColorMNet] ERROR: Failed to load nodes: {e}")
    print(f"[ColorMNet] Check the error above and verify installation")
    import traceback
    traceback.print_exc()

    # Provide empty mappings so ComfyUI doesn't crash
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
