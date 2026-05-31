"""
Auto-Installer for Modern Components
Automatically installs missing dependencies when needed.
"""

import sys
import subprocess
import importlib
from typing import Optional

# Cache to avoid repeated installation attempts
_INSTALLATION_CACHE = {}


def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_spec: str, display_name: Optional[str] = None) -> bool:
    """
    Silently install a package via pip.

    Args:
        package_spec: Package specification (e.g., "timm>=0.9.0" or "git+https://...")
        display_name: Human-readable name for logging

    Returns:
        True if installation successful, False otherwise
    """
    if display_name is None:
        display_name = package_spec.split(">=")[0].split("==")[0]

    # Check cache
    if package_spec in _INSTALLATION_CACHE:
        return _INSTALLATION_CACHE[package_spec]

    print(f"[AutoInstall] Installing {display_name}...")
    try:
        # Run pip install
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_spec],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"[AutoInstall] ✓ {display_name} installed successfully")
        _INSTALLATION_CACHE[package_spec] = True
        return True
    except subprocess.CalledProcessError as e:
        print(f"[AutoInstall] ✗ Failed to install {display_name}: {e}")
        _INSTALLATION_CACHE[package_spec] = False
        return False
    except Exception as e:
        print(f"[AutoInstall] ✗ Unexpected error installing {display_name}: {e}")
        _INSTALLATION_CACHE[package_spec] = False
        return False


def ensure_timm() -> bool:
    """Ensure timm is installed (for DINOv2)."""
    if check_package_installed('timm'):
        return True

    print("[AutoInstall] timm not found (required for DINOv2)")
    return install_package("timm>=0.9.0", "timm (DINOv2 support)")


def ensure_clip() -> bool:
    """Ensure CLIP is installed."""
    if check_package_installed('clip'):
        return True

    print("[AutoInstall] CLIP not found (required for text-guided colorization)")
    return install_package("git+https://github.com/openai/CLIP.git", "CLIP")


def ensure_color_matcher() -> bool:
    """Ensure color-matcher is installed."""
    # Check for the actual package name
    try:
        from color_matcher import ColorMatcher
        return True
    except ImportError:
        pass

    print("[AutoInstall] color-matcher not found (required for color matching post-processing)")
    return install_package("color-matcher>=0.3.0", "color-matcher")


def ensure_opencv_contrib() -> bool:
    """Ensure opencv-contrib-python is installed (for WLS/Guided filters)."""
    try:
        import cv2
        # Check if ximgproc is available
        if hasattr(cv2, 'ximgproc'):
            return True
    except ImportError:
        pass

    print("[AutoInstall] opencv-contrib-python not found (required for advanced filters)")
    # Uninstall opencv-python first to avoid conflicts
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except:
        pass

    return install_package("opencv-contrib-python>=4.7.0", "opencv-contrib-python")


def ensure_dependencies_for_encoder(encoder_type: str) -> bool:
    """
    Ensure all dependencies for a given encoder are installed.

    Args:
        encoder_type: Encoder name (e.g., 'dinov2_vitb', 'clip_vitb')

    Returns:
        True if all dependencies are available, False otherwise
    """
    if encoder_type == 'vgg19':
        return True  # No extra dependencies

    elif encoder_type.startswith('dinov2'):
        return ensure_timm()

    elif encoder_type.startswith('clip'):
        return ensure_clip()

    else:
        return True  # Unknown encoder, assume no dependencies


def install_all_modern_components() -> dict:
    """
    Install all modern components at once.

    Returns:
        Dictionary with installation status for each component
    """
    print("[AutoInstall] Installing all modern components...")
    print("[AutoInstall] This may take a few minutes on first run...")

    results = {
        'timm': ensure_timm(),
        'clip': ensure_clip(),
    }

    # Summary
    print("\n[AutoInstall] Installation Summary:")
    for component, success in results.items():
        status = "✓" if success else "✗"
        print(f"[AutoInstall]   {status} {component}")

    all_success = all(results.values())
    if all_success:
        print("[AutoInstall] ✓ All components installed successfully!")
    else:
        print("[AutoInstall] ⚠ Some components failed to install")

    return results


if __name__ == "__main__":
    # When run directly, install everything
    install_all_modern_components()
