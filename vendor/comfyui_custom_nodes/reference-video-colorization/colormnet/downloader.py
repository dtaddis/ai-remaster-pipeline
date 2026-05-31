"""Automatic model checkpoint downloader."""

import sys
import subprocess
import importlib
import os
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import shutil


def ensure_gdown():
    """Ensure gdown is installed."""
    try:
        importlib.import_module('gdown')
        return True
    except ImportError:
        print("[ColorMNet] Installing gdown for model download...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "gdown", "--no-cache-dir"
            ], stdout=subprocess.DEVNULL)
            return True
        except:
            return False


def download_with_progress(url: str, output_path: Path, description: str = "file"):
    """Download file with progress bar using urllib."""
    print(f"[ColorMNet] Downloading {description}...")
    print(f"[ColorMNet]   URL: {url}")
    print(f"[ColorMNet]   Destination: {output_path.name}")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024**2)
            mb_total = total_size / (1024**2)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f'\r[ColorMNet]   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)', end='', flush=True)

    try:
        urllib.request.urlretrieve(url, output_path, show_progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n[ColorMNet] ✗ Download failed: {e}")
        return False


def download_with_gdown(url: str, output_path: Path, description: str = "file"):
    """Download file using gdown (better for Google Drive)."""
    print(f"[ColorMNet] Downloading {description} with gdown...")
    try:
        import gdown
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"[ColorMNet] ✗ gdown failed: {e}")
        return False


def install_git_dependency(repo_url: str, package_name: str) -> bool:
    """Install a package from git repository."""
    print(f"[ColorMNet] Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            f"git+{repo_url}",
            "--no-cache-dir"
        ], stdout=subprocess.DEVNULL)
        print(f"[ColorMNet] ✓ {package_name} installed")
        return True
    except subprocess.CalledProcessError:
        print(f"[ColorMNet] ⚠ {package_name} installation failed (may not be critical)")
        return False


def ensure_model_downloaded(checkpoint_dir: Path) -> Optional[Path]:
    """Ensure model checkpoint is downloaded.

    Args:
        checkpoint_dir: Directory to store checkpoint

    Returns:
        Path to checkpoint file, or None if download failed
    """
    checkpoint_dir.mkdir(exist_ok=True)

    model_filename = "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
    model_path = checkpoint_dir / model_filename

    # Check if already exists
    if model_path.exists():
        file_size_mb = model_path.stat().st_size / (1024**2)
        if file_size_mb > 400:  # Model should be ~500MB
            print(f"[ColorMNet] ✓ Model checkpoint found ({file_size_mb:.1f}MB)")
            return model_path
        else:
            print(f"[ColorMNet] ⚠ Existing checkpoint seems corrupted ({file_size_mb:.1f}MB), re-downloading...")
            model_path.unlink()

    print(f"[ColorMNet] Model checkpoint not found, downloading (~500MB)...")
    print(f"[ColorMNet] This is a one-time download, please wait...")

    # Try multiple download sources
    download_sources = [
        # Direct GitHub release
        ("https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth", "direct"),
        # Hugging Face mirror (if available)
        ("https://huggingface.co/yyang181/colormnet/resolve/main/DINOv2FeatureV6_LocalAtten_s2_154000.pth", "huggingface"),
    ]

    for url, source_name in download_sources:
        print(f"[ColorMNet] Trying {source_name}...")

        success = download_with_progress(url, model_path, f"model from {source_name}")

        if success and model_path.exists():
            file_size_mb = model_path.stat().st_size / (1024**2)
            if file_size_mb > 400:
                print(f"[ColorMNet] ✓ Model downloaded successfully ({file_size_mb:.1f}MB)")
                return model_path
            else:
                print(f"[ColorMNet] ⚠ Downloaded file too small ({file_size_mb:.1f}MB), trying next source...")
                model_path.unlink()

    # All sources failed
    print("[ColorMNet] ✗ Automatic download failed from all sources")
    print("[ColorMNet] Please download manually:")
    print("[ColorMNet]   1. Visit: https://github.com/yyang181/colormnet/releases/tag/v0.1")
    print(f"[ColorMNet]   2. Download: {model_filename}")
    print(f"[ColorMNet]   3. Place at: {model_path}")

    return None


def check_cuda_requirements() -> Tuple[bool, str]:
    """Check if CUDA development tools are available for building extensions.

    Returns:
        Tuple of (requirements_met, feedback_message)
    """
    issues = []

    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not cuda_home:
        issues.append("❌ CUDA_HOME not set")
    else:
        print(f"[ColorMNet] ✓ CUDA_HOME: {cuda_home}")

        # Check for nvcc (CUDA compiler)
        nvcc_path = Path(cuda_home) / "bin" / "nvcc.exe"
        if not nvcc_path.exists():
            issues.append(f"❌ nvcc not found at {nvcc_path}")
        else:
            print(f"[ColorMNet] ✓ nvcc found")

    # Check for Visual Studio C++ compiler
    try:
        result = subprocess.run(['cl.exe'], capture_output=True, text=True)
        print("[ColorMNet] ✓ Visual Studio C++ compiler found")
    except FileNotFoundError:
        issues.append("❌ Visual Studio C++ compiler (cl.exe) not found")

    # Check PyTorch CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("❌ PyTorch CUDA not available")
        else:
            print(f"[ColorMNet] ✓ PyTorch CUDA {torch.version.cuda}")
    except ImportError:
        issues.append("❌ PyTorch not installed")

    if issues:
        feedback = "\n[ColorMNet] ═══════════════════════════════════════════════════════════════\n"
        feedback += "[ColorMNet] Missing requirements for CUDA-optimized compilation:\n"
        feedback += "[ColorMNet] ───────────────────────────────────────────────────────────────\n"
        for issue in issues:
            feedback += f"[ColorMNet]   ✗ {issue}\n"

        feedback += "\n[ColorMNet] To enable 2-5x faster performance, install:\n"
        feedback += "[ColorMNet] ───────────────────────────────────────────────────────────────\n"

        if any("CUDA_HOME" in i for i in issues):
            feedback += "[ColorMNet]   1. CUDA Toolkit (match your PyTorch CUDA version)\n"
            feedback += "[ColorMNet]      • Download: https://developer.nvidia.com/cuda-toolkit\n"
            feedback += "[ColorMNet]      • Set CUDA_HOME environment variable after install\n"
            feedback += "[ColorMNet]        Example: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\n\n"

        if any("cl.exe" in i for i in issues):
            feedback += "[ColorMNet]   2. Visual Studio Build Tools (C++ compiler)\n"
            feedback += "[ColorMNet]      • Download: https://visualstudio.microsoft.com/downloads/\n"
            feedback += "[ColorMNet]      • Install: 'Desktop development with C++' workload\n"
            feedback += "[ColorMNet]      • Restart terminal after installation\n\n"

        feedback += "[ColorMNet] ───────────────────────────────────────────────────────────────\n"
        feedback += "[ColorMNet] Currently using PyTorch fallback (20-40% slower, same quality)\n"
        feedback += "[ColorMNet] No action needed - colorization will work without compilation\n"
        feedback += "[ColorMNet] ═══════════════════════════════════════════════════════════════\n"
        return False, feedback

    return True, "[ColorMNet] ✓ All CUDA requirements met"


def try_install_correlation_sampler() -> Tuple[bool, str]:
    """Attempt to install spatial_correlation_sampler.

    Returns:
        Tuple of (success, message)
    """
    print("[ColorMNet] Checking for spatial_correlation_sampler...")

    # First check if already installed
    try:
        import spatial_correlation_sampler
        print("[ColorMNet] ✓ spatial_correlation_sampler already installed")
        return True, "Optimized CUDA extension available"
    except (ImportError, SyntaxError) as e:
        # SyntaxError can occur if the package is corrupted (null bytes in __init__.py)
        if isinstance(e, SyntaxError):
            print(f"[ColorMNet] ⚠ Corrupted spatial_correlation_sampler installation detected")
            print(f"[ColorMNet] Will use PyTorch fallback implementation")
            # Try to uninstall the corrupted package
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "spatial-correlation-sampler", "-y"],
                             capture_output=True)
            except:
                pass
        pass

    # Check requirements BEFORE attempting installation
    print("[ColorMNet] Checking compilation prerequisites...")
    requirements_met, feedback = check_cuda_requirements()
    if not requirements_met:
        print(feedback)
        print("[ColorMNet] ⚠ Skipping compilation - prerequisites not met")
        print("[ColorMNet] ✓ Using PyTorch fallback implementation (fully functional)")
        return False, "Missing CUDA development tools - using fallback"

    # Try to install
    print("[ColorMNet] ✓ All compilation prerequisites found")
    print("[ColorMNet] Attempting to install spatial_correlation_sampler...")
    print("[ColorMNet] This may take 2-5 minutes to compile...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git",
            "--no-cache-dir"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)

        # Verify installation
        try:
            import spatial_correlation_sampler
            print("[ColorMNet] ✓ spatial_correlation_sampler installed successfully!")
            print("[ColorMNet] 🚀 Optimized CUDA mode enabled (2-5x faster)")
            return True, "Optimized CUDA extension installed"
        except (ImportError, SyntaxError):
            print("[ColorMNet] ⚠ Installation completed but import failed")
            return False, "Installation succeeded but module not importable"

    except subprocess.TimeoutExpired:
        print("[ColorMNet] ⚠ Installation timed out (>10 minutes)")
        return False, "Installation timeout"
    except subprocess.CalledProcessError as e:
        print("[ColorMNet] ⚠ Installation failed (compilation error)")
        return False, "Compilation failed"
    except Exception as e:
        print(f"[ColorMNet] ⚠ Installation failed: {e}")
        return False, f"Installation error: {e}"


def ensure_dependencies() -> bool:
    """Ensure all dependencies are installed.

    Returns:
        True if all critical dependencies are available
    """
    all_ok = True

    # Check thin_plate_spline
    try:
        import thin_plate_spline
        print("[ColorMNet] ✓ py-thin-plate-spline available")
    except ImportError:
        print("[ColorMNet] py-thin-plate-spline not found, installing...")
        if install_git_dependency(
            "https://github.com/cheind/py-thin-plate-spline.git",
            "py-thin-plate-spline"
        ):
            try:
                import thin_plate_spline
                print("[ColorMNet] ✓ py-thin-plate-spline installed successfully")
            except ImportError:
                print("[ColorMNet] ⚠ py-thin-plate-spline install verification failed")
                all_ok = False
        else:
            all_ok = False

    # Check spatial_correlation_sampler (optional, can work without it)
    try:
        import spatial_correlation_sampler
        print("[ColorMNet] ✓ Pytorch-Correlation-extension available")
    except (ImportError, SyntaxError):
        print("[ColorMNet] Pytorch-Correlation-extension not found, installing...")
        print("[ColorMNet] Note: This may require C++ compiler on Windows")
        installed = install_git_dependency(
            "https://github.com/ClementPinard/Pytorch-Correlation-extension.git",
            "Pytorch-Correlation-extension"
        )
        if not installed:
            print("[ColorMNet] ⚠ Pytorch-Correlation-extension failed (may work without it)")
            # Don't set all_ok to False - this is optional

    return all_ok


def setup_model() -> Optional[Path]:
    """Complete setup: install dependencies and download model.

    Returns:
        Path to model checkpoint if successful, None otherwise
    """
    print("[ColorMNet] " + "="*60)
    print("[ColorMNet] Running first-time setup...")
    print("[ColorMNet] " + "="*60)

    # Get checkpoint directory
    script_dir = Path(__file__).parent.parent
    checkpoint_dir = script_dir / "checkpoints"

    # Ensure dependencies
    print("\n[ColorMNet] Step 1/2: Checking dependencies...")
    deps_ok = ensure_dependencies()

    if not deps_ok:
        print("[ColorMNet] ⚠ Some dependencies failed, but attempting to continue...")

    # Download model
    print("\n[ColorMNet] Step 2/2: Downloading model checkpoint...")
    model_path = ensure_model_downloaded(checkpoint_dir)

    print("\n[ColorMNet] " + "="*60)
    if model_path:
        print("[ColorMNet] ✓ Setup complete! ColorMNet is ready to use.")
    else:
        print("[ColorMNet] ✗ Setup incomplete. Please check errors above.")
    print("[ColorMNet] " + "="*60 + "\n")

    return model_path
