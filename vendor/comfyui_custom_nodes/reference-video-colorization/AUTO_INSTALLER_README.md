# Auto-Installer Documentation

**Silent dependency installer for modern video colorization components**

---

## Overview

The auto-installer automatically detects and installs missing Python packages when users select modern components (DINOv2, CLIP, color-matcher, etc.) in the ComfyUI nodes.

**Key Features:**
- ✅ Silent installation with progress logging
- ✅ Smart caching to avoid repeated attempts
- ✅ Graceful failure handling
- ✅ Can run standalone or integrated
- ✅ No user interaction required

---

## Architecture

### Module: `auto_installer.py`

**Core Functions:**

1. **`check_package_installed(package_name)`**
   - Checks if a package is importable
   - Returns True/False

2. **`install_package(package_spec, display_name)`**
   - Installs via pip subprocess
   - Silent installation (stdout suppressed)
   - Caches result to avoid retry
   - Returns True on success

3. **Component-Specific Installers:**
   - `ensure_timm()` - For DINOv2 encoders
   - `ensure_clip()` - For CLIP encoder
   - `ensure_color_matcher()` - For color-matcher post-processing
   - `ensure_opencv_contrib()` - For WLS/Guided filters

4. **Smart Installers:**
   - `ensure_dependencies_for_encoder(encoder_type)` - Auto-detects encoder needs
   - `ensure_dependencies_for_post_processor(processor_type)` - Auto-detects processor needs

5. **Batch Installer:**
   - `install_all_modern_components()` - Installs everything at once

---

## Integration with Nodes

### Image Node Integration

```python
# In colorize_image() method
if feature_encoder != "vgg19":
    # Auto-install dependencies
    from .auto_installer import ensure_dependencies_for_encoder
    deps_ok = ensure_dependencies_for_encoder(feature_encoder)

    if deps_ok:
        # Load modern encoder
        encoder_net = get_feature_encoder(feature_encoder, device='cuda')
    else:
        # Fall back to VGG19
        encoder_net = VGG_NET
```

### Post-Processor Integration

```python
# In colorize_image() method
if post_processor != "none":
    # Auto-install dependencies
    from .auto_installer import ensure_dependencies_for_post_processor
    deps_ok = ensure_dependencies_for_post_processor(post_processor)

    if deps_ok:
        # Create post-processor
        proc = get_post_processor(post_processor, **kwargs)
```

---

## Installation Flow

### User Perspective:

1. User selects `dinov2_vitb` in ComfyUI node
2. Runs workflow
3. Console shows:
   ```
   [AutoInstall] timm not found (required for DINOv2)
   [AutoInstall] Installing timm (DINOv2 support)...
   [AutoInstall] ✓ timm installed successfully
   [DeepExColorImageNode] ✓ Using feature encoder: dinov2_vitb
   ```
4. Node loads DINOv2 and continues normally
5. Next run: Already installed, no messages

### Developer Perspective:

```python
# Call chain:
ensure_dependencies_for_encoder('dinov2_vitb')
  └─> ensure_timm()
      └─> check_package_installed('timm')  # Returns False
      └─> install_package('timm>=0.9.0', 'timm (DINOv2 support)')
          └─> subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'timm>=0.9.0'])
          └─> Cache result in _INSTALLATION_CACHE
          └─> Return True
```

---

## Dependency Mapping

### Encoders:

| Encoder Type | Dependencies | Auto-Install Function |
|--------------|--------------|----------------------|
| `vgg19` | None | N/A |
| `dinov2_vits` | `timm>=0.9.0` | `ensure_timm()` |
| `dinov2_vitb` | `timm>=0.9.0` | `ensure_timm()` |
| `dinov2_vitl` | `timm>=0.9.0` | `ensure_timm()` |
| `clip_vitb` | `clip @ git+https://...` | `ensure_clip()` |

### Post-Processors:

| Processor Type | Dependencies | Auto-Install Function |
|----------------|--------------|----------------------|
| `none` | None | N/A |
| `bilateral` | `opencv-python` | Already required |
| `wls` | `opencv-contrib-python>=4.7.0` | `ensure_opencv_contrib()` |
| `guided` | `opencv-contrib-python>=4.7.0` | `ensure_opencv_contrib()` |
| `color_matcher` | `color-matcher>=0.3.0` | `ensure_color_matcher()` |

---

## Caching Mechanism

**Problem:** Don't want to retry installation every time if it failed once

**Solution:** Global cache dictionary

```python
_INSTALLATION_CACHE = {}

def install_package(package_spec, display_name):
    # Check cache first
    if package_spec in _INSTALLATION_CACHE:
        return _INSTALLATION_CACHE[package_spec]

    # Try installation
    success = do_pip_install(package_spec)

    # Cache result
    _INSTALLATION_CACHE[package_spec] = success
    return success
```

**Cache lifetime:** Process lifetime (cleared on ComfyUI restart)

---

## Error Handling

### Graceful Failure:

```python
try:
    # Auto-install
    deps_ok = ensure_dependencies_for_encoder(feature_encoder)

    if deps_ok:
        encoder_net = get_feature_encoder(feature_encoder)
    else:
        raise ImportError(f"Dependencies could not be installed")

except Exception as e:
    print(f"Warning: Could not load {feature_encoder}: {e}")
    print("Falling back to VGG19")
    encoder_net = VGG_NET
```

**Node never crashes** - Always falls back to VGG19 or no post-processing

---

## Standalone Usage

### Install Everything:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Reference-Based-Video-Colorization
python auto_installer.py
```

Output:
```
[AutoInstall] Installing all modern components...
[AutoInstall] This may take a few minutes on first run...
[AutoInstall] Installing timm (DINOv2 support)...
[AutoInstall] ✓ timm installed successfully
[AutoInstall] Installing CLIP...
[AutoInstall] ✓ CLIP installed successfully
[AutoInstall] Installing color-matcher...
[AutoInstall] ✓ color-matcher installed successfully
[AutoInstall] Installing opencv-contrib-python...
[AutoInstall] ✓ opencv-contrib-python installed successfully

[AutoInstall] Installation Summary:
[AutoInstall]   ✓ timm
[AutoInstall]   ✓ clip
[AutoInstall]   ✓ color_matcher
[AutoInstall]   ✓ opencv_contrib
[AutoInstall] ✓ All components installed successfully!
```

### Programmatic Usage:

```python
from auto_installer import (
    ensure_timm,
    ensure_clip,
    ensure_color_matcher,
    install_all_modern_components
)

# Install specific component
if ensure_timm():
    print("timm is ready!")

# Install everything
results = install_all_modern_components()
```

---

## Console Output Examples

### First-Time DINOv2 Usage:

```
[AutoInstall] timm not found (required for DINOv2)
[AutoInstall] Installing timm (DINOv2 support)...
[AutoInstall] ✓ timm installed successfully
[DeepExColorImageNode] ✓ Using feature encoder: dinov2_vitb
```

### First-Time CLIP Usage:

```
[AutoInstall] CLIP not found (required for text-guided colorization)
[AutoInstall] Installing CLIP...
[AutoInstall] ✓ CLIP installed successfully
[DeepExColorImageNode] ✓ Using feature encoder: clip_vitb
[DeepExColorImageNode] ✓ Text guidance: 'warm sunset colors' (weight=0.3)
```

### First-Time color-matcher Usage:

```
[AutoInstall] color-matcher not found (required for color matching post-processing)
[AutoInstall] Installing color-matcher...
[AutoInstall] ✓ color-matcher installed successfully
[DeepExColorImageNode] ✓ Post-processing applied: color_matcher
```

### Installation Failure:

```
[AutoInstall] timm not found (required for DINOv2)
[AutoInstall] Installing timm (DINOv2 support)...
[AutoInstall] ✗ Failed to install timm: <error details>
[DeepExColorImageNode] Warning: Could not load dinov2_vitb: Dependencies could not be installed
[DeepExColorImageNode] Falling back to VGG19
```

---

## Platform Considerations

### Windows:
- Works out of the box
- May need to run ComfyUI as administrator if permissions issue

### Linux/Mac:
- Usually works without sudo
- If permission denied: User needs to manually run with sudo or --user flag

### Virtual Environments:
- Auto-installs into current Python environment
- Uses `sys.executable` to ensure correct pip

---

## Security Considerations

### Safe:
- ✅ Only installs from PyPI or official GitHub repos
- ✅ Uses specific version constraints
- ✅ No arbitrary code execution
- ✅ Suppresses stdout but preserves stderr for debugging

### Limitations:
- ⚠️ Installs packages without user confirmation
- ⚠️ May conflict with existing installations
- ⚠️ Requires internet connection

**Mitigation:**
- Users can disable by manually setting `feature_encoder: vgg19`
- Clear console output shows what's being installed
- Graceful failure doesn't break workflow

---

## Testing

### Test Auto-Installer Standalone:

```bash
# Uninstall packages first
pip uninstall -y timm clip color-matcher

# Run auto-installer
python auto_installer.py

# Verify installation
python -c "import timm; import clip; from color_matcher import ColorMatcher; print('✓ All installed')"
```

### Test Node Integration:

1. Uninstall timm: `pip uninstall -y timm`
2. In ComfyUI, set `feature_encoder: dinov2_vitb`
3. Run workflow
4. Check console for auto-install messages
5. Verify node uses DINOv2 successfully

---

## Future Enhancements

**Potential improvements:**

1. **User confirmation prompt:**
   ```python
   print(f"Install {package}? [Y/n]")
   if input().lower() == 'y':
       install_package(package)
   ```

2. **Download progress bar:**
   ```python
   # Instead of suppressing stdout
   subprocess.check_call([...], stdout=None)
   ```

3. **Offline mode:**
   ```python
   def install_package(..., allow_network=True):
       if not allow_network:
           print("Offline mode - skipping installation")
           return False
   ```

4. **Version checking:**
   ```python
   def check_package_version(package, min_version):
       # Check if installed version meets requirement
   ```

---

## Troubleshooting

### Auto-installer not running:

**Check:**
- Is `auto_installer.py` in the correct directory?
- Are imports working? `from .auto_installer import ...`

### Installation fails:

**Common causes:**
- No internet connection
- Firewall blocking pip
- Insufficient permissions
- Conflicting package versions

**Solution:**
- Check console for specific error
- Try manual install: `pip install <package>`
- Check pip works: `pip --version`

### Packages install but don't import:

**Possible cause:**
- Multiple Python environments
- ComfyUI using different Python than pip

**Solution:**
- Use `sys.executable` (auto-installer already does this)
- Verify with: `which python` and `which pip`

---

## API Reference

### `check_package_installed(package_name: str) -> bool`

Check if package can be imported.

**Args:**
- `package_name`: Module name to import (e.g., 'timm', 'clip')

**Returns:**
- `True` if importable, `False` otherwise

---

### `install_package(package_spec: str, display_name: Optional[str] = None) -> bool`

Install a package via pip.

**Args:**
- `package_spec`: Pip package specification (e.g., 'timm>=0.9.0')
- `display_name`: Human-readable name for logging

**Returns:**
- `True` if installation successful, `False` otherwise

---

### `ensure_dependencies_for_encoder(encoder_type: str) -> bool`

Ensure all dependencies for encoder are installed.

**Args:**
- `encoder_type`: Encoder name ('dinov2_vitb', 'clip_vitb', etc.)

**Returns:**
- `True` if dependencies available, `False` otherwise

---

### `ensure_dependencies_for_post_processor(processor_type: str) -> bool`

Ensure all dependencies for post-processor are installed.

**Args:**
- `processor_type`: Processor name ('color_matcher', 'wls', etc.)

**Returns:**
- `True` if dependencies available, `False` otherwise

---

### `install_all_modern_components() -> dict`

Install all modern components at once.

**Returns:**
- Dictionary with installation status: `{'timm': True, 'clip': True, ...}`

---

**Auto-installer makes modern AI models accessible with zero configuration!** 🚀
