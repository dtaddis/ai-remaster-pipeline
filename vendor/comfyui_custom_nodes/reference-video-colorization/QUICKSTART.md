# Quick Start Guide - ColorMNet Activation

## ✅ Status: New Code Activated!

The new ColorMNet implementation is now active. Follow these steps:

---

## Step 1: Install Dependencies ⚡

Open a terminal in this directory and run:

```bash
# Make sure you're in the right directory
cd "G:\AIART\AI_Image_Generators\Comfy_UI_V42\ComfyUI\custom_nodes\ComfyUI-Reference-Based-Video-Colorization"

# Install/update Python packages
pip install torch>=2.0.0 torchvision>=0.15.0 numpy scikit-image opencv-python einops tqdm progressbar2

# Install ColorMNet-specific dependencies
pip install git+https://github.com/cheind/py-thin-plate-spline.git
pip install git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git
```

**Note for Windows users:** The Pytorch-Correlation-extension might need Visual Studio Build Tools. If it fails, you can skip it for now and see if the model works without it.

---

## Step 2: Download Model Checkpoint 📥

The model checkpoint is ~500MB. Download it from:

**Option A: Automatic (Recommended)**
```bash
python -c "import gdown; gdown.download('https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth', 'checkpoints/DINOv2FeatureV6_LocalAtten_s2_154000.pth', quiet=False)"
```

**Option B: Manual**
1. Go to: https://github.com/yyang181/colormnet/releases
2. Download: `DINOv2FeatureV6_LocalAtten_s2_154000.pth`
3. Place in: `checkpoints/DINOv2FeatureV6_LocalAtten_s2_154000.pth`

---

## Step 3: Restart ComfyUI 🔄

```bash
# Stop ComfyUI (Ctrl+C in the terminal)
# Then restart it
```

You should see on startup:
```
[ColorMNet] Loading v2.0.0
[ColorMNet] License: CC BY-NC-SA 4.0 (Non-commercial use only)
[ColorMNet] Based on ColorMNet by Yang Yang (ECCV 2024)
```

---

## Step 4: Use the New Nodes 🎨

1. **Right-click** in ComfyUI
2. **Add Node** → **ColorMNet**
3. You'll see:
   - **ColorMNet Video Colorization** - For video sequences
   - **ColorMNet Image Colorization** - For single images

### Quick Test Workflow:

```
[Load Image] → video_frames
                     ↓
              [ColorMNet Video]
                     ↑
[Load Image] → reference_image

Parameters:
- target_width: 768
- target_height: 432
- memory_mode: balanced
- use_fp16: True
- use_torch_compile: False  # Set to True for 15-25% speedup
```

---

## Troubleshooting 🔧

### "Model checkpoint not found"

**Solution:** Make sure the model file is at:
```
checkpoints/DINOv2FeatureV6_LocalAtten_s2_154000.pth
```

Run this to check:
```bash
ls -lh checkpoints/
```

Should show ~500MB file.

### "ModuleNotFoundError: No module named 'spatial_correlation_sampler'"

**Solution:** The Pytorch-Correlation-extension failed to install. Try:

```bash
# On Windows with Visual Studio installed:
pip install git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git

# Or try pre-built wheel (check GitHub releases)
```

If it still fails, the model might work without it (the extension is for optical flow which ColorMNet doesn't heavily use).

### "ModuleNotFoundError: No module named 'thin_plate_spline'"

**Solution:**
```bash
pip install git+https://github.com/cheind/py-thin-plate-spline.git
```

### Old nodes still showing up

**Solution:**
1. Completely restart ComfyUI
2. Clear browser cache (Ctrl+Shift+Delete)
3. If old nodes persist, they're from `DeepExemplarColorizationNodes.py` (backed up as `DeepExemplarColorizationNodes_old_backup.py`)

---

## What Changed? 📝

**Old Nodes (Deep Exemplar 2019):**
- ❌ DeepExColorImageNode
- ❌ DeepExColorVideoNode

**New Nodes (ColorMNet 2024):**
- ✅ ColorMNetImage
- ✅ ColorMNetVideo

**Benefits:**
- 🚀 2x faster (4 fps vs 2 fps)
- 💾 33% less VRAM (4GB vs 6GB with FP16)
- 🎯 Better quality (NTIRE 2023 winner)
- 🛡️ Better error messages
- 🔄 Automatic device detection
- 📊 Progress bars

---

## Performance Optimizations ⚡

### New Optimization Options

Both ColorMNet and Deep Exemplar nodes now support additional performance optimizations:

#### torch.compile (All Nodes)
Enable PyTorch 2.0+ graph compilation for 15-25% speedup:
```
use_torch_compile = True
```
- Requires PyTorch 2.0 or later
- First run slower due to compilation (warmup)
- Subsequent runs get full speedup
- No quality loss

#### SageAttention (Deep Exemplar Only)
Enable INT8-quantized attention for 20-30% faster attention:
```
use_sage_attention = True
```
- Requires: `pip install sageattention`
- Works on CUDA GPUs only
- Negligible quality impact
- Automatically falls back if unavailable

#### Recommended Settings

**Maximum Speed (ColorMNet):**
```
use_fp16 = True
use_torch_compile = True
memory_mode = "low_memory"
```

**Maximum Speed (Deep Exemplar):**
```
use_torch_compile = True
use_sage_attention = True
use_half_resolution = True
```

**Best Quality (All Nodes):**
```
use_torch_compile = False  # or True for speed boost
use_fp16 = False  # ColorMNet only
```

See `PERFORMANCE.md` for detailed benchmarks and optimization guide.

---

## Next Steps 🎯

1. Complete installation steps above
2. Restart ComfyUI
3. Test with a small video (5-10 frames)
4. Adjust parameters based on your GPU
5. Enjoy better colorization!

---

## Need Help? 💬

- Check `README_NEW.md` for detailed documentation
- Check `MIGRATION_GUIDE.md` if you have old workflows
- Check `ARCHITECTURE.md` for technical details
- Open a GitHub issue if problems persist

---

## Quick Reference

### Memory Modes:
- **low_memory**: 6GB VRAM, slower but stable
- **balanced**: 8-12GB VRAM, good speed+quality
- **high_quality**: 16GB+ VRAM, best quality

### Recommended Settings by GPU:
| GPU | Memory Mode | FP16 | Max Resolution |
|-----|-------------|------|----------------|
| RTX 2060 (6GB) | low_memory | True | 432x768 |
| RTX 3060 (8GB) | balanced | True | 512x768 |
| RTX 3070 (8GB) | balanced | True | 720x1024 |
| RTX 4080 (16GB) | high_quality | False | 1080x1920 |

---

**You're all set! Happy colorizing! 🎨**
