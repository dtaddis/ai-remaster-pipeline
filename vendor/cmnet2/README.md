# CMNET2 : Reference-Based Video Colorization

**CMNET2** is a deep-learning system for colorizing grayscale images and videos using colored reference frames. It is built on top of [ColorMNet](https://github.com/yyang181/colormnet) and extends it with an improved three-tier memory architecture inspired by [XMem++](https://github.com/mbzuai-metaverse/XMem2), enabling robust colorization of long videos with hundreds of reference frames.

---

## Key Features

- **Reference-based colorization** : propagates color from one or more colored reference frames to a grayscale video, operating in the LAB color space for perceptual accuracy.
- **Permanent memory (XMem++ style)** : reference frames are stored in a dedicated `perm_mem` store that is never compressed or evicted, ensuring color fidelity across the entire video.
- **Preloading API** : reference frames can be bulk-loaded into memory before colorization begins, decoupling the reference ingestion phase from the inference phase.
- **Sliding window memory management** : for long videos with thousands of reference frames, a configurable sliding window evicts the oldest references and loads new ones as the video progresses, keeping VRAM usage bounded.
- **Adaptive VRAM management** : gradual memory pressure response: slides 70% of permanent memory when VRAM drops below 500 MB, full reset only as a last resort below 100 MB.
- **DINOv2 + ResNet50 fusion backbone** : multi-scale key features are extracted by fusing DINOv2 ViT-S/14 semantic features with ResNet50 spatial features at 1/4, 1/8, and 1/16 scales.
- **GPU-accelerated LAB→RGB conversion** : `lab2rgb` implemented with exact CIE formulas on GPU via PyTorch, replacing the CPU-bound skimage conversion (-14% total frame time).
- **Chroma transfer pipeline** : optional input resize + YUV chroma transfer for a 3× speedup on full-resolution videos, with no perceptible quality loss.

---

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- CUDA-capable GPU (16 GB VRAM recommended for long videos)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow scikit-image tqdm numpy
```

---

## Directory Structure

```
cmnet2/
├── weights/
│   └── DINOv2FeatureV6_LocalAtten_s2_154000.pth   # ColorMNet pre-trained weights
│
├── models/
│   ├── checkpoints/
│   │   ├── dinov2_vits14_pretrain.pth              # DINOv2 ViT-S/14 backbone weights
│   │   ├── resnet18-5c106cde.pth                   # ResNet18 pre-trained weights
│   │   └── resnet50-19c8e357.pth                   # ResNet50 pre-trained weights
│   │
│   └── facebookresearch_dinov2_main/               # DINOv2 source code (required by torch.hub)
│
├── assets/
│   ├── image/                                      # sample image for test_imge.py
│   ├── video/                                      # sample short video for test_video.py
│   ├── video_full/
│   │   ├── sample_bw_full.mp4                      # sample 5-min B&W clip for test_video_full.py
│   │   └── ref/                                    # colored reference frames
│   └── video_slide/                                # sample video for test_video_slide.py
│
├── colormnet/                                      # model source code
├── test_imge.py                                    # single image colorization
├── test_video.py                                   # video colorization (all refs preloaded)
├── test_video_slide.py                             # video colorization (basic sliding window)
└── test_video_full.py                              # long video with full sliding window pipeline
```

> **Note:** The `weights/` and `models/` directories are not included in the repository.
> Download all required files from the [Releases page](https://github.com/dan64/cmnet2/releases/tag/v1.0.0) as described below.

---

## Download Model Weights

Download the following files from the [v1.0.0 Release](https://github.com/dan64/cmnet2/releases/tag/v1.0.0) and place them in the correct directories:

| File                                       | Destination           | Download                                                                                                      |
| ------------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `DINOv2FeatureV6_LocalAtten_s2_154000.pth` | `weights/`            | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/DINOv2FeatureV6_LocalAtten_s2_154000.pth) |
| `dinov2_vits14_pretrain.pth`               | `models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/dinov2_vits14_pretrain.pth)               |
| `resnet18-5c106cde.pth`                    | `models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet18-5c106cde.pth)                    |
| `resnet50-19c8e357.pth`                    | `models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet50-19c8e357.pth)                    |
| `facebookresearch_dinov2_main.zip`         | extract to `models/`  | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/facebookresearch_dinov2_main.zip)         |

> **Note:** `facebookresearch_dinov2_main/` contains the DINOv2 source code required by
> `torch.hub` to instantiate the model. Extract the zip so that the folder is located at
> `models/facebookresearch_dinov2_main/`.

---

## Usage

### Colorize a single image

```bash
python test_imge.py \
  --input  assets/image/image_bw.jpg \
  --ref    assets/image/image_color_ref.jpg \
  --output assets/image/output.jpg
```

### Colorize a video (all references preloaded)

Reference images must be named with the target frame number embedded in the filename
(e.g. `ref_000040.jpg` → applies to frame 40).

```bash
python test_video.py \
  --input    assets/video/sample_bw.mp4 \
  --ref_path assets/video/ref/ \
  --output   assets/video/output.mp4
```

All reference frames are preloaded into `perm_mem` before colorization begins.
The first reference frame is also passed normally at frame 0 to initialize the working memory.

### Colorize a long video with sliding window (`test_video_full.py`)

The main script for production use. Supports long videos with hundreds of reference frames,
optional input resize with chroma transfer, and automatic VRAM-aware window sizing.

```bash
python test_video_full.py \
  --input       assets/video_full/sample_bw_full.mp4 \
  --ref_path    assets/video_full/ref/ \
  --output      assets/video_full/output.mp4 \
  --max_side    512 \
  --window_size 100
```

**CLI parameters:**

| Parameter       | Default | Description                                                                         |
| --------------- | ------- | ----------------------------------------------------------------------------------- |
| `--max_side`    | `-1`    | Resize longest side before colorization. `-1` = original resolution.                |
| `--window_size` | `-1`    | Max reference frames in `perm_mem`. `-1` or `0` = auto (fills until 15% VRAM free). |
| `--top_k`       | `30`    | Top-K for memory matching softmax. Lower = faster, less accurate.                   |
| `--mem_every`   | `5`     | Store a colorized frame in working memory every N frames.                           |

**Performance profile** on a 960×730 clip with 158 reference frames (RTX 5070 Ti, 16 GB VRAM):

| Mode                              | FPS  | Notes                       |
| --------------------------------- | ---- | --------------------------- |
| Full resolution, no resize        | 2.63 | Best quality                |
| Resize to 512px + chroma transfer | 5.80 | Recommended for long videos |

---

## Architecture

```
Grayscale input frame (L channel in LAB)
    ↓
KeyEncoder  ←  ResNet50 (1/4, 1/8, 1/16) + DINOv2 ViT-S/14 (fused via Fuse blocks)
    ↓
Key / Shrinkage / Selection tensors
    ↓
MemoryManager : 3-tier memory
    ├── perm_mem   : reference frames, never evicted         ← XMem++ extension
    ├── work_mem   : recent colorized frames (LRU tracking)
    └── long_mem   : compressed prototypes (128 per consolidation)
    ↓
Memory readout (scaled L2 affinity + softmax, top-k=30)
    ↓
ValueEncoder  ←  ResNet18-based, fuses image features + memory readout
    ↓
Decoder (GRU hidden state + upsampling blocks)
    ↓
AB color channels → LAB →[GPU CIE]→ RGB → colorized frame
    ↓ (if --max_side)
Chroma transfer: L from original full-size + UV from colorized resized → final frame
```

### Core classes

| Class                  | File                                    | Description                                                                                |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------ |
| `ColorMNetRender`      | `colormnet/colormnet_render.py`         | Public API. Singleton. Handles GPU memory, reference management, sliding window.           |
| `InferenceCore`        | `colormnet/inference/inference_core.py` | Frame-by-frame inference loop. Exposes `step()`, `step_AnyExemplar()`, `load_reference()`. |
| `MemoryManager`        | `colormnet/inference/memory_manager.py` | Manages `perm_mem`, `work_mem`, `long_mem`. Handles consolidation and sliding.             |
| `ColorMNet`            | `colormnet/model/network.py`            | Top-level `nn.Module`.                                                                     |
| `KeyEncoder_DINOv2_v6` | `colormnet/model/modules.py`            | DINOv2 + ResNet50 fusion backbone.                                                         |

---

## Public API

```python
from colormnet.colormnet_render import ColorMNetRender
from PIL import Image

colorizer = ColorMNetRender(
    image_size=-1,          # -1 = original resolution
    vid_length=1000,        # total number of frames to colorize
    max_memory_frames=5000, # long-term memory capacity
    encode_mode=1,          # 0=remote, 1=async, 2=sync
    top_k=30,               # memory matching top-K
    mem_every=5,            # working memory update frequency
    project_dir="."
)

# Option A : preload all references before colorization
for ref_img in reference_images:
    colorizer.preload_reference(ref_img)          # loads into perm_mem

colorizer.set_ref_frame(reference_images[0])      # initialize work_mem
frame_colored = colorizer.colorize_frame(ti=0, frame_i=grayscale_frame)

# Option B : pass reference alongside each frame
colorizer.set_ref_frame(ref_img)
frame_colored = colorizer.colorize_frame(ti=i, frame_i=grayscale_frame)

# Sliding window control
count = colorizer.get_perm_mem_frame_count()      # current perm_mem size
colorizer.slide_permanent_memory(n_frames=50)     # evict oldest 50 refs
```

---

## Performance Optimizations

### LAB→RGB conversion on GPU

The original ColorMNet uses `skimage.color.lab2rgb()` on CPU for every output frame.
CMNET2 replaces this with an exact CIE LAB→XYZ→RGB implementation running entirely
on GPU via PyTorch, keeping the tensor on the GPU until the final `detach().cpu()`.
Both implementations are available via the `mode` parameter:

```python
# colormnet/util/transforms.py
lab2rgb_transform_PIL(mask, mode="gpu")  # default : CIE exact on GPU
lab2rgb_transform_PIL(mask, mode="cpu")  # fallback : skimage on CPU
```

This saves ~60ms per frame (-14% total) on a 960×730 input.

### Chroma transfer pipeline (`--max_side`)

When `--max_side` is set, colorization runs at reduced resolution and the color channels
are transferred back to the original frame via YUV chroma transfer:

1. The input frame is downscaled to `max_side` px on the longest side (aspect ratio preserved, even dimensions guaranteed).
2. ColorMNet colorizes the reduced frame.
3. The colorized output is upscaled with LANCZOS4 and its U/V channels are transferred to the original full-resolution frame in YUV space, preserving the original luminance (Y channel) exactly.

This yields a **3× speedup** (1.94 → 5.80 FPS on 960×730) with no perceptible quality loss on the color channels.

---

## Differences from the original ColorMNet

| Feature                | Original ColorMNet         | CMNET2                                      |
| ---------------------- | -------------------------- | ------------------------------------------- |
| Memory stores          | working + long-term        | **permanent** + working + long-term         |
| Reference handling     | passed with each frame     | **preloadable in bulk** before inference    |
| Long video support     | resets memory periodically | **sliding window** over permanent memory    |
| VRAM pressure response | full reset                 | **graduated**: slide 70% → full reset       |
| `reset_on_ref_update`  | active                     | deprecated (permanent memory handles it)    |
| LAB→RGB conversion     | skimage CPU                | **CIE exact on GPU** (-14% frame time)      |
| Full-res output        | always                     | optional **chroma transfer** for 3× speedup |
| Window size            | fixed constant             | **CLI parameter + auto VRAM-aware mode**    |

---

## Credits

CMNET2 is based on:

- **ColorMNet** : [yyang181/colormnet](https://github.com/yyang181/colormnet)
- **XMem** : [hkchengrex/XMem](https://github.com/hkchengrex/XMem)
- **XMem++** : [mbzuai-metaverse/XMem2](https://github.com/mbzuai-metaverse/XMem2)
- **DINOv2** : [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

---

## Projects Using CMNET2

CMNET2 is used as a core component in the following projects:

- **[HAVCServerDiT](https://github.com/dan64/HAVCServerDiT)** — Hybrid Automatic Video Colorizer (HAVC) server that exposes a GPU-accelerated colorization pipeline for B&W images and video frames based on Diffusion Transformer (DiT) models, with CMNET2 as the exemplar-based backbone.
- **[vs-cmnet2](https://github.com/dan64/vs-cmnet2)** — VapourSynth filter for exemplar-based video colorization using CMNET2.
- **[vs-havc](https://github.com/dan64/vs-havc)** — A Deep Learning based VapourSynth filter for colorizing and restoring old images and video, based on DeOldify, DDColor, ColorMNet/CMNET2 and DeepRemaster.

---

## License

This project inherits the license terms of the original ColorMNet repository.
Please refer to the [original repository](https://github.com/yyang181/colormnet) for details.
