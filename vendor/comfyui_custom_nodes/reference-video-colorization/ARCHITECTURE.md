# ColorMNet ComfyUI Integration - Architecture Design

## Overview
Modern, production-ready implementation of ColorMNet for reference-based video colorization in ComfyUI.

## Design Principles

1. **Separation of Concerns**
   - Core model logic separate from ComfyUI interface
   - Clear boundaries between components

2. **Modern Python Practices**
   - Type hints throughout
   - Proper error handling with specific exceptions
   - Structured logging (not print statements)
   - Context managers for resource management

3. **Device Agnostic**
   - Automatic device detection (CUDA/CPU/MPS)
   - Graceful fallback
   - Proper memory management

4. **Production Ready**
   - Comprehensive error messages
   - Input validation
   - Progress tracking
   - Memory efficient (FP16 support)

## Directory Structure

```
ComfyUI-Reference-Based-Video-Colorization/
├── __init__.py                      # ComfyUI entry point
├── nodes.py                         # ComfyUI node definitions
├── requirements.txt                 # Modern dependencies
├── install.py                       # Model downloader
├── README.md                        # User documentation
├── ARCHITECTURE.md                  # This file
│
├── colormnet/                       # ColorMNet integration
│   ├── __init__.py
│   ├── model.py                     # Core model wrapper
│   ├── inference.py                 # Inference pipeline
│   ├── config.py                    # Configuration management
│   └── utils.py                     # Utility functions
│
├── core/                            # Shared core functionality
│   ├── __init__.py
│   ├── device.py                    # Device management
│   ├── logger.py                    # Logging configuration
│   ├── validation.py                # Input validation
│   ├── transforms.py                # Color space conversions
│   └── exceptions.py                # Custom exceptions
│
├── models/                          # Model architecture (from ColorMNet)
│   └── (ColorMNet model files)
│
├── inference/                       # Inference logic (from ColorMNet)
│   └── (ColorMNet inference files)
│
└── checkpoints/                     # Model weights
    └── (downloaded automatically)
```

## Core Components

### 1. Device Manager (`core/device.py`)
```python
class DeviceManager:
    """Manages device selection and memory"""
    - Auto-detect CUDA/MPS/CPU
    - Memory monitoring
    - Automatic FP16 where applicable
    - Graceful degradation
```

### 2. Logger (`core/logger.py`)
```python
"""Structured logging with levels"""
- DEBUG: Detailed diagnostic info
- INFO: Progress updates
- WARNING: Recoverable issues
- ERROR: Failures with context
```

### 3. Validation (`core/validation.py`)
```python
"""Input validation with clear error messages"""
- Tensor shape/type checking
- Resolution constraints
- Memory estimation
- Early failure detection
```

### 4. ColorMNet Wrapper (`colormnet/model.py`)
```python
class ColorMNetModel:
    """Clean interface to ColorMNet"""
    - Lazy loading
    - Memory-efficient inference
    - Batch processing
    - Progress callbacks
```

### 5. ComfyUI Nodes (`nodes.py`)
```python
class ColorMNetVideoNode:
    """Main ComfyUI interface"""
    - Type-safe inputs
    - Progress integration
    - Error handling
    - Memory management
```

## Key Features

### Error Handling
```python
try:
    result = process_video(frames, reference)
except ModelNotFoundError as e:
    logger.error(f"Model checkpoint missing: {e}")
    # Provide download instructions
except InsufficientVRAMError as e:
    logger.error(f"Not enough VRAM: {e}")
    # Suggest lower resolution or FP16
except InvalidInputError as e:
    logger.error(f"Invalid input: {e}")
    # Clear error message to user
```

### Memory Management
```python
with torch.cuda.amp.autocast(enabled=use_fp16):
    # Automatic mixed precision
    result = model(input)

# Explicit cleanup
del intermediate_tensors
torch.cuda.empty_cache()
```

### Progress Tracking
```python
from comfy.utils import ProgressBar

pbar = ProgressBar(total_frames)
for i, frame in enumerate(frames):
    result = process_frame(frame)
    pbar.update_absolute(i+1, total_frames)
```

## Configuration System

### User-facing parameters
```python
{
    "reference_image": IMAGE,      # ComfyUI IMAGE tensor
    "video_frames": IMAGE,         # Batch of frames
    "target_resolution": (h, w),   # Output size
    "use_fp16": bool,              # Memory/speed tradeoff
    "memory_mode": str,            # "low" | "balanced" | "high"
}
```

### Internal configuration
```python
{
    "device": "cuda" | "cpu" | "mps",
    "max_memory_frames": int,      # Based on available VRAM
    "enable_long_term_memory": bool,
    "num_prototypes": int,
}
```

## ColorMNet Integration

### Key differences from Deep Exemplar:
1. **Memory-based propagation** - Uses memory bank for temporal consistency
2. **DINOv2 features** - Better semantic understanding than VGG19
3. **Local attention** - More efficient than global non-local operations
4. **Adaptive memory** - Long-term and short-term memory management

### Inference Pipeline:
```python
1. Initialize model and memory manager
2. Encode reference image features
3. For each frame:
   a. Encode frame features (DINOv2)
   b. Query memory bank
   c. Apply local attention
   d. Decode to LAB colorization
   e. Update memory bank
4. Post-process and return
```

## Migration from Deep Exemplar

### What stays:
- ComfyUI node structure (INPUT_TYPES, RETURN_TYPES, etc.)
- Color space handling (LAB)
- Output format (ComfyUI IMAGE tensors)

### What changes:
- Model architecture (ColorMNet instead of ColorVidNet)
- Feature extraction (DINOv2 instead of VGG19)
- Temporal consistency (Memory bank instead of optical flow)
- Dependencies (modern PyTorch 2.x)

### Backward compatibility:
- Same node categories
- Similar parameter names
- Compatible input/output formats
- Can coexist with old nodes initially

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Speed | 3-5 fps @ 512x768 | On RTX 3070 with FP16 |
| VRAM | 6-8 GB peak | For 50-frame sequences |
| Quality | NTIRE 2023 level | Competition-proven |
| Startup | < 10s | Model loading |
| Error recovery | 100% | No crashes |

## Testing Strategy

1. **Unit tests** - Core functionality
2. **Integration tests** - ComfyUI interaction
3. **Performance tests** - Speed/memory benchmarks
4. **Windows testing** - Specific platform issues
5. **VRAM testing** - Various GPU sizes (6GB, 8GB, 12GB, 24GB)

## Documentation

1. **User Guide** - How to use the nodes
2. **Installation Guide** - Windows-specific instructions
3. **Troubleshooting** - Common issues and solutions
4. **API Documentation** - For developers
5. **Examples** - Workflow JSON files

## License Compliance

**ColorMNet License: CC BY-NC-SA 4.0**
- ✅ Non-commercial use
- ✅ Attribution required
- ✅ Share-alike required
- ❌ Commercial use NOT allowed

Our implementation:
- Clear license notice in README
- Attribution to original authors
- Same license for our code
- Warning about commercial restrictions
