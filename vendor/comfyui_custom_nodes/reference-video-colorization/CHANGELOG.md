# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-13

### 🎉 Major Release - Dual Implementation

This release includes a complete rewrite with **two colorization methods** available side-by-side!

### Added

#### ColorMNet Implementation (New)
- ✨ **ColorMNet Video Colorization Node** - Modern memory-based temporal colorization
- ✨ **ColorMNet Image Colorization Node** - Single image colorization
- 🚀 DINOv2-based feature extraction for superior quality
- ⚙️ Multiple memory modes (balanced, low_memory, high_quality)
- ⚡ FP16 mixed precision support for faster processing
- 📊 Performance reports with timing and FPS metrics
- 🎯 Automatic model download (~500MB DINOv2 model)
- 📈 Real-time progress bars in ComfyUI interface

#### Deep Exemplar Implementation (Enhanced Original)
- ✨ **Deep Exemplar Video Colorization Node** - Original CVPR 2019 method
- ✨ **Deep Exemplar Image Colorization Node** - Classic exemplar-based approach
- 📊 Performance reports with timing and FPS metrics
- 📈 Real-time progress bars in ComfyUI interface
- 🎯 Automatic model download for all checkpoints
- 🔧 Enhanced WLS filtering with configurable parameters

#### Infrastructure
- 📦 Automatic checkpoint downloading for all models
- 🔄 Lazy loading of VGG19 models to prevent import-time errors
- 🐛 Fixed relative import issues across all modules
- 📁 Proper extraction of Deep Exemplar checkpoint ZIP files
- 🎨 Example workflow with both methods for comparison
- 📸 Workflow screenshot for README
- 📚 Comprehensive documentation updates

### Fixed

- 🐛 ColorMNet greyscale output issue - proper RGB reference passing
- 🐛 ColorMNet color saturation - normalized reference ab channels
- 🐛 Progress bar not working in ColorMNet nodes
- 🐛 Import errors with absolute vs relative paths
- 🐛 VGG19 model loading at module import time
- 🐛 Deep Exemplar checkpoint extraction path issues
- 🐛 Missing auto-download for Deep Exemplar models

### Changed

- 📝 Complete README rewrite with both implementations
- 🏗️ Reorganized project structure for clarity
- 🎯 Node display names clarified: "(New)" for ColorMNet, "(Original)" for Deep Exemplar
- 🔧 Improved error messages and logging
- ⚡ Optimized memory usage and performance

### Technical Details

#### ColorMNet Fixes
1. **Reference Image Processing** - Changed from grayscale (L×3) to full RGB
2. **Color Space Normalization** - Proper LAB ab channel normalization (÷110)
3. **Output Denormalization** - Correct scaling from [-1,1] to LAB range (×110)
4. **Progress Updates** - Added `set_progress()` calls for UI updates

#### Deep Exemplar Improvements
1. **Auto-download System** - Downloads from GitHub releases automatically
2. **ZIP Extraction** - Proper handling of nested directory structure
3. **VGG Loading** - Lazy initialization with correct absolute paths
4. **Import Fixes** - All relative imports (`.` and `..`) instead of absolute

#### Performance Reports
All four nodes now output optional STRING reports containing:
- Processing time and FPS
- Frame count and resolution
- Configuration parameters used
- Enable/disable status of features

### Performance

**Example Benchmarks (768x432, 240 frames):**
- ColorMNet (balanced, FP16): ~5.3 FPS, 45s total
- Deep Exemplar (half-res, propagation): ~4.6 FPS, 52s total

## [1.0.0] - 2024-XX-XX

### Initial Release

- Initial Deep Exemplar implementation
- Basic video and image colorization nodes
- Manual model download required
- WLS filtering support

---

## Migration Guide

### From 1.x to 2.0

**Good News:** Both old and new implementations are available!

**Changes:**
1. **Node Names**: Original nodes renamed to "Deep Exemplar ... (Original)"
2. **New Nodes**: "ColorMNet ... (New)" nodes added
3. **Outputs**: All nodes now have optional `performance_report` output
4. **Auto-download**: Models download automatically on first use

**Existing Workflows:**
- Will continue to work with Deep Exemplar nodes
- Consider trying ColorMNet for comparison
- Performance reports are optional (can be ignored)

**Recommended Actions:**
1. Update ComfyUI custom node
2. Try the new example workflow
3. Compare ColorMNet vs Deep Exemplar quality
4. Review performance reports to optimize settings

---

## Future Roadmap

### Planned Features
- [ ] Batch processing optimization
- [ ] Memory-efficient streaming mode
- [ ] Additional colorization algorithms
- [ ] Color palette transfer utilities
- [ ] Integration with temporal smoothing nodes

### Under Consideration
- [ ] Web UI for reference image search
- [ ] Automatic reference selection
- [ ] Multi-reference blending
- [ ] Training script for custom models
- [ ] Real-time preview mode

---

## Version Support

| Version | Status | Support Until | Notes |
|---------|--------|---------------|-------|
| 2.0.x | ✅ Active | Ongoing | Current release |
| 1.0.x | ⚠️ Legacy | 2025-12-31 | Security fixes only |

---

## Breaking Changes

### 2.0.0
- Node display names changed (internal IDs unchanged)
- New output socket added (backwards compatible)
- Model paths changed (auto-downloaded)

**Migration:** Update nodes in workflow to see new names. Existing connections work unchanged.

---

## Contributors

Thanks to everyone who contributed to this release!

- ColorMNet integration and bugfixes
- Deep Exemplar enhancements
- Documentation improvements
- Testing and feedback

---

For detailed technical changes, see the [commit history](https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization/commits/main).
