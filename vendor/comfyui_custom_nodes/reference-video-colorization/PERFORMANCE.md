Performance and GPU Support
===========================

- GPU detection is automatic; CUDA is used when available, with FP16 enabled on modern NVIDIA cards (Turing/RTX 20 series and newer, including RTX 50-series/5090).
- If CUDA dev tools are missing, the correlation CUDA extension falls back to the PyTorch implementation; quality is unchanged, speed is just a bit lower.
- Model download still occurs on first use (~500MB); all Python dependencies and git-based CUDA extensions are auto-installed when the node loads.
