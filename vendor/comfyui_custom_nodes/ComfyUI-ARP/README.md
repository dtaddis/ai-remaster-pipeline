# ComfyUI-ARP

ARP-owned ComfyUI integration nodes. This package keeps pipeline-specific LTX
optimizations separate from the bundled upstream `ComfyUI-LTXVideo` package.

`ARPLTXVideoOnlyICLoRALoader` is used only by ARP's outpainting graph. When it
executes, it lazily enables exact partitioned guide attention and bounded
feed-forward evaluation, then removes LTXAV audio transformer modules that are
unreachable from ARP's video-only latent. Source audio is remuxed separately.

The patch validates the ComfyUI hook signatures it depends on and fails with an
actionable compatibility error when those internals change. Merely starting
ComfyUI or using another LTX workflow does not install the execution patch.

The following environment variables retain the tested defaults while allowing
hardware-specific tuning:

- `ARP_LTX_ATTN_QUERY_CHUNK_TOKENS` (default `4096`)
- `ARP_LTX_FF_CHUNK_TOKENS` (default `4096`)
- `ARP_LTX_VIDEO_ONLY_MEMORY_USAGE_FACTOR` (default `13`)
