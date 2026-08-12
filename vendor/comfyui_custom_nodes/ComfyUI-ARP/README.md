# ComfyUI-ARP

ARP-owned ComfyUI integration nodes. This package keeps pipeline-specific LTX
optimizations separate from the bundled upstream `ComfyUI-LTXVideo` package.

`ARPLTXVideoOnlyICLoRALoader` is used only by ARP's outpainting graph. When it
executes, it creates a private model structure with exact partitioned guide
attention and bounded feed-forward evaluation, then removes LTXAV audio
transformer modules that are unreachable from ARP's video-only latent. Source
audio is remuxed separately. Shared ComfyUI and ComfyUI-LTXVideo classes and
functions are never replaced.

The `/arp/ltx-compatibility` endpoint validates the live capabilities and hook
signatures before ARP loads model weights. It fails with an actionable
compatibility error when those internals change. This is deliberately a
capability check, not a ComfyUI version pin. Merely starting ComfyUI or using
another LTX workflow does not install the execution patch.

Guide strengths retain their exact values. In particular, soft guide weights
such as `0.95` are not rounded or promoted to full-strength guides.

The following environment variables retain the tested defaults while allowing
hardware-specific tuning:

- `ARP_LTX_ATTN_QUERY_CHUNK_TOKENS` (default `4096`)
- `ARP_LTX_FF_CHUNK_TOKENS` (default `4096`)
- `ARP_LTX_VIDEO_ONLY_MEMORY_USAGE_FACTOR` (optional override; otherwise the
  upstream factor is retained, with `13` used only when upstream reports its
  current placeholder value)
