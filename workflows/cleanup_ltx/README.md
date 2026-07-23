# LTX 2.3 Dearchive Clean Up

`DeArchive.json` is the example ComfyUI workflow published with
[`oumoumad/ltx-2.3-dearchive-lora`](https://huggingface.co/oumoumad/ltx-2.3-dearchive-lora).

ARP patches it at runtime to use the configured ComfyUI model paths and the downloaded
`ltx-2.3-dearchive-lora.safetensors`. Clean Up processes model-safe chunks internally, then
normalizes the stitched result back to the input video's exact resolution, frame rate, and frame
count. Geometry changes remain the responsibility of later Outpainting and Upscaling phases.

The source clip is supplied as the workflow's full video IC-LoRA guide. ARP's **Source Fidelity**
setting controls that guide's `strength`: `1.0` retains the author's exact-control default, while
lower values allow the Dearchive model to replace damage that would otherwise be copied from the
input. This is independent of both the Dearchive LoRA strength and the optional single-frame i2v
guide used by Outpainting.
