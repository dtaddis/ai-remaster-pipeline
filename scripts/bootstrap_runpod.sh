#!/usr/bin/env bash
set -euo pipefail

RUNTIME="${1:-/workspace/arp-runtime}"
STAGE="${2:-all}"
ROOT=/workspace/ai-remaster-pipeline
VENV="$RUNTIME/venv"
COMFY="$RUNTIME/ComfyUI"
CORE_STAMP="$RUNTIME/.arp-core-cuda13-v2"
OLD_STAMP="$RUNTIME/.arp-bootstrap-cuda13-v1"

declare -A NODES=(
  [ComfyUI-LTXVideo]=https://github.com/Lightricks/ComfyUI-LTXVideo.git
  [ComfyUI-GGUF]=https://github.com/city96/ComfyUI-GGUF.git
  [ComfyUI-VideoHelperSuite]=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
  [ComfyUI_ProPainter_Nodes]=https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git
  [ComfyUI-FlashVSR_Ultra_Fast]=https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
  [ComfyUI-SeedVR2_VideoUpscaler]=https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
  [ComfyUI-MMAudio]=https://github.com/kijai/ComfyUI-MMAudio.git
  [reference-video-colorization]=https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization.git
)

# Migrate workers created by the original all-purpose bootstrap without reinstalling them.
if [[ -f "$OLD_STAMP" && ! -f "$CORE_STAMP" ]]; then
  touch "$CORE_STAMP"
  for name in "${!NODES[@]}"; do
    touch "$RUNTIME/.arp-node-${name}-v1"
  done
fi

if [[ ! -f "$CORE_STAMP" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl git ffmpeg libgl1 libglib2.0-0 python3-venv build-essential
  # The CUDA 13 RunPod image already carries a tested GPU-enabled PyTorch build.
  # Let the persistent venv see those image packages instead of copying another
  # multi-gigabyte CUDA stack onto slower network storage.  The guarded fallback
  # keeps custom/older images working when they do not provide CUDA 13 themselves.
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
  if "$VENV/bin/python" -c 'import sys, torch; cuda = str(torch.version.cuda or ""); sys.exit(0 if torch.__version__.startswith("2.9.1+") and cuda.startswith("13.") and torch.cuda.is_available() else 1)'; then
    # Match the versions bundled by runpod/pytorch:*cu1300-torch291*.  Installing
    # the unpinned latest torchvision would make pip replace torch 2.9.1 and pull
    # the whole CUDA SDK into the portable volume again.
    "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu130 --no-deps \
      torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130
  else
    "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
  fi
  "$VENV/bin/pip" install -r "$ROOT/requirements.txt"
  if [[ ! -d "$COMFY/.git" ]]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
  fi
  "$VENV/bin/pip" install -r "$COMFY/requirements.txt"
  mkdir -p "$COMFY/custom_nodes"
  touch "$CORE_STAMP"
fi

case "$STAGE" in
  outpaint) required_nodes=(ComfyUI-LTXVideo ComfyUI-GGUF ComfyUI-VideoHelperSuite) ;;
  cleanup) required_nodes=(ComfyUI-LTXVideo ComfyUI-GGUF ComfyUI-VideoHelperSuite ComfyUI_ProPainter_Nodes) ;;
  references) required_nodes=(ComfyUI-GGUF) ;;
  colour) required_nodes=(ComfyUI-LTXVideo ComfyUI-GGUF ComfyUI-VideoHelperSuite reference-video-colorization) ;;
  audio) required_nodes=(ComfyUI-MMAudio ComfyUI-VideoHelperSuite) ;;
  upscale) required_nodes=(ComfyUI-FlashVSR_Ultra_Fast ComfyUI-SeedVR2_VideoUpscaler ComfyUI-VideoHelperSuite) ;;
  stabilize|shots|recomp) required_nodes=() ;;
  *) required_nodes=("${!NODES[@]}") ;;
esac

mkdir -p "$COMFY/custom_nodes"
for name in "${required_nodes[@]}"; do
  bundled_node="$ROOT/vendor/comfyui_custom_nodes/$name"
  node_stamp="$RUNTIME/.arp-node-${name}-v2"
  if [[ -d "$bundled_node" ]]; then
    # ARP's bundled copies carry the same compatibility fixes used by the Windows
    # installer (including the current ComfyUI/Kornia LTX fixes).  Prefer them to
    # an unpinned upstream HEAD so local and cloud renders use identical nodes.
    rm -rf "$COMFY/custom_nodes/$name"
    cp -a "$bundled_node" "$COMFY/custom_nodes/$name"
  fi
  if [[ ! -f "$node_stamp" ]]; then
    [[ -d "$COMFY/custom_nodes/$name" ]] || git clone --depth 1 "${NODES[$name]}" "$COMFY/custom_nodes/$name"
    [[ -f "$COMFY/custom_nodes/$name/requirements.txt" ]] && "$VENV/bin/pip" install -r "$COMFY/custom_nodes/$name/requirements.txt"
    touch "$node_stamp"
  fi
done

# This node ships with ARP itself, so refresh it on every job even when the much slower
# environment/model bootstrap has already been cached on the persistent volume.
rm -rf "$COMFY/custom_nodes/ComfyUI-ARP"
cp -a "$ROOT/vendor/comfyui_custom_nodes/ComfyUI-ARP" "$COMFY/custom_nodes/ComfyUI-ARP"

COMFY_PID=""
if ! curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  nohup "$VENV/bin/python" "$COMFY/main.py" --listen 127.0.0.1 --port 8188 --output-directory "$COMFY/output" >"$RUNTIME/comfyui.log" 2>&1 &
  COMFY_PID="$!"
fi

for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && exit 0
  if [[ -n "$COMFY_PID" ]] && ! kill -0 "$COMFY_PID" >/dev/null 2>&1; then
    echo "ComfyUI exited during startup; recent log output:" >&2
    tail -n 120 "$RUNTIME/comfyui.log" >&2 || true
    exit 1
  fi
  sleep 2
done
echo "ComfyUI did not become ready within ten minutes; recent log output:" >&2
tail -n 120 "$RUNTIME/comfyui.log" >&2 || true
exit 1
