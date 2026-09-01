import os
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from skimage import color
from argparse import ArgumentParser
from pathlib import Path
import sys

# Ensures that local modules are found
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

# configuring torch
import torch

from colormnet.colormnet_render import ColorMNetRender

def cv2_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def compute_process_size(orig_w: int, orig_h: int, max_side: int) -> tuple[int, int]:
    """
    Computes the new (W, H) while maintaining the aspect ratio,
    so that the longest side equals max_side.
    The resulting dimensions are always rounded to the nearest even number.
    Returns the original dimensions if max_side <= 0 or
    if the frame is already smaller than max_side.
    """
    if max_side <= 0:
        return orig_w, orig_h
    longest = max(orig_w, orig_h)
    if longest <= max_side:
        return orig_w, orig_h
    scale = max_side / longest
    new_w = int(round(orig_w * scale / 2)) * 2
    new_h = int(round(orig_h * scale / 2)) * 2
    return new_w, new_h


def chroma_transfer(original_bgr: np.ndarray, colorized_small: Image.Image) -> np.ndarray:
    """
    Transfers the UV (color) channels from the reduced colorized frame
    to the original full-size frame using CV2/YUV.
    Returns np.ndarray BGR ready for out_video.write().
    """
    h, w = original_bgr.shape[:2]

    # upscale the colorized frame to the original size
    colorized_bgr = pil_to_cv2(colorized_small)
    colorized_full = cv2.resize(colorized_bgr, (w, h),
                                 interpolation=cv2.INTER_LANCZOS4)

    # convert both to YUV
    orig_yuv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YUV)
    col_yuv  = cv2.cvtColor(colorized_full, cv2.COLOR_BGR2YUV)

    # transfer only U and V from the colorized frame
    orig_yuv[:, :, 1] = col_yuv[:, :, 1]  # U
    orig_yuv[:, :, 2] = col_yuv[:, :, 2]  # V

    # convert back to BGR
    return cv2.cvtColor(orig_yuv, cv2.COLOR_YUV2BGR)


def compute_window_size_auto(colorizer, refs: list, ref_path: str,
                              proc_w: int, proc_h: int, do_resize: bool,
                              vram_threshold: float = 0.20, max_window_size: int = 99) -> int:
    """
    Preloads references one at a time until free VRAM drops below vram_threshold (default 20%) of total,
    or number of loaded reference frames is above max_window_size (default 99).
    Returns the number of loaded frames.
    """
    gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
    vram_limit = round(round(gpu_mem_total / 1024 / 1024, 1) * vram_threshold,0)
    loaded = 0
    for f in refs:
        gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
        gpu_mem_k = round(gpu_mem_free / 1024 / 1024, 1)
        if gpu_mem_k < vram_limit or loaded > max_window_size:
            break
        ref = Image.open(os.path.join(ref_path, f)).convert('RGB')
        if do_resize:
            ref = ref.resize((proc_w, proc_h), Image.LANCZOS)
        colorizer.preload_reference(ref)
        loaded += 1
        if loaded % 10 == 0:
            torch.cuda.empty_cache()  # empty cache from tensors not more referenced
    return loaded


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', default='./assets/video_full/sample_bw_full.mp4', help='video target')
    parser.add_argument('--ref_path', default='./assets/video_full/ref', help='color reference images')
    parser.add_argument('--output', default='./assets/video_full/sample_bw_full_cmnet2.mp4', help='Colorized output')
    parser.add_argument('--max_side', type=int, default=512,
        help='Resize longest side to this value before colorization. '
             '-1 = original resolution (no resize).')
    parser.add_argument('--window_size', type=int, default=40,
        help='Max reference frames in permanent memory. '
             '-1 or 0 = auto (fills until 20%% VRAM free).')
    parser.add_argument('--top_k', type=int, default=30,
        help='Top-K for memory matching softmax (default 30, try 10-15 for speed).')
    parser.add_argument('--mem_every', type=int, default=5,
        help='Store a frame in working memory every N frames (default 5, try 10 for speed).')
    args = parser.parse_args()

    torch.hub.set_dir(model_dir)

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proc_w, proc_h = compute_process_size(width, height, args.max_side)
    do_resize = (proc_w != width or proc_h != height)
    print(f"Frame size: {width}x{height} → processing at {proc_w}x{proc_h}")

    if args.window_size <= 0:
        AUTO_WINDOW = True
        SLIDE_STEP = None  # will be computed later
    else:
        AUTO_WINDOW = False
        WINDOW_SIZE = args.window_size
        SLIDE_STEP = max(1, round(WINDOW_SIZE * 0.2 + 0.5))
        print(f"Window size: {WINDOW_SIZE} refs, slide step: {SLIDE_STEP}")

    # MP4 Direct
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    refs = sorted([f for f in os.listdir(args.ref_path) if f.lower().endswith(('.png', '.jpg'))])
    refs_id = sorted(int(''.join(filter(str.isdigit, f))) for f in refs if any(c.isdigit() for c in f))

    print("--- Loading CMNET2 model ---")
    colorizer = ColorMNetRender(vid_length=total_frames, encode_mode=1, max_memory_frames=total_frames,
                                reset_on_ref_update=False, top_k=args.top_k, mem_every=args.mem_every,
                                project_dir=package_dir)

    # phase 1: preload the first WINDOW_SIZE references
    print("Preloading references...")
    if AUTO_WINDOW:
        refs_loaded = compute_window_size_auto(
            colorizer, refs, args.ref_path, proc_w, proc_h, do_resize)
        WINDOW_SIZE = refs_loaded
        SLIDE_STEP = max(1, round(WINDOW_SIZE * 0.2 + 0.5))
        print(f"Auto window size: {WINDOW_SIZE} refs, slide step: {SLIDE_STEP}")
    else:
        refs_loaded = 0
        for f in refs[:WINDOW_SIZE]:
            ref = Image.open(os.path.join(args.ref_path, f)).convert('RGB')
            if do_resize:
                ref = ref.resize((proc_w, proc_h), Image.LANCZOS)
            colorizer.preload_reference(ref)
            refs_loaded += 1

    refs_queue_idx = refs_loaded
    print(f"perm_mem frames: {colorizer.get_perm_mem_frame_count()}")

    # phase 2: colorize frames with sliding window
    first_ref = Image.open(os.path.join(args.ref_path, refs[0])).convert('RGB')

    print(f"--- [VIDEO] Saving {total_frames} color frames ---")
    t_encode = 0.0
    t_colorize = 0.0
    t_chroma = 0.0
    t_write = 0.0
    N_profile = 50  # measure on the first 50 frames

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # sliding window (unchanged)
        if refs_queue_idx < len(refs):
            oldest_ref_still_needed = refs_id[refs_queue_idx - WINDOW_SIZE + SLIDE_STEP]
            if i > oldest_ref_still_needed:
                colorizer.slide_permanent_memory(SLIDE_STEP)
                for f in refs[refs_queue_idx:refs_queue_idx + SLIDE_STEP]:
                    ref = Image.open(os.path.join(args.ref_path, f)).convert('RGB')
                    if do_resize:
                        ref = ref.resize((proc_w, proc_h), Image.LANCZOS)
                    colorizer.preload_reference(ref)
                refs_queue_idx += SLIDE_STEP

        t0 = time.perf_counter()
        if do_resize:
            rgb_frame_proc = cv2_to_pil(cv2.resize(frame, (proc_w, proc_h),
                                         interpolation=cv2.INTER_LANCZOS4))
        else:
            rgb_frame_proc = cv2_to_pil(frame)
        t_encode += time.perf_counter() - t0

        if i == 0:
            first_ref_proc = first_ref.resize((proc_w, proc_h), Image.LANCZOS) if do_resize else first_ref
            colorizer.set_ref_frame(first_ref_proc)
        else:
            colorizer.set_ref_frame(None)

        t0 = time.perf_counter()
        img_color_proc = colorizer.colorize_frame(ti=i, frame_i=rgb_frame_proc, lab_mode="gpu")
        torch.cuda.synchronize()  # essential: wait for the GPU to finish
        t_colorize += time.perf_counter() - t0

        t0 = time.perf_counter()
        if do_resize:
            final_bgr = chroma_transfer(frame, img_color_proc)
        else:
            final_bgr = pil_to_cv2(img_color_proc)
        t_chroma += time.perf_counter() - t0

        t0 = time.perf_counter()
        out_video.write(final_bgr)
        t_write += time.perf_counter() - t0

        if i == N_profile - 1:
            print(f"\n--- Profiling first {N_profile} frames ---")
            print(f"  encode+resize CPU : {t_encode:.2f}s  ({t_encode/N_profile*1000:.1f}ms/frame)")
            print(f"  colorize GPU      : {t_colorize:.2f}s  ({t_colorize/N_profile*1000:.1f}ms/frame)")
            print(f"  chroma transfer   : {t_chroma:.2f}s  ({t_chroma/N_profile*1000:.1f}ms/frame)")
            print(f"  write video       : {t_write:.2f}s  ({t_write/N_profile*1000:.1f}ms/frame)")
            print(f"  TOTAL             : {(t_encode+t_colorize+t_chroma+t_write)/N_profile*1000:.1f}ms/frame")
            print(f"  Estimated FPS     : {N_profile/(t_encode+t_colorize+t_chroma+t_write):.2f}")

    cap.release()
    out_video.release()
    print(f"\n--- [DONE] Video saved to: {args.output} ---")

if __name__ == '__main__':
    main()
