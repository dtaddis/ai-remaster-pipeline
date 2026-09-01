import os
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

# sliding window parameters
WINDOW_SIZE = 6      # max reference frame in perm_mem
SLIDE_STEP = 3        # how many to remove/add at a time

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', default='./assets/video_slide/sample_bw.mp4', help='video target')
    parser.add_argument('--ref_path', default='./assets/video_slide/ref', help='color reference images')
    parser.add_argument('--output', default='./assets/video_slide/sample_bw_slide_cmnet2.mp4', help='Colorized output')
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

    # MP4 Direct
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    refs = sorted([f for f in os.listdir(args.ref_path) if f.lower().endswith(('.png', '.jpg'))])
    refs_id = sorted(int(''.join(filter(str.isdigit, f))) for f in refs if any(c.isdigit() for c in f))

    print("--- Loading CMNET2 model ---")
    colorizer = ColorMNetRender(vid_length=total_frames, enable_resize=False, encode_mode=1,
                                max_memory_frames=total_frames, reset_on_ref_update=False, project_dir=package_dir)

    # phase 1: preload the first WINDOW_SIZE references
    print("Preloading references...")
    refs_loaded = 0
    for f in refs[:WINDOW_SIZE]:
        ref = Image.open(os.path.join(args.ref_path, f)).convert('RGB')
        colorizer.preload_reference(ref)
        refs_loaded += 1
    refs_queue_idx = refs_loaded  # index of the next ref to load

    print(f"perm_mem frames: {colorizer.get_perm_mem_frame_count()}")

    # phase 2: colorize frames with sliding window
    first_ref = Image.open(os.path.join(args.ref_path, refs[0])).convert('RGB')

    print(f"--- [VIDEO] Saving {total_frames} color frames ---")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # sliding: when the current frame exceeds the frame_id
        # of the (refs_queue_idx - WINDOW_SIZE + SLIDE_STEP)-th reference
        # it means the first SLIDE_STEP references are "in the past"
        if refs_queue_idx < len(refs):
            oldest_ref_still_needed = refs_id[refs_queue_idx - WINDOW_SIZE + SLIDE_STEP]
            if i > oldest_ref_still_needed:
                colorizer.slide_permanent_memory(SLIDE_STEP)
                for f in refs[refs_queue_idx:refs_queue_idx + SLIDE_STEP]:
                    ref = Image.open(os.path.join(args.ref_path, f)).convert('RGB')
                    colorizer.preload_reference(ref)
                refs_queue_idx += SLIDE_STEP
        # first frame: pass the ref normally to initialize work_mem
        colorizer.set_ref_frame(first_ref if i == 0 else None)

        # Frame
        rgb_frame = cv2_to_pil(frame)
        img_color = colorizer.colorize_frame(ti=i, frame_i=rgb_frame, lab_mode="gpu")
        final_bgr = pil_to_cv2(img_color)
        out_video.write(final_bgr)

    cap.release()
    out_video.release()
    print(f"\n--- [DONE] Video saved to: {args.output} ---")

if __name__ == '__main__':
    main()
