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

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', default='./assets/video/sample_bw.mp4', help='video target')
    parser.add_argument('--ref_path', default='./assets/video/ref', help='color reference images')
    parser.add_argument('--output', default='./assets/video/sample_bw_cmnet2.mp4', help='Colorized output')
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

    print("--- Loading CMNET2 model ---")
    colorizer = ColorMNetRender(vid_length=total_frames, enable_resize=False, encode_mode=1,
                                max_memory_frames=total_frames, reset_on_ref_update=False, project_dir=package_dir)

    print("Preloading references...")
    for f in refs:
        ref_path = os.path.join(args.ref_path, f)
        ref = Image.open(ref_path).convert('RGB')
        colorizer.preload_reference(ref)
    print(f"perm_mem size: {colorizer.processor.memory.perm_mem.size}")
    first_ref = Image.open(os.path.join(args.ref_path, refs[0])).convert('RGB')

    print(f"--- [VIDEO] Saving {total_frames} color frames ---")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # on the first frame, pass the first reference normally
        if i == 0:
            colorizer.set_ref_frame(first_ref)
        else:
            colorizer.set_ref_frame(None)
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
