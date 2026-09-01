import os
import torch
import numpy as np
import cv2
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

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', default='./assets/image/image_bw.jpg', help='Target image (grayscale or color, L channel will be extracted)')
    parser.add_argument('--ref', default='./assets/image/image_color_ref.jpg', help='Color reference image')
    parser.add_argument('--output', default='./assets/image/image_bw_cmnet2.jpg', help='Colorized output')
    args = parser.parse_args()

    torch.hub.set_dir(model_dir)

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    # 1. Caricamento immagini
    ref_raw = Image.open(args.ref).convert('RGB')
    target_raw = Image.open(args.input).convert('RGB')
    if ref_raw is None or target_raw is None:
        print("❌ Error loading images. Please check the paths.")
        return

    print("--- Loading CMNET2 model ---")
    colorizer = ColorMNetRender(vid_length=1, enable_resize=False, encode_mode=1, max_memory_frames=100,
                                reset_on_ref_update=False, project_dir=package_dir)

    colorizer.set_ref_frame(ref_raw, False)
    img_color = colorizer.colorize_frame(ti=0, frame_i=target_raw, lab_mode="gpu")

    img_color.save(args.output)
    img_color.show()

if __name__ == '__main__':
    main()
