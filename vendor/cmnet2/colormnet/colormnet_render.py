"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-14
version:
LastEditors: Dan64
LastEditTime: 2026-01-17
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
ColorMNet rendering class.
"""
import os
from os import path
import torch
import gc
import warnings
import torch.backends.cudnn as cudnn
from skimage import color
from torchvision import transforms
from torchvision.transforms import InterpolationMode
# import torch.nn.functional as Ff
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math

from colormnet.dataset.range_transform import im_normalization, im_rgb2lab_normalization, ToTensor, RGB2Lab
from colormnet.model.network import ColorMNet
from colormnet.inference.inference_core import InferenceCore
from colormnet.util.transforms import lab2rgb_transform_PIL

DEF_MAX_MEMORY_FRAMES = 1000

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ColorMNetRender:
    """
    This class is used to render a frame at a time
    """
    _instance = None
    _initialized = False
    _frame_size = None
    ref_img: Image = None
    ref_img_valid: Image = None
    reset_on_ref_update = False
    img: Image = None
    first_mask_loaded: bool = False
    max_memory_frames: int = None
    frame_count: int = 0
    ref_count: int = 0
    ref_count_prv: int = 0
    total_colored_frames = 0
    processor: InferenceCore = None
    encode_mode: int = None  # 0: remote, 1: async, 2: sync

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, image_size: int = -1, vid_length: int = None, enable_resize: bool = False,
                 encode_mode: int = None, propagate: bool = False, max_memory_frames: int = None,
                 reset_on_ref_update: bool = True, top_k: int = 30, mem_every: int = 5,
                 project_dir: str = None, model_path: str = None):

        self.reset_on_ref_update = reset_on_ref_update  # deprecated with XMem2
        self.top_k = top_k
        self.mem_every = mem_every
        self.enable_resize = enable_resize
        if project_dir is None:
            project_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = project_dir
        self.model_path = model_path
        self._frame_size = image_size
        if encode_mode is None:
            self.encode_mode = 0
        else:
            self.encode_mode = encode_mode
        if max_memory_frames is None or max_memory_frames == 0:
            self.max_memory_frames = min(DEF_MAX_MEMORY_FRAMES, vid_length)
        else:
            self.max_memory_frames = min(DEF_MAX_MEMORY_FRAMES, max_memory_frames)
        self.total_colored_frames = 0

        self._colorize_config_init(image_size, vid_length, propagate)

        if not self._initialized:
            self._colorize_model_init(vid_length)
            self._initialized = True

    def _colorize_config_init(self, image_size: int = -1, vid_length: int = 100, propagate: bool = False):
        """
                        size - resize min. side to size. Does nothing if <0.
                        Resize the shorter side to this size. -1 to use original resolution.
       """

        cudnn.benchmark = True
        torch.autograd.set_grad_enabled(False)

        self.config = {}
        # model checkpoint location
        self.config['model'] = self.model_path or path.join(
            self.project_dir, 'weights/DINOv2FeatureV6_LocalAtten_s2_154000.pth'
        )
        # Whether the provided reference frame is exactly the first input frame
        self.config['FirstFrameIsNotExemplar'] = not propagate
        # dataset setting
        # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
        self.config['dataset'] = 'D16_batch'  # D16/D17/Y18/Y19/LV1/LV3/G
        # Long-term memory options
        self.config['max_mid_term_frames'] = min(10, vid_length)  # T_max in paper, decrease to save memory
        self.config['min_mid_term_frames'] = min(5, int(
            self.config['max_mid_term_frames'] / 2))  # T_min in paper, decrease to save memory
        self.config[
            'max_long_term_elements'] = self.max_memory_frames  # LT_max in paper, increase if objects disappear for a long time
        self.config['num_prototypes'] = 128  # P in paper
        self.config['top_k'] = self.top_k
        self.config['mem_every'] = min(self.mem_every, self.config[
            'max_mid_term_frames'])  # r in paper. Increase to improve running speed
        self.config['deep_update_every'] = -1  # Leave -1 normally to synchronize with mem_every
        # Multi-scale options
        self.config['save_scores'] = False
        self.config['size'] = image_size  # Resize the shorter side to this size. -1 to use original resolution
        self.config['disable_long_term'] = False
        self.config['enable_long_term'] = not self.config['disable_long_term']

        if image_size < 0:
            self.im_transform = transforms.Compose([
                RGB2Lab(),
                ToTensor(),
                im_rgb2lab_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = image_size
        self.ref_img = None
        self.img = None

    def _colorize_model_init(self, vid_length: int = 100):

        cudnn.benchmark = True
        torch.autograd.set_grad_enabled(False)

        # Model setup
        self.network = ColorMNet(self.config, self.config['model']).cuda().eval()
        self.model_weights = torch.load(self.config['model'])
        self.network.load_weights(self.model_weights, init_as_zero_if_needed=True)
        self.vid_length = vid_length
        self.config['enable_long_term_count_usage'] = (
                self.config['enable_long_term'] and
                (self.vid_length
                 / (self.config['max_mid_term_frames'] - self.config['min_mid_term_frames'])
                 * self.config['num_prototypes'])
                >= self.config['max_long_term_elements']
        )
        self.processor = InferenceCore(self.network, config=self.config)

    def set_config(self, param_name: str = None, param_value: any = None):
        self.config[param_name] = param_value
        self.processor.update_config(self.config)

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = False):
        self.ref_img = frame_ref
        self.config['FirstFrameIsNotExemplar'] = not frame_propagate
        if not (frame_ref is None):
            self.ref_img_valid = frame_ref
            if self.frame_count > 0:
                self.ref_count_prv = self.ref_count
            else:
                self.ref_count_prv = 0
            self.ref_count = self.frame_count

    def preload_reference(self, ref_img: Image):
        """
        Preloads a reference frame into perm_mem before starting colorization.
        Can be called N times consecutively.
        """
        if self.processor is None:
            return

        img = self.im_transform(ref_img)
        img_l = img[:1, :, :]
        img_lll = img_l.repeat(3, 1, 1).cuda()
        img_ab = img[1:3, :, :].cuda()

        if self.processor.all_labels is None:
            self.processor.set_all_labels(list(range(1, 3)))

        self.processor.load_reference(img_lll, img_ab)

    def slide_permanent_memory(self, n_frames: int):
        """
        Removes the first n_frames reference frames from permanent memory.
        Call when perm_mem exceeds the desired window size.
        """
        if self.processor is not None:
            self.processor.memory.slide_permanent_memory(n_frames)

    def get_perm_mem_frame_count(self) -> int:
        """Returns the number of reference frames currently in perm_mem."""
        if self.processor is not None:
            return self.processor.memory._perm_frame_count
        return 0

    def reset_memory(self, vid_length: int = None):
        """Clear all three memory tiers while retaining the loaded network weights.

        ARP calls this at real shot boundaries so permanent references and propagated
        colours from one scene cannot leak into the next scene.
        """
        if vid_length is not None:
            self.vid_length = max(1, int(vid_length))
        self.total_colored_frames = 0
        self.frame_count = 0
        self.ref_count = 0
        self.ref_count_prv = 0
        self.first_mask_loaded = False
        self.ref_img = None
        self.ref_img_valid = None
        self.img = None
        self.processor = InferenceCore(self.network, config=self.config)

    def colorize_batch_frames(self, frame_list: list[Image] = None, ref_list: list[Image] = None,
                              frame_propagate: bool = False) -> list[Image]:
        nframes = len(frame_list)
        frames_colored = []
        for i in range(0, nframes, 1):
            frame_i = frame_list[i]
            ref_i = ref_list[i]
            self.set_ref_frame(ref_i, frame_propagate)
            col_i = self.colorize_frame(i, frame_i)
            frames_colored.append(col_i)
        return frames_colored

    def get_frame_count(self) -> int:
        return self.frame_count

    def colorize_frame(self, ti: int = None, frame_i: Image = None, lab_mode: str = "gpu") -> Image:

        self.total_colored_frames += 1

        gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
        gpu_mem_k = round(gpu_mem_free / 1024 / 1024, 1)
        # vs.core.log_message(vs.MESSAGE_TYPE_WARNING, "CUDA free memory: " + str(gpu_mem_k) + " MB")
        reset_cond_1 = (gpu_mem_k < 100)
        reset_cond_2 = (gpu_mem_k < 500)
        # if self.frame_count >= self.vid_length or self.frame_count >= self.max_memory_frames:
        if reset_cond_1:
            """
            if gpu_mem_k < 10:
               if self.encode_mode == 0:
                   warnings.warn(f"Free memory at: {self.total_colored_frames}/{self.vid_length} -> {self.frame_count}/{self.max_memory_frames}")
               else:
                   vs.core.log_message(vs.MESSAGE_TYPE_WARNING, f"Free memory at: {self.total_colored_frames}/{self.vid_length} -> {self.frame_count}/{self.max_memory_frames}")
            """
            self.frame_count = 0
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
            self.config['FirstFrameIsNotExemplar'] = True  # because the reference image is the previous colored frame
            self.processor = InferenceCore(self.network, config=self.config)
            data = self.get_image(ti, frame_i, self.ref_img_valid)
        else:
            if reset_cond_2:
                # VRAM below threshold: aggressive slide of 70% of perm_mem
                n = int(self.get_perm_mem_frame_count() * 0.7)
                self.slide_permanent_memory(n)
                torch.cuda.empty_cache()
            data = self.get_image(ti, frame_i, self.ref_img)
            self.frame_count += 1

        rgb = data['rgb'].cuda()[0]

        msk = data.get('mask')
        if not self.config['FirstFrameIsNotExemplar']:
            msk = msk[:, 1:3, :, :] if msk is not None else None

        info = data['info']
        # frame = '{:0>5}'.format(info['frame'])
        shape = info['shape']
        need_resize = info['need_resize']

        if not self.first_mask_loaded:
            if msk is not None:
                self.first_mask_loaded = True
            else:
                # no point to do anything without a mask
                return frame_i

        # Map possibly non-continuous labels to continuous ones
        if msk is not None:
            msk = torch.Tensor(msk[0]).cuda()
            if need_resize:
                msk = self.resize_mask(msk.unsqueeze(0))[0]
            self.processor.set_all_labels(list(range(1, 3)))
            labels = range(1, 3)
        else:
            labels = None

        # Run the model on this frame
        is_last_frame = self.vid_length == self.total_colored_frames - 1  # (ti == (self.vid_length - 1))
        if self.config['FirstFrameIsNotExemplar']:

            if msk is None:
                prob = self.processor.step_AnyExemplar(rgb, None, None, labels, end=is_last_frame)
            else:
                prob = self.processor.step_AnyExemplar(rgb, msk[:1, :, :].repeat(3, 1, 1), msk[1:3, :, :], labels,
                                                       end=is_last_frame)
        else:
            prob = self.processor.step(rgb, msk, labels, end=is_last_frame)

        # Upsample to original size if needed
        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

        # return the colored frame
        out_img_final = lab2rgb_transform_PIL(torch.cat([rgb[:1, :, :], prob], dim=0), mode = lab_mode)
        out_img_final = out_img_final * 255
        out_img_final = out_img_final.astype(np.uint8)

        out_pil_img = Image.fromarray(out_img_final)

        self.save_last_image(out_pil_img)

        # empty torch cache
        torch.cuda.empty_cache()

        return out_pil_img

    def get_image(self, idx: int = None, img: Image = None, ref_img: Image = None) -> dict:

        shape = np.array(img).shape[:2]

        img = self.im_transform(img)
        img_l = img[:1, :, :]
        img_lll = img_l.repeat(3, 1, 1)

        data = {}
        info = {}

        if not (ref_img is None):
            mask = self.im_transform(ref_img)

            # keep L channel of reference image in case First frame is not exemplar
            # mask_ab = mask[1:3,:,:]
            # data['mask'] = mask_ab
            data['mask'] = torch.unsqueeze(mask, dim=0)

        info['shape'] = [torch.tensor(shape[0]), torch.tensor(shape[1])]
        info['need_resize'] = not (self.size < 0)
        info['frame'] = idx
        data['rgb'] = torch.unsqueeze(img_lll, dim=0)
        data['info'] = info

        return data

    def save_last_image(self, img: Image = None):
        self.img = img
        self.ref_img_valid = img

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h / min_hw * self.size), int(w / min_hw * self.size)),
                             mode='nearest')
