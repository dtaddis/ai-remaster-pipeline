import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as Ff
from PIL import Image
import numpy as np

from colormnet.dataset.range_transform import im_normalization, im_rgb2lab_normalization, ToTensor, RGB2Lab

class VideoReader_221128_TransColorization(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        # print('use_all_mask', use_all_mask);assert 1==0
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = [img for img in sorted(os.listdir(self.image_dir)) if (img.endswith('.jpg') or img.endswith('.png')) and not img.startswith('.')]
        self.palette = Image.open(path.join(mask_dir, sorted([msk for msk in os.listdir(mask_dir) if not msk.startswith('.')])[0])).getpalette()
        self.first_gt_path = path.join(self.mask_dir, sorted([msk for msk in os.listdir(self.mask_dir) if not msk.startswith('.')])[0])
        self.suffix = self.first_gt_path.split('.')[-1]

        if size < 0:
            self.im_transform = transforms.Compose([
                RGB2Lab(),
                ToTensor(),
                im_rgb2lab_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = size


    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['vid_name'] = self.vid_name
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[idx]) if idx < len(os.listdir(self.mask_dir)) else None

        img = self.im_transform(img)
        img_l = img[:1,:,:]
        img_lll = img_l.repeat(3,1,1)

        load_mask = (self.use_all_mask or (gt_path == self.first_gt_path)) and gt_path is not None
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert('RGB')
            mask = self.im_transform(mask)

            # keep L channel of reference image in case First frame is not exemplar
            # mask_ab = mask[1:3,:,:]
            # data['mask'] = mask_ab
            data['mask'] = mask

        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        data['rgb'] = img_lll
        data['info'] = info

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return Ff.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)),
                    mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
