import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from colormnet_dataset.range_transform import im_normalization, im_mean, im_rgb2lab_normalization, ToTensor, RGB2Lab
from colormnet_dataset.reseed import reseed

import util.functional as F

class DAVISVidevoDataset(Dataset):
    """
    Works for DAVIS/Videvo/NVCC2023 training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, num_frames=3, max_num_obj=2, finetune=False):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}
        vid_list = sorted([vid for vid in os.listdir(self.im_root) if not vid.startswith('.')])
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted([img for img in os.listdir(os.path.join(self.im_root, vid)) if not img.startswith('.')])
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            # transforms.RandomGrayscale(0.05),
        ])

        patchsz = 448 # 224
        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((patchsz, patchsz), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((patchsz, patchsz), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            RGB2Lab(),
            ToTensor(),
            im_rgb2lab_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx]
                png_name = jpg_name.replace('.jpg', '.png')
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)

                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                # print('1', torch.max(this_im[:1,:,:]), torch.min(this_im[:1,:,:]))
                # print('2', torch.max(this_im[1:3,:,:]), torch.min(this_im[1:3,:,:]))
                # print('3', torch.max(this_im), torch.min(this_im));assert 1==0
                # print(this_im.size());assert 1==0

                this_gt = np.array(this_gt)

                this_im_l = this_im[:1,:,:]
                this_im_ab = this_im[1:3,:,:]
                # print(this_im_l.size(), this_im_ab.size());assert 1==0

                # images.append(this_im_l)
                # masks.append(this_im_ab)

                this_im_lll = this_im_l.repeat(3,1,1)
                images.append(this_im_lll)
                masks.append(this_im_ab)

            images = torch.stack(images, 0)
            # print(images.size());assert 1==0

            # target_objects = labels.tolist()
            break

        first_frame_gt = masks[0].unsqueeze(0)
        # print(first_frame_gt.size());assert 1==0

        info['num_objects'] = 2

        masks = np.stack(masks, 0)
        # print(np.shape(masks));assert 1==0


        cls_gt = masks

        # # Generate one-hot ground-truth
        # cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int)
        # first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int)
        # for i, l in enumerate(target_objects):
        #     this_mask = (masks==l)
        #     cls_gt[this_mask] = i+1
        #     first_frame_gt[0,i] = (this_mask[0])
        # cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]

        # print(info['num_objects'], self.max_num_obj, selector);assert 1==0

        selector = torch.FloatTensor(selector)
        
        # print(images.size(), np.shape(first_frame_gt), np.shape(cls_gt));assert 1==0
        ### torch.Size([8, 3, 384, 384]) torch.Size([1, 2, 384, 384]) (8, 2, 384, 384)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
