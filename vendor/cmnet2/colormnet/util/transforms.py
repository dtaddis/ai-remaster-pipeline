
from colormnet.dataset.range_transform import inv_im_trans, inv_lll2rgb_trans
from skimage import color, io
import cv2
import numpy as np
import torch
import os
from PIL import Image

def detach_to_cpu(x):
    return x.detach().cpu()

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def lab2rgb_transform_PIL(mask, mode: str = "gpu"):

    if mode == "gpu":
        return lab2rgb_transform_PIL_gpu(mask)
    return lab2rgb_transform_PIL_cpu(mask)

def lab2rgb_transform_PIL_gpu(mask):
    """
    LAB->RGB conversion on GPU with PyTorch - exact CIE formulas.
    mask: torch tensor (3, H, W) in normalized range [-1, 1]
    """
    mask_d = inv_lll2rgb_trans(mask)

    L = mask_d[0:1, :, :]
    a = mask_d[1:2, :, :]
    b = mask_d[2:3, :, :]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps = 0.2069
    kappa = 7.787
    X = torch.where(fx > eps, fx ** 3, (fx - 16.0 / 116.0) / kappa)
    Y = torch.where(fy > eps, fy ** 3, (fy - 16.0 / 116.0) / kappa)
    Z = torch.where(fz > eps, fz ** 3, (fz - 16.0 / 116.0) / kappa)

    X = X * 0.95047
    Y = Y * 1.00000
    Z = Z * 1.08883

    r    =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g    = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_ch =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    eps_gamma = 0.0031308
    r    = torch.where(r    > eps_gamma, 1.055 * r    ** (1/2.4) - 0.055, 12.92 * r)
    g    = torch.where(g    > eps_gamma, 1.055 * g    ** (1/2.4) - 0.055, 12.92 * g)
    b_ch = torch.where(b_ch > eps_gamma, 1.055 * b_ch ** (1/2.4) - 0.055, 12.92 * b_ch)

    rgb = torch.cat([r, g, b_ch], dim=0)
    im = rgb.detach().cpu().numpy()
    im = im.transpose((1, 2, 0))
    return im.clip(0, 1).astype(np.float32)

def lab2rgb_transform_PIL_cpu(mask):
    flag_test = False

    mask_d = detach_to_cpu(mask)

    if flag_test: print('before inv', mask_d.size(), torch.max(mask_d), torch.min(mask_d))
    mask_d = inv_lll2rgb_trans(mask_d)
    if flag_test: print('after inv', mask_d.size(), torch.max(mask_d), torch.min(mask_d));assert 1==0

    im = tensor_to_np_float(mask_d)

    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    im = color.lab2rgb(im)

    return im.clip(0, 1)

def calculate_psnr(img1, img2):
    mse_value = ((img1 - img2)**2).mean()
    if mse_value == 0:
        result = float('inf')
    else:
        result = 20. * np.log10(255. / np.sqrt(mse_value))
    return result

def calculate_psnr_for_folder(gt_folder, result_folder):
    result_clips = sorted(os.listdir(result_folder))

    psnr_values = []
    for clip in result_clips:
        path_clip = os.path.join(result_folder, clip)
        test_files = sorted(os.listdir(path_clip))

        for img in test_files:
            gt_path = os.path.join(gt_folder, clip, img)
            result_path = os.path.join(path_clip, img)

            gt_img = np.array(Image.open(gt_path))
            result_img = np.array(Image.open(result_path))

            psnr = calculate_psnr(gt_img, result_img)
            psnr_values.append(psnr)

    avg_psnr = np.mean(psnr_values)
    return avg_psnr
