
from dataset.range_transform import inv_im_trans, inv_lll2rgb_trans
from skimage import color, io
import cv2
import numpy as np

def detach_to_cpu(x):
    return x.detach().cpu()

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def lab2rgb_transform_PIL(mask):
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