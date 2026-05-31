import torchvision.transforms as transforms
import util.functional as F
import numpy as np
from skimage import color

im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

# tensor l[-1, 1]  ab[-1,    1]
# numpy  l[0 100]  ab[-127 128]
# transforms.Normalize: x_new = (x-mean) / std
inv_lll2rgb_trans = transforms.Normalize(
                mean=[-1, 0, 0],
                std=[1/50., 1/110., 1/110.])

im_rgb2lab_normalization = transforms.Normalize(
                mean=[50, 0, 0],
                std=[50, 110, 110])

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return F.to_mytensor(inputs)

class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        # default return float64
        # return color.rgb2lab(inputs)

        # return float32
        return np.float32(color.rgb2lab(inputs))