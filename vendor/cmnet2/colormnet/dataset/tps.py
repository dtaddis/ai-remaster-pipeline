import numpy as np
from PIL import Image
import cv2
# import thinplate as tps

cv2.setNumThreads(0)

def pick_random_points(h, w, n_samples):
    y_idx = np.random.choice(np.arange(h), size=n_samples, replace=False)
    x_idx = np.random.choice(np.arange(w), size=n_samples, replace=False)
    return y_idx/h, x_idx/w
