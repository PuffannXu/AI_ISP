import os
import pandas as pd
from torch import Tensor
import cv2
import numpy as np
from AF.code.core.settings import ROI_SIZE
import torchvision.transforms as standard_transforms
from PIL import Image
import torchvision.transforms.functional as F
from PIL.Image import Image
from scipy.spatial.distance import jensenshannon
from torch import Tensor
from torch.nn.functional import interpolate
import torch
from typing import Union, List, Tuple
import math
from core.settings import DEVICE
def log_metrics(train_loss: float, val_loss: float, current_metrics: dict, best_metrics: dict, path_to_log: str):
    log_data = pd.DataFrame({
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "best_mean": best_metrics["mean"],
        "best_median": best_metrics["median"],
        "best_trimean": best_metrics["trimean"],
        "best_bst25": best_metrics["bst25"],
        "best_wst25": best_metrics["wst25"],
        "best_wst5": best_metrics["wst5"],
        **{k: [v] for k, v in current_metrics.items()}
    })
    header = log_data.keys() if not os.path.exists(path_to_log) else False
    log_data.to_csv(path_to_log, mode='a', header=header, index=False)


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst5"], best_metrics["wst5"]))


def scale(x: np.ndarray) -> np.ndarray:
    """ Scales all values of a tensor between 0 and 1 """
    x = x - x.min()
    x = x / x.max()
    return x


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    """ Converts an image from height x width x channels to channels x height x width """
    return x.transpose(2, 0, 1)


def cal_ig(img):
    ig = cv2.Sobel(img, cv2.CV_8U, 1, 1)
    ig = np.mean(ig)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
        tmp = 0
    
    for i in range(rows):
        for j in range(cols):
            if i == 0:
                dx = 2 * img[i, j] - img[i + 1, j]
            elif i == rows - 1:
                dx = 2 * img[i, j] - img[i - 1, j]
            else:
                dx = 2 * img[i, j] - img[i - 1, j] - img[i + 1, j]
            if j == 0:
                dy = 2 * img[i, j] - img[i, j + 1]
            elif j == cols - 1:
                dy = 2 * img[i, j] - img[i, j - 1]
            else:
                dy = 2 * img[i, j] - img[i, j - 1] - img[i, j + 1]
            ds = (dx * dx + dy * dy) ** 0.5
            tmp += ds
    ig = tmp / rows / cols
    '''

    return ig


def cal_la(img):
    imageVar = cv2.Laplacian(img, cv2.CV_8U).var()
    return imageVar


def resize(img, new_size):

    old_height, old_width = img.shape[0], img.shape[1]
    new_height, new_width = new_size[0], new_size[1]
    scale_height, scale_width = new_height / old_height, new_width / old_width

    scaled_img = np.zeros((new_height, new_width), dtype="uint16")

    for y in range(new_height):
        for x in range(new_width):
            y_nearest = int(np.floor(y / scale_height))
            x_nearest = int(np.floor(x / scale_width))
            scaled_img[y, x] = img[y_nearest, x_nearest]
    return scaled_img


def crop(img, pos_x, pos_y, roi_size):
    # pos_x是x方向上的比例
    # pos_y是y方向上的比例
    # 左上角为0,0
    # transform1 = standard_transforms.ToTensor()
    # transform2 = standard_transforms.ToPILImage()
    # img = transform2(img)
    x_max = img.shape[0]
    y_max = img.shape[1]
    point_x = int(pos_x * x_max)
    point_y = int(pos_y * y_max)
    left = int(point_x - roi_size / 2)
    up = int(point_y - roi_size / 2)
    right = int(point_x + roi_size / 2)
    down = int(point_y + roi_size / 2)

    roi_img = img[left:right, up:down]
    # roi_img = transform1(roi_img)
    return roi_img


def depth2focus(depth):
    focus_distance = [3910.92, 2289.27, 1508.71, 1185.83, 935.91, 801.09, 700.37, 605.39, 546.23, 486.87,
                      447.99, 407.40, 379.91, 350.41, 329.95, 307.54, 291.72, 274.13, 261.53, 247.35,
                      237.08, 225.41, 216.88, 207.10, 198.18, 191.60, 183.96, 178.29, 171.69, 165.57,
                      160.99, 155.61, 150.59, 146.81, 142.35, 142.35, 134.99, 131.23, 127.69, 124.99,
                      121.77, 118.73, 116.40, 113.63, 110.99, 108.47, 106.54, 104.23, 102.01]
    min = 4000
    min_index = 0
    for index in range(49):
        diff = abs(depth - focus_distance[index])
        if diff < min:
            min = diff
            min_index = index
    return min_index, min


def correct(img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).to(DEVICE)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(normalized_img.squeeze(), mode="RGB")


def linear_to_nonlinear(img: Union[np.array, Image, Tensor]) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / 2.2))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / 2.2)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")


def normalize(img: np.ndarray) -> np.ndarray:
    max_int = 65535.0
    return np.clip(img, 0.0, max_int) * (1.0 / max_int)


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def bgr_to_rgb(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1]


def rescale(x: Tensor, size: Tuple) -> Tensor:
    """ Rescale tensor to image size for better visualization """
    return interpolate(x, size, mode='bilinear')


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle).item()


def tvd(pred: Tensor, label: Tensor) -> Tensor:
    """
    Total Variation Distance (TVD) is a distance measure for probability distributions
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """
    return (Tensor([0.5]) * torch.abs(pred - label)).sum()


def jsd(p: List, q: List) -> float:
    """
    Jensen-Shannon Divergence (JSD) between two probability distributions as square of scipy's JS distance. Refs:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    - https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    return jensenshannon(p, q) ** 2


