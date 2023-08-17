import os
import pandas as pd
from torch import Tensor
import cv2
import numpy as np
from AF.code.core.settings import ROI_SIZE
import torchvision.transforms as standard_transforms
from PIL import Image


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

