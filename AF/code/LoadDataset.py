import os
from typing import Tuple
import numpy as np
import torch
import torch.utils.data as data
import time
from AF.code.core.utils import scale,hwc_to_chw
from torch.utils.data import DataLoader

save_data = False


class Dataset(data.Dataset):

    def __init__(self, basepath="/home/project/xupf/Databases/AF_new", txt_path = None):

        self.__path_to_data = os.path.join(basepath, "numpy_data")#("E:/Dataset/preprocessed", "numpy_data")#
        self.__path_to_label = os.path.join(basepath, "numpy_labels")#("E:/Dataset/preprocessed", "numpy_labels")#

        metadata = []
        if txt_path == None:
            for filename in os.listdir(self.__path_to_data):
                metadata.append(filename)
        else:
            metadata = open(txt_path, 'r').readlines()
        self.__fold_data = metadata
    def __getitem__(self, index: int) -> Tuple:

        file_name = self.__fold_data[index].strip().split('data')[0]
        pd_img = np.array(np.load(os.path.join(self.__path_to_data, file_name + 'data.npy')), dtype='uint16')
        focus_step = np.array(np.load(os.path.join(self.__path_to_label, file_name + 'label.npy')), dtype='float')
        pd_img = hwc_to_chw(scale(pd_img))
        #pd_img = hwc_to_chw(pd_img)
        #pd_img = pd_img.astype(float)
        pd_img = torch.from_numpy(pd_img.copy())
        focus_step = torch.from_numpy(focus_step.copy())
        focus_step = focus_step.squeeze()

        return pd_img, focus_step, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
