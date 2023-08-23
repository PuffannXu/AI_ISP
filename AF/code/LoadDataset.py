import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import time
from core.utils import scale,hwc_to_chw
from torch.utils.data import DataLoader
import cv2
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
        img512 = np.array(np.load(os.path.join(self.__path_to_data, file_name + 'data.npy')), dtype='uint16')
        focus_step = np.array(np.load(os.path.join(self.__path_to_label, file_name + 'label.npy')), dtype='float')
        scaled_img512 = scale(img512)
        #fig, axs = plt.subplots(1,2)
        #axs[0].imshow(scaled_img512[:,:,0])
        scaled_img256 = cv2.resize(scaled_img512, dsize = (256,256), fx=0.5, fy=0.5)#resize(scaled_img512, (256,256,2))
        #axs[1].imshow(scaled_img256[:,:,0])
        #fig.show()
        #pd_img = pd_img.reshape(-1, 256, 256)
        chw_scaled_img256 = hwc_to_chw(scaled_img256)
        #pd_img = pd_img.astype(float)
        pd_img = torch.from_numpy(chw_scaled_img256.copy())
        focus_step = torch.from_numpy(focus_step.copy())
        focus_step = focus_step.squeeze()



        return pd_img, focus_step, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
