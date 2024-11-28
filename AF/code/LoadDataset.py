import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import time
from core.utils import scale, hwc_to_chw, crop, resize
from torch.utils.data import DataLoader
import cv2
save_data = False


class Dataset(data.Dataset):

    def __init__(self, channel_number=4,height=224,width=224,basepath="/home/project/xupf/Databases/AF_new", txt_path = None):
        self.channel_number = channel_number
        self.height = height
        self.width = width
        #self.__path_to_data = os.path.join(basepath, "numpy_data")#("E:/Dataset/preprocessed", "numpy_data")#
        self.__path_to_label = os.path.join(basepath, "numpy_labels")#("E:/Dataset/preprocessed", "numpy_labels")#
        self.__path_to_data = os.path.join("/home/project/xupf/Databases/AF_train0_top", "visual_data")
        metadata = []
        if txt_path == None:
            for filename in os.listdir(self.__path_to_data):
                metadata.append(filename)
        else:
            metadata = open(txt_path, 'r').readlines()
        self.__fold_data = metadata
    def __getitem__(self, index: int) -> Tuple:

        file_name = self.__fold_data[index].strip().split('data')[0]
        id = self.__fold_data[index].strip().split('top')[1]
        id = id.strip().split('data')[0]
        img512 = np.array(np.load(os.path.join(self.__path_to_data, file_name + 'data.npy')), dtype='uint16')
        focus_step = np.array(np.load(os.path.join(self.__path_to_label, file_name.strip().split('_top')[0] + id + 'top_0.6_0.6_label.npy')), dtype='float')
        scaled_img512 = scale(img512)
        #fig, axs = plt.subplots(1,2)
        #axs[0].imshow(scaled_img512[:,:,0])
        scaled_img256 = resize(scaled_img512, (self.height, self.width))
        #scaled_img256 = cv2.resize(scaled_img512, dsize = (256,256), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)#resize(scaled_img512, (256,256,2))
        cropped_img = crop(scaled_img256, 0.6, 0.6, 256 / 4)
        #scaled_img256 = cv2.resize(cropped_img, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        #axs[1].imshow(scaled_img256[:,:,0])
        #fig.show()
        #pd_img = pd_img.reshape(-1, 256, 256)
        chw_scaled_img256 = hwc_to_chw(scaled_img256)
        if (self.channel_number == 4):
            img_rggb = np.empty((4, self.height, self.width), dtype=np.uint8)
            img_rggb[0, :, :] = chw_scaled_img256[0, :, :]
            img_rggb[1, :, :] = chw_scaled_img256[1, :, :]
            img_rggb[2, :, :] = chw_scaled_img256[1, :, :]
            img_rggb[3, :, :] = chw_scaled_img256[2, :, :]
            img_copy = torch.from_numpy((img_rggb / 1).copy())
        elif (self.channel_number == 3):
            img_copy = torch.from_numpy((chw_scaled_img256 / 1).copy())
        else:
            img_copy = torch.from_numpy((chw_scaled_img256[1, :, :].copy()))
        #pd_img = pd_img.astype(float)

        focus_step = torch.from_numpy(focus_step.copy())
        focus_step = focus_step.squeeze()

        return img_copy, focus_step, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
