import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import time
from AF.code.core.utils import scale,hwc_to_chw
import cv2
save_data = False


class Dataset(data.Dataset):

    def __init__(self, rgb_channel = 3, basepath="/home/project/xupf/Databases/RGBD/numpy"):

        self.rgb_channel = rgb_channel
        self.__path_to_rgb = os.path.join(basepath, "rgb_images")
        self.__path_to_label = os.path.join(basepath, "labels")
        self.__path_to_depth = os.path.join(basepath, "depths")
        self.__path_to_rawDepth = os.path.join(basepath, "rawDepths")

        metadata = []
        for filename in os.listdir(self.__path_to_depth):
                metadata.append(filename)

        self.__fold_data = metadata
        self.rgb_images, self.depths, self.rawDepths, self.filenames = self.__load_dataset__()

    def __getitem__(self, index: int) -> Tuple:
        rgb_images = self.rgb_images[index]
        depths = self.depths[index]
        rawDepths = self.rawDepths[index]
        filenames = self.filenames[index]
        return rgb_images, depths, rawDepths, filenames
    def __load_dataset__(self):
        rgb_images = []
        file_names = []
        depths = []
        rawDepths = []
        for index in range(len(self.__fold_data)):
            file_name = self.__fold_data[index].strip().split('_')[-1]
            rgb_image = np.array(np.load(os.path.join(self.__path_to_rgb, 'rgb_image_{}'.format(file_name))), dtype='uint8')
            #label = np.array(np.load(os.path.join(self.__path_to_label, 'label_{}'.format(file_name))), dtype='uint8')
            depth = np.array(np.load(os.path.join(self.__path_to_depth, 'depth_{}'.format(file_name))), dtype='uint8')
            rawDepth = np.array(np.load(os.path.join(self.__path_to_rawDepth, 'rawDepth_{}'.format(file_name))), dtype='uint8')
            rgb_image = hwc_to_chw(rgb_image)
            if self.rgb_channel == 1:
                rgb_image = rgb_image[1,:,:]
                rgb_image = rgb_image.reshape(1, 256, 256)
            #rgb_image = rgb_image / 255
            #label = label / 255

            '''
            x = np.linspace(-0.5, 0.5, 256)
            x_grid, y_grid = np.meshgrid(x, x, indexing='ij')

            x_grid = np.expand_dims(x_grid, axis=0)
            y_grid = np.expand_dims(y_grid, axis=0)

            rgb_image = np.concatenate([rgb_image, x_grid, y_grid], axis=0)
            '''
            #label = label.reshape(1, 256, 256)
            depth = depth[:,:,0].reshape(1, 256, 256)

            rawDepth = rawDepth[:,:,0].reshape(1, 256, 256)
            rgb_images.append(rgb_image)
            #labels.append(label)
            depths.append(depth)
            rawDepths.append(rawDepth)
            file_names.append(file_name)
        #labels = labels / np.max(labels)
        rgb_images = rgb_images / np.max(rgb_images)
        depths = depths /np.max(depths)
        rawDepths = rawDepths/np.max(rawDepths)
        print("rgb max=",np.amax(rgb_images))
        print("depths max=", np.amax(depths))
        print("rawDepths max=", np.amax(rawDepths))
        return rgb_images, depths, rawDepths, file_names

    def __len__(self) -> int:
        return len(self.__fold_data)
