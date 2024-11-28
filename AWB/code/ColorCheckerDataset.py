import os
from typing import Tuple
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import time
from core.utils import normalize, bgr_to_rgb, linear_to_nonlinear, hwc_to_chw, resize
from core.DataAugmenter import DataAugmenter

import cv2
save_data = False
class ColorCheckerDataset(data.Dataset):

    def __init__(self, channel_number=4,height=224,width=224,train: bool = True, folds_num: int = 1):

        self.__train = train
        self.__da = DataAugmenter()
        self.channel_number = channel_number
        self.height = height
        self.width = width
        path_to_folds = os.path.join("/home/project/xupf/Databases/AWB", "folds.mat")
        path_to_metadata = os.path.join("/home/project/xupf/Databases/AWB", "metadata.txt")
        self.__path_to_data = os.path.join("/home/project/xupf/Databases/AWB", "preprocessed", "numpy_data")
        self.__path_to_label = os.path.join("/home/project/xupf/Databases/AWB", "preprocessed", "numpy_labels")
        self.__path_to_save = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/output/vis", "{}".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time())))))

        folds = scipy.io.loadmat(path_to_folds)
        #img_idx = [1]
        img_idx = folds["tr_split" if self.__train else "te_split"][0][folds_num][0]

        metadata = open(path_to_metadata, 'r').readlines()
        self.__fold_data = [metadata[i - 1] for i in img_idx]

    def __getitem__(self, index: int) -> Tuple:
        file_name = self.__fold_data[index].strip().split(' ')[1]
        img = np.array(np.load(os.path.join(self.__path_to_data, file_name + '.npy')), dtype='uint16')
        illuminant = np.array(np.load(os.path.join(self.__path_to_label, file_name + '.npy')), dtype='float32')
        #img = img.float()
        #img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        img = resize(img, (self.height, self.width))
        img = hwc_to_chw((bgr_to_rgb(normalize(img)*255)))
        if(self.channel_number==4):
            img_rggb = np.empty((4,self.height,self.width),dtype = np.uint8)
            img_rggb[0,:,:] = img[0,:,:]
            img_rggb[1, :, :] = img[1, :, :]
            img_rggb[2, :, :] = img[1, :, :]
            img_rggb[3, :, :] = img[2, :, :]
            img_copy = torch.from_numpy((img_rggb / 1).copy())
        elif (self.channel_number == 3):
            img_copy = torch.from_numpy((img / 1).copy())
        else:
            img_copy = torch.from_numpy((img[1, :, :].copy()))
        #img = hwc_to_chw((bgr_to_rgb(img)))
        if save_data:
            unaugment_linear_bgr_image = torch.from_numpy(normalize(img).copy())
            cropped_linear_bgr_image = torch.from_numpy(normalize(img).copy())
            cropped_linear_rgb_image = torch.from_numpy(bgr_to_rgb(normalize(img)).copy())
            cropped_nonlinear_rgb_image = torch.from_numpy(linear_to_nonlinear(bgr_to_rgb(normalize(img))).copy())
            path_to_save = os.path.join(self.__path_to_save, "fold_{}".format(0), file_name)
            os.makedirs(path_to_save, exist_ok=True)
            fig, axs = plt.subplots(1, 4)
            axs[0].imshow(cropped_linear_bgr_image)
            axs[0].set_title("linear_bgr")
            axs[1].imshow(cropped_linear_rgb_image)
            axs[1].set_title("linear_rgb")
            axs[2].imshow(cropped_nonlinear_rgb_image)
            axs[2].set_title("nonlinear_rgb")
            axs[3].imshow(unaugment_linear_bgr_image)
            axs[3].set_title("unaugment")
            fig.suptitle("Image ID: {} | gt: {}".format(file_name, illuminant))
            fig.show()
            fig.savefig(os.path.join(path_to_save, "pre_img.png"), dpi=200)

        illuminant = torch.from_numpy(illuminant.copy())
        if not self.__train:
            img_copy = img_copy.type(torch.FloatTensor)
        return img_copy, illuminant, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
