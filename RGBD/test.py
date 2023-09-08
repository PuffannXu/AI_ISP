"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/8/22 18:16
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from core.settings import DEVICE, make_deterministic
from core.utils import print_metrics, log_metrics
from core.Evaluator import Evaluator
from core.LossTracker import LossTracker
from RGBD.code.LoadDataset import Dataset
from RGBD.RGBDNet import Model
from RRAM import my_utils as my

# ======================================== #
# 量化训练参数
# ======================================== #
NUM_SAMPLES = 1

img_quant_flag = 1

qn_on = True
isint = 0
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0.075

version = 5
RELOAD_CHECKPOINT = True
DOWN_SCALE = 4
in_range = 1

PATH_TO_PTH_CHECKPOINT = os.path.join("/home/project/xupf/Projects/AI_ISP/RGBD/output/train/FULL/model_down{}_{}_V{}.pth".format(DOWN_SCALE,in_range,version))#/model/I8W4O8n0.075_model_V2.pth")#I8W4O8_n0.075
PATH_TO_SAVE = os.path.join("/home/project/xupf/Projects/AI_ISP/RGBD/output/test/I8W4O8_n0.075/")
os.makedirs(PATH_TO_SAVE, exist_ok=True)
model = Model(qn_on = qn_on, version=version, rgb_channel = 3,weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, noise_scale=noise_scale)

print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
model.load(PATH_TO_PTH_CHECKPOINT)


model.print_network()

test_set = Dataset(rgb_channel=3)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=10, drop_last=True)
print(" Test set size ....... : {}\n".format(len(test_set)))

model.evaluation_mode()

print("\n--------------------------------------------------------------")
print("\t\t\t Validation")
print("--------------------------------------------------------------\n")
val_loss = LossTracker()
with torch.no_grad():
    for i, (rgb_image, depth, rawDepth, file_name) in enumerate(test_loader):
        if NUM_SAMPLES > -1 and i >= NUM_SAMPLES:
            break
        rgb_image, depth, rawDepth = rgb_image.to(DEVICE), depth.to(DEVICE), rawDepth.to(DEVICE)
        # quant
        if img_quant_flag == 1:
            rgb_image, _ = my.data_quantization_sym(rgb_image, half_level=2 ** input_bit / 2 - 1)
            rgb_image = rgb_image.float()
            depth, _ = my.data_quantization_sym(depth, half_level=2 ** input_bit / 2 - 1)
            depth = depth.float()
            if DOWN_SCALE:
                down_rawDepth = torch.nn.functional.interpolate(rawDepth, scale_factor=1 / 4)
                rawDepth = torch.nn.functional.interpolate(down_rawDepth, scale_factor=4)

            rawDepth, _ = my.data_quantization_sym(rawDepth, half_level=2 ** input_bit / 2 - 1)
            rawDepth = rawDepth.float()
        for test_time in range(5):
            pred_depth, a = model.predict(rgb_image, rawDepth)
            loss = model.get_loss(pred_depth, depth).cpu().detach().numpy()
            raw_loss = model.get_loss(rawDepth, depth).cpu().detach().numpy()
            val_loss.update(loss)
            print("[ file name: {}] | Val loss: {:.4f} | improve loss: {:.4f}]".format(file_name, loss, raw_loss-loss))
            fig, axs = plt.subplots(2, 2)
            showrgb = rgb_image.cpu()[0, :, :, :].squeeze()
            axs[0,0].imshow(np.transpose(showrgb, (1, 2, 0))/showrgb.max(), cmap='gray')
            axs[0, 0].set_title("rgb")
            showraw = down_rawDepth.cpu()[0, :, :, :].squeeze()
            showpred = pred_depth.cpu()[0, :, :, :].squeeze()
            axs[0,1].imshow(showraw, cmap='gray')
            axs[0, 1].set_title("raw")
            showdepth = depth.cpu()[0, :, :, :].squeeze()
            axs[1,0].imshow(showdepth, cmap='gray')  # , cmap='gray')
            axs[1, 0].set_title("gt")
            axs[1,1].imshow(showpred.detach().numpy(), cmap='gray')  # , cmap='gray')
            axs[1, 1].set_title("pred")
            fig.suptitle("[ file name: {}] | Val loss: {:.4f} | improve loss: {:.4f}]".format(file_name, loss, raw_loss-loss))
            plt.axis('off')
            fig.show()
            fig.savefig(os.path.join(PATH_TO_SAVE,"{}_test{}.png".format(file_name,test_time)))
