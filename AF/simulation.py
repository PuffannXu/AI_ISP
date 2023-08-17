"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : ${DATE} ${TIME}
 @Author : Pufan Xu
 @Function : 用于仿真自动对焦流程，对于一个场景，随机初始化最初的对焦点，预测对焦点移动位置并移到下一
             个对焦点。如果RMESLoss<5，则认为对焦成功；否则若对焦次数>5，则对焦失败。
======================================================================================
"""

import os
from random import random
import numpy as np
import torch
import time
from AF.code.core.utils import scale, hwc_to_chw
from AF.model.AFNet import Model
from RRAM import my_utils as my
from AF.code.core.settings import DEVICE
from AF.code.core.Evaluator import Evaluator

# ====================================================================================
# 定义常量
# ====================================================================================
qn_on = True # 权重及特征图量化加噪on
in_q_on = 1  # 输入量化on
input_bit = 8   # 输入比特
weight_bit = 4  # 权重比特
output_bit = 8  # 输出比特
noise_scale = 0.075 # 噪声系数
clamp_std = 0   # 权重阶段
isint = 0       # 量化为整型
#
version = 2     # 网络版本，2为包含全连接的版本，与AF更相似，效果更好；1为全卷积版
#
MAX_FOCUS_TIME = 7 # 最大对焦次数，文献中为7
NEARNESS_THRESHOLD = 40 # 误差范围内视为成功，文献中为7,5,3,1
POSITION_ERROR = 20

# ====================================================================================
# 定义路径
# ====================================================================================

DATASET = "top_0.5_0.5"
# 输出结果
PATH_TO_SAVED = os.path.join("/home/project/xupf/Projects/AI_ISP/AF/output/test_{}".format(DATASET),
                             "PositionError_{}_{}".format(POSITION_ERROR, str(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))))
os.makedirs(PATH_TO_SAVED, exist_ok=True)
# 场景元数据
#path_to_metadata = "/home/project/xupf/Databases/AF_new/best_of_a_scene.csv"
path_to_metadata = "/home/project/xupf/Databases/AF_{}/{}.csv".format(DATASET,DATASET)
# 输入数据
path_to_data = os.path.join("/home/project/xupf/Databases/AF_{}", "numpy_data").format(DATASET)
# 权重
path_to_pretrained = os.path.join("/home/project/xupf/Projects/AI_ISP/AF/output/train/I8W4O8_n0.075/model_V2.pth")


# ====================================================================================
# 网络初始化
# ====================================================================================

model = Model(qn_on=qn_on, version=version, weight_bit=weight_bit, output_bit=output_bit, isint=isint,
              clamp_std=clamp_std, noise_scale=noise_scale)
model.load(path_to_pretrained)
model.evaluation_mode()


# ====================================================================================
# Main Function
# ====================================================================================

# 每一次预测的结果存入test_interval_results.csv
with open(os.path.join(PATH_TO_SAVED, 'test_interval_results.csv'), 'a') as f:
    print("filename", "CorrectPosition", "focus_time", "index", "label", "pred", "loss", sep=',', end='\n', file=f)

# 整体的结果存入test_total_results.csv
with open(os.path.join(PATH_TO_SAVED, 'test_total_results.csv'), 'a') as f:
    print("Filename", "Correct Position", "Initial Position", "Success or Not", "Final Position Error", "Number of Movements",
          sep=',', end='\n', file=f)

# 读入数据进行处理
metadata = open(path_to_metadata, 'r').readlines()

InputNumber = 0
SuccessTimes = 0
FinalPositionErrorList = []
MovementsTimesList=[]
InitialPositionErrorList = []
for r in range(len(metadata)):
    Filename, median_depth, CorrectPosition, min_diff = metadata[r].strip().split(',')
    # 如果是标题行，跳过
    if median_depth == "median_depth":
        continue
    median_depth = float(median_depth)
    CorrectPosition = int(CorrectPosition)
    # 如果没有文件，跳过
    try:
        img = np.array(np.load(os.path.join(path_to_data, Filename + '_{}_{}_data.npy'.format(CorrectPosition,DATASET))),
                       dtype='uint16')
    except FileNotFoundError:
        continue
    else:
        InputNumber += 1
        # 随机初始化对焦点[0,48]
        InitialPosition = round(random() * 48)
        #InitialPosition = CorrectPosition + POSITION_ERROR
        if InitialPosition > 48:
            InitialPosition = 48
        if InitialPosition < 0:
            InitialPosition = 0
        InitialPositionErrorList.append(InitialPosition - CorrectPosition)
        CurrentPosition = InitialPosition
        for focus_time in range(MAX_FOCUS_TIME):
            try:
                img = np.array(
                    np.load(os.path.join(path_to_data, Filename + '_{}_{}_data.npy'.format(CurrentPosition, DATASET))),
                    dtype='uint16')
            except FileNotFoundError:
                continue

            label = CorrectPosition - CurrentPosition

            img = hwc_to_chw(scale(img))
            img = torch.from_numpy(img.copy())
            img = img.unsqueeze(0)
            label = torch.from_numpy(np.array([label]))
            label = label.squeeze()

            img, label = img.to(DEVICE), label.to(DEVICE)
            if in_q_on == 1:
                img, _ = my.data_quantization_sym(img, half_level=2 ** input_bit / 2 - 1)
                img = img.float()
            pred, a = model.predict(img, return_steps=True)
            loss = model.get_loss(pred, label).item()
            # 中间结果
            print('\t - Input: {} - label: {} | pred: {} | Loss: {:f}'.format(Filename, label, pred, loss))
            with open(os.path.join(PATH_TO_SAVED, 'test_interval_results.csv'), 'a') as f:
                print(Filename, CorrectPosition, focus_time+1, CurrentPosition, int(label), float(pred), loss, sep=',', end='\n', file=f)
            CurrentPosition = CurrentPosition + round(float(pred))
            if CurrentPosition < 0:
                CurrentPosition = 0
            if CurrentPosition > 48:
                CurrentPosition = 48
            if loss < NEARNESS_THRESHOLD:
                MovementsTimesList.append(focus_time+1)   # 只考虑成功的移动次数
                FinalPositionErrorList.append(loss)     # 只考虑成功的误差
                SuccessTimes += 1
                break
        # 每一张图的结果
        Success = loss < NEARNESS_THRESHOLD
        print('\t ==========================================================')
        print('\t Input: {} - CorrectPosition: {} | InitialPosition: {}'.format(Filename, CorrectPosition, InitialPosition))
        print('\t Success: {} - focus_time: {} | Loss: {:f}'.format(Success, focus_time+1, loss))
        print('\t ==========================================================')
        with open(os.path.join(PATH_TO_SAVED, 'test_total_results.csv'), 'a') as f:
            print(Filename, CorrectPosition, InitialPosition, Success, loss, focus_time+1, sep=',', end='\n', file=f)

# 最终结果
SuccessRate = SuccessTimes/InputNumber
MeanInitialPositionError = np.mean(InitialPositionErrorList)
MeanError = np.mean(FinalPositionErrorList)
MeanMovement = np.mean(MovementsTimesList)
print('\t =============================== FINAL RESULTS ===================================')
print("InputNumber", InputNumber, "MeanInitialPosition", MeanInitialPositionError,
      "SuccessRate", SuccessRate, "MeanError", MeanError, "MeanMovement", MeanMovement)
with open(os.path.join(PATH_TO_SAVED, 'test_total_results.csv'), 'a') as f:
    print("InputNumber", InputNumber, "MAX_FOCUS_TIME", MAX_FOCUS_TIME, "NEARNESS_THRESHOLD", NEARNESS_THRESHOLD,
          "MeanInitialPositionError", MeanInitialPositionError,"SuccessRate", SuccessRate, "MeanError", MeanError, "MeanMovement", MeanMovement,
          sep=',', end='\n', file=f)