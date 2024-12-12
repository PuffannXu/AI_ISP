"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2024/11/27 9:44
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载 .pth 文件
model_path = 'AWB_FULL.pth'
model_weights = torch.load(model_path)
weight_bit = 8
bin_num = 2 << weight_bit -1

alpha_q = - (2 << (weight_bit -1))
beta_q = (2 << (weight_bit -1)) - 1

# 遍历每一层的权重
for layer_name, weights in model_weights.items():
    if isinstance(weights, torch.Tensor):
        # 转换为 NumPy 数组
        weights_np = weights.cpu().numpy()

        # 计算统计信息
        max_val = np.max(weights_np)
        min_val = np.min(weights_np)
        mean_val = np.mean(weights_np)
        std_val = np.std(weights_np)

        print(f"Layer: {layer_name}")
        print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}, Std: {std_val}")

        # 绘制整体权重直方图
        plt.figure(figsize=(10, 5))
        plt.hist(weights_np.flatten(), bins=bin_num, alpha=0.7, color='blue')
        plt.title(f'Overall Distribution of weights in layer: {layer_name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # # 如果是卷积层或线性层，按输出通道绘制
        # if len(weights_np.shape) >= 2:
        #     num_output_channels = weights_np.shape[0]
        #     for i in range(num_output_channels):
        #         channel_weights = weights_np[i].flatten()
        #
        #         # 计算每个通道的统计信息
        #         max_val_channel = np.max(channel_weights)
        #         min_val_channel = np.min(channel_weights)
        #         mean_val_channel = np.mean(channel_weights)
        #         std_val_channel = np.std(channel_weights)
        #
        #         print(f"Layer: {layer_name}, Channel: {i}")
        #         print(f"Max: {max_val_channel}, Min: {min_val_channel}, Mean: {mean_val_channel}, Std: {std_val_channel}")
        #
        #         # 绘制每个通道的权重直方图
        #         plt.figure(figsize=(10, 5))
        #         plt.hist(channel_weights, bins=bin_num, alpha=0.7, color='green')
        #         plt.title(f'Distribution of weights in layer: {layer_name}, Channel: {i}')
        #         plt.xlabel('Weight Value')
        #         plt.ylabel('Frequency')
        #         plt.grid(True)
        #         plt.show()
