"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/8/18 11:03
 @Author : Pufan Xu
 @Function : This is the network for RGBD Fusion, based on IFCNN.
======================================================================================
"""
'''---------------------------------------------------------------------------
IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
----------------------------------------------------------------------------'''
import torch
from torch import nn, Tensor
import RRAM.my_utils as my
from typing import Union, Tuple, Any
from core.Model import Model
import torch.nn.functional as F
from RGBD.code.ssim_loss import SSIM, MS_SSIM
import numpy as np
from core.utils import cal_la

class Model(Model):
    def __init__(self,
                 qn_on: bool, version: int, rgb_channel:int,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                ):
        super().__init__()
        if version == 1:
            self._network = Net_V1(qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 2:
            self._network = Net_V2(rgb_channel = rgb_channel,qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 3:
            self._network = Net_V3(rgb_channel = rgb_channel,qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 4:
            self._network = Net_V4(rgb_channel = rgb_channel,qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        else:
            self._network = PixTransformNet(qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
    def predict(self, rgb_image: Tensor, rawDepth: Tensor) -> Union[Tensor, Tuple]:
        return self._network(rgb_image, rawDepth)

    def get_loss(self, pred_depth: Tensor, depth: Tensor) -> Union[Tensor, Tuple]:
        ssim_loss = MS_SSIM()
        l1_loss = nn.L1Loss()
        pred_depth = pred_depth.to(torch.float32)
        depth = depth.to(torch.float32)
        l1_out = l1_loss(pred_depth.to(self._device), depth.to(self._device))
        ssim_out = ssim_loss(pred_depth.to(self._device), depth.to(self._device))
        a = 0.85#
        final_loss = a/2*(1-ssim_out)+(1-a)*l1_out
        return final_loss
    def old_get_loss(self, pred_depth: Tensor, depth: Tensor) -> Union[Tensor, Tuple]:
        loss = nn.MSELoss()
        # loss_s = self.gradient_1order(pred_depth)
        #loss = nn.SmoothL1Loss()
        pred_depth = pred_depth.to(torch.float32)
        #pred_label = pred_label.to(torch.float32)
        depth = depth.to(torch.float32)
        #label = label.to(torch.float32)
        pred_depth = pred_depth.squeeze()
        #pred_label = pred_label.squeeze()
        depth = depth.squeeze()
        #label = label.squeeze()
        depth_mse = loss(pred_depth.to(self._device), depth.to(self._device))
        #label_mse = loss(pred_label.to(self._device), label.to(self._device))
        #rmse = mse**0.5
        #pred_depth = pred_depth.cpu().detach().numpy()

        return depth_mse**0.5 #, label_mse
    def optimize(self, rgb_image: Tensor, rawDepth: Tensor, depth: Tensor) -> tuple[Any, Any]:
        self._optimizer.zero_grad()
        pred_depth, _ = self.predict(rgb_image, rawDepth)
        #pred = torch.stack(pred)
        #label = torch.stack(label)
        loss = self.get_loss(pred_depth, depth)#, label)
        loss.backward()
        #label_mse.backward()
        self._optimizer.step()
        return pred_depth, loss.cpu().detach().numpy()



class Net_V1(nn.Module):
    # 基于IFCNN
    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        self.conv1 = conv2d(in_channels=3,   out_channels=64,   kernel_size=3, stride=1, padding=1, bias=False)
        self.conv11 = conv2d(in_channels=1,   out_channels=64,   kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv2d(in_channels=64,   out_channels=64,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv2d(in_channels=64,  out_channels=64,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d(in_channels=64,  out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, rgb_image: Tensor, rawDepth: Tensor) ->Union[Tensor,tuple]:


        x1 = rgb_image
        x2 = rawDepth
        a = {}
        # 第1层
        a['input_rgb_image'] = x1
        x1 = self.conv1(x1)
        a['conv1_no_relu'] = x1
        x1 = self.relu1(x1)
        a['conv1_relu'] = x1
        # 第2层CNN
        x1 = self.conv2(x1)
        a['conv2_no_relu'] = x1
        x1 = self.relu2(x1)
        a['conv2_relu'] = x1

        a['input_rgb_image'] = x2
        x2 = self.conv11(x2)
        a['conv1_no_relu'] = x2
        x2 = self.relu1(x2)
        a['conv1_relu'] = x2
        # 第2层CNN
        x2 = self.conv2(x2)
        a['conv2_no_relu'] = x2
        x2 = self.relu2(x2)
        a['conv2_relu'] = x2


        x = x1+x2

        # 第3层
        x = self.conv3(x)
        a['conv3_no_relu'] = x
        x = self.relu3(x)
        a['conv3_relu'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        out = self.relu4(x)
        a['conv4_relu'] = out

        return out, a

class Net_V2(nn.Module):

    def __init__(self, rgb_channel: int,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        self.conv1 = conv2d(in_channels=rgb_channel, out_channels=32, kernel_size=3, stride=1, padding=1,
                            bias=False)
        self.conv2 = conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = conv2d(in_channels=257, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7 = conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = torch.nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
        # kernel =2
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, rgb_image: Tensor, rawDepth: Tensor) -> Union[Tensor, tuple]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """



        # x1 = rgb_image
        # x1 = rawDepth
        a = {}
        # Encoder
        # 第1层
        a['input_rgb_image'] = rgb_image
        rgb256 = self.conv1(rgb_image)
        a['conv1_no_relu'] = rgb256
        rgb256 = self.relu(rgb256)
        a['conv1_relu'] = rgb256
        rgb128 = self.pool(rgb256)
        a['pool1'] = rgb128
        # 第2层CNN
        rgb128 = self.conv2(rgb128)
        a['conv2_no_relu'] = rgb128
        rgb128 = self.relu(rgb128)
        a['conv2_relu'] = rgb128
        rgb64 = self.pool(rgb128)
        a['pool2'] = rgb64
        # 第3层
        rgb64 = self.conv3(rgb64)
        a['conv3_no_relu'] = rgb64
        rgb64 = self.relu(rgb64)
        a['conv3_relu'] = rgb64
        rgb32 = self.pool(rgb64)
        a['pool3'] = rgb32
        # 第4层
        rgb32 = self.conv4(rgb32)
        a['conv4_no_relu'] = rgb32
        rgb32 = self.relu(rgb32)
        a['conv4_relu'] = rgb32

        d32 = self.pool(self.pool(self.pool(rawDepth)))
        x = torch.cat([rgb32, d32], dim=1)

        # 第5层
        x = self.conv5(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x

        # 第6层
        x = self.conv6(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x
        # 第7层
        x = self.conv7(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x
        # 第8层
        # x = x + rawDepth
        x = self.conv8(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x

        return x, a

class Net_V3(nn.Module):
    # 简单的encoder-decoder
    def __init__(self, rgb_channel: int,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        self.conv1 = conv2d(in_channels=rgb_channel+1,   out_channels=32,   kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = conv2d(in_channels=32,   out_channels=64,  kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = conv2d(in_channels=64,  out_channels=128,  kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = conv2d(in_channels=128,  out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7 = conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = torch.nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
  # kernel =2
        self.pool = nn.MaxPool2d(2, stride=2)
    def forward(self, rgb_image: Tensor, rawDepth: Tensor) ->Union[Tensor,tuple]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """


        x1 = torch.cat([rgb_image, rawDepth], dim=1)
        #x1 = rgb_image
        #x1 = rawDepth
        a = {}
        # 第1层
        a['input_cat_image'] = x1
        x1 = self.conv1(x1)
        a['conv1_no_relu'] = x1
        x1 = self.relu(x1)
        a['conv1_relu'] = x1
        x1 = self.pool(x1)
        a['pool1'] = x1
        # 第2层CNN
        x1 = self.conv2(x1)
        a['conv2_no_relu'] = x1
        x1 = self.relu(x1)
        a['conv2_relu'] = x1
        x1 = self.pool(x1)
        a['pool2'] = x1
        # 第3层
        x = self.conv3(x1)
        a['conv3_no_relu'] = x
        x = self.relu(x)
        a['conv3_relu'] = x
        x = self.pool(x)
        a['pool3'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x

        # 第5层
        x = self.conv5(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x

        # 第6层
        x = self.conv6(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x
        # 第7层
        x = self.conv7(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x
        # 第8层
        #x = x + rawDepth
        x = self.conv8(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x


        return x, a

class Net_V4(nn.Module):
    # encoder-decoder换成bottleneck
    def __init__(self, rgb_channel: int,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        def bottleneck(in_channels,out_channels):
            return nn.Sequential(
                conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0, bias=False),
                conv2d(in_channels=out_channels//4, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1, bias=False),
                conv2d(in_channels=out_channels//4, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), )
        def bottleneck_up(in_channels,out_channels):
            return nn.Sequential(
                conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0, bias=False),
                conv2d(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1, bias=False),
                conv2d(in_channels=in_channels//4, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), )
        self.conv1 = conv2d(in_channels=rgb_channel+1,   out_channels=32,   kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = bottleneck(32, 64)
        self.conv3 = bottleneck(64, 128)
        self.conv4 = bottleneck(128, 256)
        self.conv5 = bottleneck_up(256,128)
        self.conv6 = bottleneck_up(128,64)
        self.conv7 = bottleneck_up(64,32)
        self.conv8 = conv2d(in_channels=33, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = torch.nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
  # kernel =2
        self.pool = nn.MaxPool2d(2, stride=2)
    def forward(self, rgb_image: Tensor, d256: Tensor) ->Union[Tensor,tuple]:

        x1 = torch.cat([rgb_image, d256], dim=1)

        a = {}
        # 第1层
        a['input_cat_image'] = x1
        x1 = self.conv1(x1)
        a['conv1_no_relu'] = x1
        x1 = self.relu(x1)
        a['conv1_relu'] = x1
        x1 = self.pool(x1)
        a['pool1'] = x1
        # 第2层CNN
        x1 = self.conv2(x1)
        a['conv2_no_relu'] = x1
        x1 = self.relu(x1)
        a['conv2_relu'] = x1
        x1 = self.pool(x1)
        a['pool2'] = x1
        # 第3层
        x = self.conv3(x1)
        a['conv3_no_relu'] = x
        x = self.relu(x)
        a['conv3_relu'] = x
        x = self.pool(x)
        a['pool3'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x



        # 第5层
        x = self.conv5(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x


        # 第6层
        x = self.conv6(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x
        # 第7层
        x = self.conv7(x)
        a['conv4_no_relu'] = x
        x = self.relu(x)
        a['conv4_relu'] = x
        x = self.unpool(x)
        a['unpool1'] = x
        # 第8层

        x = torch.cat([x, d256], dim=1)
        x = self.conv8(x)
        a['conv4_no_relu'] = x


        return x, a
class PixTransformNet(nn.Module):

    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float,  channels_in=4, kernel_size=1, weights_regularizer=None, ):
        super(PixTransformNet, self).__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride, padding):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=False),
            )

        self.channels_in = channels_in

        self.spatial_net = nn.Sequential(conv2d(1, 32, 1, stride=1, padding=0),nn.ReLU(),
                                         conv2d(32, 64, 1, stride=1, padding=0), nn.ReLU(),
                                         conv2d(64, 128, 1, stride=1, padding=0), nn.ReLU(),
                                         #conv2d(128, 256,kernel_size, stride=1, padding=(kernel_size - 1) // 2)
                                         )
        self.color_net = nn.Sequential(conv2d(channels_in - 1, 32, 1, stride=2, padding=0), nn.ReLU(),  #nn.MaxPool2d(2,2),
                                       conv2d(32, 64, 1, stride=2, padding=0), nn.ReLU(),  #nn.MaxPool2d(2, 2),
                                       conv2d(64, 128, 1, stride=2, padding=0), nn.ReLU(),  # nn.MaxPool2d(2, 2),
                                       #conv2d(128, 256, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
                                       )
        self.head_net = nn.Sequential(nn.ReLU(),
                                      conv2d(256, 128, kernel_size, stride=1, padding=(kernel_size - 1) // 2),nn.ReLU(),nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
                                      conv2d(128, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.ReLU(),nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
                                      conv2d(64, 32, kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.ReLU(),nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
                                      conv2d(32, 1, 1, stride=1, padding=0))

        if weights_regularizer is None:
            reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]

        self.params_with_regularizer = []
        self.params_with_regularizer += [{'params': self.spatial_net.parameters(), 'weight_decay': reg_spatial}]
        self.params_with_regularizer += [{'params': self.color_net.parameters(), 'weight_decay': reg_color}]
        self.params_with_regularizer += [{'params': self.head_net.parameters(), 'weight_decay': reg_head}]

    def forward(self, input_color, input_spatial):

        #input_spatial = input[:, self.channels_in - 1:, :, :]
        #input_color = input[:, 0:self.channels_in - 1, :, :]
        a = []
        input_spatial=F.interpolate(input_spatial,scale_factor=1/8)
        merged_features = torch.cat([self.spatial_net(input_spatial) , self.color_net(input_color)],dim=1)

        return self.head_net(merged_features), a