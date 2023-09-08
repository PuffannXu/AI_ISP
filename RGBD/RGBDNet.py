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
        if version == 3:
            self._network = Net_V3(rgb_channel = rgb_channel,qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 4:
            self._network = Net_V4(rgb_channel = rgb_channel,qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 5:
            self._network = Net_V5(qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                               isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 6:
            self._network = Net_V6(qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
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
        #print("l1:{},ssim:{}".format(l1_out, ssim_out))
        a = 0.85#
        final_loss = a/2*(1-ssim_out)+(1-a)*l1_out
        return final_loss
    def old_get_loss(self, pred_depth: Tensor, depth: Tensor) -> Union[Tensor, Tuple]:
        loss1 = nn.MSELoss(reduction='none')
        pred_depth = pred_depth.to(torch.float32)
        depth = depth.to(torch.float32)
        pred_depth = pred_depth.squeeze()
        depth = depth.squeeze()
        depth_mse1 = torch.div(loss1(pred_depth.to(self._device), depth.to(self._device)),depth.to(self._device))
        depth_mse1 = torch.mean(depth_mse1)

        depth_mse2 = F.mse_loss(pred_depth.to(self._device), depth.to(self._device))
        return depth_mse2**0.5
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
        self.conv7 = conv2d(in_channels=64, out_channels=29, kernel_size=3, stride=1, padding=1, bias=False)
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
        x = torch.cat([x, rgb_image], dim=1)
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
        x = self.relu(x)
        a['conv4_relu'] = x

        return x, a

class Net_V5(nn.Module):
    # 基于 CNN 的彩色图像引导的深度图像
    # 超分辨率重
    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
                #nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2,
#
 #                                     bias=bias),
            )

        self.conv1c = nn.Sequential(conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, bias=False),nn.ReLU())
        self.conv2c = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),nn.ReLU())
        self.conv1d = nn.Sequential(conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, bias=False),nn.ReLU())
        self.conv2d = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, bias=False),nn.ReLU())
        self.conv3i = nn.Sequential(conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, bias=False),nn.ReLU())
        self.conv4i = nn.Sequential(conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, bias=False),nn.ReLU())
        #self.conv_outi = nn.Sequential(conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, bias=False),nn.ReLU())
        self.conv1f = nn.Sequential(conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, bias=False),nn.ReLU())
        #self.conv2f = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),nn.ReLU())
        #self.conv3f = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False), nn.ReLU())
        #self.conv4f = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False), nn.ReLU())
        #self.conv5f = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False), nn.ReLU())
        self.conv6f = nn.Sequential(nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
                                    conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False), nn.ReLU())
        self.conv7f = nn.Sequential(nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
                                    conv2d(in_channels=64, out_channels=63, kernel_size=3, stride=1, bias=False), nn.ReLU())
        self.conv_outf = nn.Sequential(conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, bias=False), nn.ReLU())
        self.up = nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)
        self.down = nn.MaxPool2d(4,4)
    def forward(self, init_hr_c: Tensor, init_lr_d: Tensor) ->Union[Tensor,tuple]:

        a = {}
        # 第1层
        a['input_cat_image'] = init_hr_c
        hr_c = self.conv1c(init_hr_c)
        a['conv1c'] = hr_c
        # 第2层CNN
        hr_c = self.conv2c(hr_c)
        a['conv2c'] = hr_c
        # 第1层
        lr_d = self.conv1d(init_lr_d)
        a['conv1d'] = lr_d
        # 第2层
        lr_d = self.conv2d(lr_d)
        a['conv2d'] = lr_d

        hr_i = torch.cat([hr_c, lr_d], dim=1)
        #print(hr_i.size())

        # 第3层
        hr_i = self.conv3i(hr_i)
        a['conv3i'] = hr_i
        # 第4层
        hr_i = self.conv4i(hr_i)
        a['conv4i'] = hr_i

        #init_out_i = torch.cat([hr_i, init_lr_d], dim=1)
        init_out_i = hr_i
        # 第i层
        #init_out_i = self.conv_outi(init_out_i)
        #a['conv_outi'] = init_out_i

        # 第8层
        out_i = self.conv1f(init_out_i)
        a['conv1f'] = out_i
        # 第4层
        #out_i = self.conv2f(out_i)
        #a['conv2f'] = out_i
        #out_i = self.conv3f(out_i)
        #a['conv3f'] = out_i
        # 第4层
        #out_i = self.conv4f(out_i)
        #a['conv4f'] = out_i
        #out_i = self.conv5f(out_i)
        #a['conv5f'] = out_i
        # 第4层
        out_i = self.conv6f(out_i)
        a['conv6f'] = out_i
        out_i = self.conv7f(out_i)
        a['conv7f'] = out_i

        out_f = torch.cat([out_i,init_hr_c[:,0,:,:].unsqueeze(1)], dim=1)
        # 第4层
        out_f = self.conv_outf(out_f)
        a['conv_outf'] = out_f
        return out_f, a

class Net_V6(nn.Module):
    # 基于 CNN 的彩色图像引导的深度图像
    # 超分辨率重
    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        self.conv1 = nn.Sequential(conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, bias=False),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
                                    conv2d(in_channels=64, out_channels=61, kernel_size=3, stride=1, bias=False),nn.ReLU())

        self.conv3 = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False), nn.ReLU())
        self.conv4 = nn.Sequential(conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, bias=False), nn.ReLU())
        self.up = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
        self.down = nn.MaxPool2d(4,4)
    def forward(self, init_hr_c: Tensor, init_lr_d: Tensor) ->Union[Tensor,tuple]:

        a = {}
        # 第1层
        a['input_cat_image'] = self.down(init_lr_d)
        lr_d = self.conv1(self.down(init_lr_d))
        a['conv1c'] = lr_d
        # 第2层CNN
        lr_d = self.conv2(lr_d)
        a['conv2c'] = lr_d


        hr_i = torch.cat([init_hr_c, self.up(lr_d)], dim=1)
        #print(hr_i.size())

        # 第3层
        hr_i = self.conv3(hr_i)
        a['conv3i'] = hr_i
        # 第4层
        hr_i = self.conv4(hr_i)
        a['conv4i'] = hr_i
        hr_i = self.conv4(hr_i)
        a['conv4i'] = hr_i
        hr_i = self.conv4(hr_i)
        a['conv4i'] = hr_i

        out_f = self.conv5(hr_i)
        a['conv_outf'] = out_f
        return out_f, a