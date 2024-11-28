"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/10/11 19:57
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import torch
from torch import nn, Tensor
from torch.nn.functional import normalize
from typing import Union, Tuple
from Alex_FC4 import Net_in_channel


channel_number=4


# #
model = Net_in_channel(channel_number)
          # 需要加载量化加噪之前的网络

checkpoint = torch.load('/home/project/xupf/Projects/AI_ISP/AWB/model/AWB_4channel_post_easy.pth',map_location='cpu')
model.load_state_dict(checkpoint,strict=True)
# model = torch.load('/home/project/FDheadclass.pt')
# model = torch.load('FDheadclass.pt', map_location='cpu')
x = torch.randn(1, 4, 160, 120)
torch.save(model, '/home/project/xupf/Projects/AI_ISP/AWB/model/Model_AWB_4channel_post_easy.pth')
torch.onnx.export(model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "/home/project/xupf/Projects/AI_ISP/AWB/model/AWB_4channel_post_easy.onnx",  # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  do_constant_folding=True,  # whether to execute constant folding for optimization

                  )