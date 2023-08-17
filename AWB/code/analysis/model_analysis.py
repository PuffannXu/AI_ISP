import torch
from torchsummary import summary
from AWB.model.Alex_FC4 import Model
from AWB.code.core.settings import DEVICE
# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 1
isint = 0
input_bit = 12
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0.075

input_size=(3,256,256)
path_to_pretrained = "/home/project/xupf/Projects/AI_ISP/AWB/output/train/I8W4O8_n0.075_fold_0/model.pth"
dummy_input = torch.randn(1, input_size[0],input_size[1],input_size[2]).to(DEVICE)
model=Model(weight_bit, output_bit, isint, clamp_std, noise_scale)
summary(model._network,input_size)
#model.load(path_to_pretrained)
#torch.onnx.export(model._network, dummy_input, "model.onnx", verbose=True)
