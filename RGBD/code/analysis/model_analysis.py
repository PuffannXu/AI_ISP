import torch
from torchsummary import summary
from RGBD.RGBDNet import Model
from AWB.code.core.settings import DEVICE
# ======================================== #
# 量化训练参数
# ======================================== #
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
DOWN_SCALE = 16
in_range = 1

rgb = (3,256,256)
depth = (1,256,256)
path_to_pretrained = "/home/project/xupf/Projects/AI_ISP/RGBD/output/train/I8W4O8_n0.075/model_down16_1_V5.pth"
rgb_input = torch.randn(1, rgb[0],rgb[1],rgb[2]).to(DEVICE)
model = Model(qn_on = True, version=5, rgb_channel = 3,weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, noise_scale=noise_scale)
summary(model._network, [rgb, depth])
model.load(path_to_pretrained)
dummy_input1= torch.randn(1,3,256,256).to(DEVICE) # RGB输入大小
dummy_input2= torch.randn(1,1,256,256).to(DEVICE) # depth输入大小
torch.onnx.export(model._network, (dummy_input1,dummy_input2), "model.onnx", verbose=True)
