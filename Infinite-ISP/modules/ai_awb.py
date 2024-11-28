import os

import numpy as np
import torch
from AWB.code.core.settings import DEVICE
from AWB.code.core.utils import correct, scale, hwc_to_chw, resize, normalize
from AWB.model.Alex_FC4 import Model
from RRAM import my_utils as my

def ai_awb(ori_img, sensor_info):

    # ======================================== #
    # 量化训练参数
    # ======================================== #
    img_quant_flag = 1
    isint = 0
    qn_on = True
    input_bit = 8
    weight_bit = 4
    output_bit = 8
    clamp_std = 0
    noise_scale = 0.075
    # Where to save the generated visualizations
    #path_to_pretrained = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/model/I8W4O8_n0.075_AWB_V2.pth")
    path_to_pretrained = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/output/train/I8W4O8_n0.075_fold_0/model_VAlex.pth")
    model = Model(qn_on, weight_bit, output_bit, isint, clamp_std, noise_scale)
    model.load(path_to_pretrained)
    model.evaluation_mode()
    #ori_img = bayer_img
    img = resize(ori_img, (256, 256))
    img = hwc_to_chw(scale(img))

    img = torch.from_numpy(img.copy())
    img = img.unsqueeze(0)

    img = img.to(DEVICE)
    if img_quant_flag == 1:
        img, _ = my.data_quantization_sym(img, half_level=2 ** input_bit / 2 - 1)
        # img = img / (2 ** input_bit)
        img = img.float()
    pred, rgb, confidence, a = model.predict(img, return_steps=True)
    #output = correct(original, pred)
    pred = pred.squeeze().cpu().detach().numpy()
    #rGain = pred[1]/pred[0]
    #gGain = pred[1]/pred[1]
    #bGain = pred[1]/pred[2]
    rGain = pred[0]
    gGain = pred[1]
    bGain = pred[2]

    bGain = 1 if bGain <= 0 else bGain
    rGain = 1 if rGain <= 0 else rGain
    print("rGain = ", rGain)
    print("gGain = ", gGain)
    print("bGain = ", bGain)
    cor_img = np.empty(ori_img.shape, np.float16)
    cor_img[:,:,0] = ori_img[:,:,0] * rGain
    cor_img[:, :, 1] = ori_img[:, :, 1] * gGain
    cor_img[:, :, 2] = ori_img[:, :, 2] * bGain

    return np.uint16(np.clip(cor_img, 0, int(sensor_info['sensor_range']))), pred