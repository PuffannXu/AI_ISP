# File: isp_pipeline.py
# Description: Executes the complete pipeline
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------
import os
import math
import scipy
from matplotlib import pyplot as plt
import numpy as np
import yaml
import rawpy
from pathlib import Path
from modules.dead_pixel_correction import DeadPixelCorrection as DPC
from modules.lens_shading_correction import LensShadingCorrection as LSC
from modules.auto_white_balance import AutoWhiteBalance as AWB
from modules.demosaic import CFAInterpolation as CFA_I
from modules.ai_awb import ai_awb


yaml.preserve_quotes = True
config_path = './config/CANON1D2_configs.yml'
with open(config_path, 'r') as f:
    c_yaml = yaml.safe_load(f)

# Extract basic sensor info
sensor_info = c_yaml['sensor_info']
platform = c_yaml['platform']
raw_file = platform['filename']
raw_folder = platform['folder']
black_level_correction = c_yaml['black_level_correction']
r_offset, gr_offset, gb_offset, b_offset = \
    int(black_level_correction['r_offset']),int(black_level_correction['gr_offset']),\
    int(black_level_correction['gb_offset']),int(black_level_correction['b_offset'])
r_sat, gr_sat, gb_sat, b_sat = \
    int(black_level_correction['r_sat']),int(black_level_correction['gr_sat']),\
    int(black_level_correction['gb_sat']),int(black_level_correction['b_sat'])
sensor_range = int(sensor_info['sensor_range'])
bayer = str(sensor_info['bayer_pattern'])
width = int(sensor_info['width'])
height = int(sensor_info['height'])
bpp = int(sensor_info['bitdep'])

# Get isp module params
parm_dpc = c_yaml['dead_pixel_correction']
parm_lsc = c_yaml['lens_shading_correction']
parm_wbc = c_yaml['white_balance']
parm_awb = c_yaml['auto_white_balance']

wb = [1,1,1]
for filename in os.listdir(raw_folder):

    raw_path = os.path.join(raw_folder,str(filename))
    img = rawpy.imread(raw_path)
    raw = img.raw_image_visible

    print(50*'-' + '\nLoading RAW Image Done......\n')

    plt.subplot(3,3,1)
    plt.imshow(raw/sensor_range, cmap='gray')
    plt.title("raw")
    #plt.show()

    # -------------------------------------------------------
    # raw signal processing start
    # -------------------------------------------------------

    # -------------blc-----------
    blc_raw = np.empty(raw.shape, dtype=np.uint16)
    raw = np.float32(raw)
    #r_offset, gr_offset, gb_offset, b_offset = img.black_level_per_channel
    #r_sat, gr_sat, gb_sat, b_sat = img.camera_white_level_per_channel
    blc_raw[0::2, 0::2] = np.clip(raw[0::2, 0::2] - r_offset, 0, r_sat)
    blc_raw[0::2, 1::2] = np.clip(raw[0::2, 1::2] - gr_offset, 0, gr_sat)
    blc_raw[1::2, 0::2] = np.clip(raw[1::2, 0::2] - gb_offset, 0, gb_sat)
    blc_raw[1::2, 1::2] = np.clip(raw[1::2, 1::2] - b_offset, 0, b_sat)
    blc_raw = np.uint16(blc_raw)

    plt.subplot(3, 3, 2)
    plt.imshow(blc_raw/sensor_range, cmap='gray')
    plt.title("blc_raw")
    #plt.show()

    # -------------dpc-----------
    dpc = DPC(blc_raw, sensor_info, parm_dpc, platform)
    dpc_raw = dpc.execute()

    # -------------lsc-----------
    rGain = np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/rGain_A95I4988.mat.npy')
    grGain = np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/grGain_A95I4988.mat.npy')
    gbGain = np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/gbGain_A95I4988.mat.npy')
    bGain = np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/bGain_A95I4988.mat.npy')

    lsc = LSC(blc_raw, sensor_info, parm_lsc, rGain, grGain, gbGain, bGain)
    lsc_raw = lsc.mesh_shading_correction()

    # --------demosaic-----------
    cfa_inter = CFA_I(lsc_raw, sensor_info)
    demos_img = cfa_inter.execute()

    plt.subplot(3, 3, 3)
    plt.imshow(demos_img/sensor_range)
    plt.title("demos_img")

    if bayer=='bggr':
        b=demos_img[:,:,0]
        g=demos_img[:,:,1]
        r=demos_img[:,:,2]
        demos_img = np.stack((r,g,b),axis=2)
    #plt.show()


    # --------wb gain-----------
    demos_img = np.float32(demos_img)
    wb_raw = np.uint16(np.minimum(demos_img * wb, sensor_range))
    print()
    print("rGain = ", wb[0])
    print("gGain = ", wb[1])
    print("bGain = ", wb[2])
    plt.subplot(3, 3, 4)
    plt.imshow(wb_raw/sensor_range)
    plt.title("wb_gt\n{:.3f},{:.3f},{:.3f}".format(wb[0],wb[1],wb[2]))
    #plt.show()

    # Auto White Balance
    awb = AWB(demos_img, sensor_info, parm_wbc, parm_awb)
    awb_img, wb = awb.execute()

    plt.subplot(3, 3, 5)
    plt.imshow(awb_img/awb_img.max())
    plt.title("awb_img\n{:.3f},{:.3f},{:.3f}".format(wb[0],wb[1],wb[2]))
    #plt.show()
    #  --------ai af-----------
    ai_awb_img, wb = ai_awb(demos_img, sensor_info)

    plt.subplot(3, 3, 6)
    plt.imshow(ai_awb_img / ai_awb_img.max())
    plt.title("ai_awb_img\n{:.3f},{:.3f},{:.3f}".format(wb[0], wb[1], wb[2]))

    #  --------ai awb-----------
    ai_awb_img, wb = ai_awb(demos_img, sensor_info)

    plt.subplot(3, 3, 6)
    plt.imshow(ai_awb_img/ai_awb_img.max())
    plt.title("ai_awb_img\n{:.3f},{:.3f},{:.3f}".format(wb[0],wb[1],wb[2]))

    # -------
    gamma_gt = wb_raw**(1/2.2)

    plt.subplot(3, 3, 7)
    plt.imshow(gamma_gt / gamma_gt.max())
    plt.title("gamma_gt")

    gamma_awb = awb_img ** (1 / 2.2)

    plt.subplot(3, 3, 8)
    plt.imshow(gamma_awb / gamma_awb.max())
    plt.title("gamma_awb")

    gamma_ai_awb = ai_awb_img ** (1 / 2.2)

    plt.subplot(3, 3, 9)
    plt.imshow(gamma_ai_awb / gamma_ai_awb.max())
    plt.title("gamma_ai_awb")

    plt.suptitle(filename)
    plt.show()
    plt.savefig("/home/project/xupf/Projects/AI_ISP/Infinite-ISP/out_frames/{}.png".format(filename))


