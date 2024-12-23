# File: isp_pipeline.py
# Description: Executes the complete pipeline
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------
import os
import scipy
from matplotlib import pyplot as plt
import numpy as np
import yaml
import rawpy
from modules.dead_pixel_correction import DeadPixelCorrection as DPC
from modules.lens_shading_correction import LensShadingCorrection as LSC
from modules.auto_white_balance import AutoWhiteBalance as AWB
from modules.demosaic import CFAInterpolation as CFA_I
from modules.ai_awb import ai_awb

#not to jumble any tags
yaml.preserve_quotes = True
#config_path = './config/NUS8_configs.yml'
config_path = './config/SonyA57_configs.yml'
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


print(50*'-' + '\nLoading metadata......\n')
path_to_meta_data = '/home/project/xupf/Databases/NUS8/SonyA57_RAW/SonyA57_gt.mat'
#path_to_meta_data = '/home/project/xupf/Databases/NUS8/NikonD40_RAW/NikonD40_gt.mat'

meta_data = []
ground_truth = scipy.io.loadmat(path_to_meta_data)
illums = ground_truth['groundtruth_illuminants']
illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]
filenames = ground_truth['all_image_names']
filnames_array = np.empty((),dtype='<U12')
illums_array = np.empty((3),dtype='float16')
for i in range(len(filenames)):
    filnames_array = np.vstack((filnames_array, np.array(str(filenames[i,0]).strip("[]'"))))
    illums_array = np.vstack((illums_array, np.array(illums[i])))
filnames_array = filnames_array[1:,:]
illums_array = illums_array[1:,:]
for i in range(len(filnames_array)):
    fn = filnames_array[i,:]
    ill = illums_array[i,:]
    raw_path = os.path.join(raw_folder,str(fn[0])+".ARW")
    img = rawpy.imread(raw_path)
    raw = img.raw_image

    print(50*'-' + '\nLoading RAW Image Done......\n')

    plt.subplot(2,3,1)
    plt.imshow(raw/raw.max(), cmap='gray')
    plt.title("raw")
    #plt.show()


    blc_raw = np.empty(raw.shape, dtype=np.uint16)
    blc_raw[0::2, 0::2] = raw[0::2, 0::2] - r_offset
    blc_raw[0::2, 1::2] = raw[0::2, 1::2] - gr_offset
    blc_raw[1::2, 0::2] = raw[1::2, 0::2] - gb_offset
    blc_raw[1::2, 1::2] = raw[1::2, 1::2] - b_offset
    blc_raw = np.uint16(np.clip(blc_raw, 0, sensor_range))

    plt.subplot(2, 3, 2)
    plt.imshow(blc_raw/blc_raw.max(), cmap='gray')
    plt.title("blc_raw")
    #plt.show()

    #  Dead pixels correction
    dpc = DPC(blc_raw, sensor_info, parm_dpc, platform)
    dpc_raw = dpc.execute()


    # 6 Lens shading correction
    #lsc = LSC(blc_raw, sensor_info, parm_lsc)
    #lsc_raw = lsc.execute()

    lsc_raw = dpc_raw

    # 9 CFA demosaicing
    #demos_img =np.stack(wb_raw[::2, ::2], (wb_raw[::2, 1::2] + wb_raw[1::2, ::2])/2.0,wb_raw[1::2, 1::2])
    cfa_inter = CFA_I(lsc_raw, sensor_info)
    demos_img = cfa_inter.execute()

    plt.subplot(2, 3, 3)

    if bayer=='bggr':
        b=demos_img[:,:,0]
        g=demos_img[:,:,1]
        r=demos_img[:,:,2]
        demos_img = np.stack((r,g,b),axis=2)
    plt.imshow(demos_img / demos_img.max())
    plt.title("demos_img")
    #plt.show()


    # 8 White balancing
    #wb = np.array(img.camera_whitebalance[:3], np.float32)
    #wb = wb / wb[1]
    wb = 1/ill
    wb_raw = np.uint16(np.minimum(demos_img * wb, sensor_range))
    print()
    print("rGain = ", wb[0])
    print("gGain = ", wb[1])
    print("bGain = ", wb[2])
    plt.subplot(2, 3, 4)
    plt.imshow((wb_raw/wb_raw.max())**(1/2.2))
    plt.title("wb_gt\n{:.3f},{:.3f},{:.3f}".format(wb[0],wb[1],wb[2]))
    #plt.show()

    # Auto White Balance
    awb = AWB(demos_img, sensor_info, parm_wbc, parm_awb)
    awb_img, wb = awb.execute()

    plt.subplot(2, 3, 5)
    plt.imshow((awb_img/awb_img.max())**(1/2.2))
    plt.title("awb_img\n{:.3f},{:.3f},{:.3f}".format(wb[0],wb[1],wb[2]))
    #plt.show()

    # Auto White Balance
    ai_awb_img, wb = ai_awb(demos_img, sensor_info)

    plt.subplot(2, 3, 6)
    plt.imshow((ai_awb_img/ai_awb_img.max())**(1/2.2))
    plt.title("ai_awb_img\n{:.3f},{:.3f},{:.3f}".format(wb[0],wb[1],wb[2]))
    #plt.show()
    # Auto White Balance
    #gamma_gt = wb_raw**(1/2.2)

    #plt.subplot(3, 3, 7)
    #plt.imshow(gamma_gt / gamma_gt.max())
    #plt.title("gamma_gt")
    #plt.show()
    #gamma_awb = awb_img ** (1 / 2.2)

    #plt.subplot(3, 3, 8)
    #plt.imshow(gamma_awb / gamma_awb.max())
    #plt.title("gamma_awb")
    #plt.show()
    #gamma_ai_awb = ai_awb_img ** (1 / 2.2)

    #plt.subplot(3, 3, 9)
    #plt.imshow(gamma_ai_awb / gamma_ai_awb.max())
    #plt.title("gamma_ai_awb")

    plt.suptitle(fn)
    plt.show()
   # plt.savefig("/home/project/xupf/Projects/AI_ISP/Infinite-ISP/out_frames/{}.png".format(fn))


