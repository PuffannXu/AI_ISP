"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/9/11 13:44
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import os
from AWB.code.core.utils import correct, scale, hwc_to_chw, resize
import cv2
from matplotlib import pyplot as plt
import numpy as np
import yaml
import rawpy
from dead_pixel_correction import DeadPixelCorrection as DPC
from lens_shading_correction import LensShadingCorrection as LSC
from demosaic import CFAInterpolation as CFA_I
yaml.preserve_quotes = True
config_path = '../config/CANON1D2_configs.yml'
#config_path = './config/SonyA57_configs.yml'
with open(config_path, 'r') as f:
    c_yaml = yaml.safe_load(f)

new_height = 60 #916
new_width = 80 #int(width/4)
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

rGain = np.float16(np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/rGain{}x{}.mat.npy'.format(new_height,new_width)))
grGain = np.float16(np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/grGain{}x{}.mat.npy'.format(new_height,new_width)))
gbGain = np.float16(np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/gbGain{}x{}.mat.npy'.format(new_height,new_width)))
bGain = np.float16(np.load('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/config/bGain{}x{}.mat.npy'.format(new_height,new_width)))
import struct
rg = np.empty(rGain.shape,dtype = np.uint16)
grg = np.empty(rGain.shape,dtype = np.uint16)
gbg = np.empty(rGain.shape,dtype = np.uint16)
bg = np.empty(rGain.shape,dtype = np.uint16)
for i in range(rGain.shape[0]):
    for j in range(rGain.shape[1]):
        rg[i, j] = struct.unpack('<h', struct.pack('<e', rGain[i, j]))[0]  # 将浮点数按照2byte float转换
        grg[i, j] = struct.unpack('<h', struct.pack('<e', grGain[i, j]))[0]
        gbg[i, j] = struct.unpack('<h', struct.pack('<e', gbGain[i, j]))[0]
        bg[i, j] = struct.unpack('<h', struct.pack('<e', bGain[i, j]))[0]
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/rgain.txt',rg,fmt="%x",delimiter=' ')
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/grgian.txt',grg,fmt="%x",delimiter=' ')
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/gbgain.txt',gbg,fmt="%x",delimiter=' ')
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/bgain.txt',bg,fmt="%x",delimiter=' ')

raw_folder = '/home/project/xupf/Databases/NUS8/CANON1D2/LSC'
filename ='A95I4988.CR2'
raw_path = os.path.join(raw_folder,str(filename))
img = rawpy.imread(raw_path)
raw = img.raw_image_visible
height,width = raw.shape
demosic_height = int(height/2)
demosic_width = int(width/2)
r = np.empty((demosic_height,demosic_width,4), dtype=np.uint16)


new_demosic_height = int(new_height/2)
new_demosic_width = int(new_width/2)

r[:,:,0] = raw[::2, ::2]
r[:,:,1] = raw[::2, 1::2]
r[:,:,2] = raw[1::2, ::2]
r[:,:,3] = raw[1::2, 1::2]



#r = cv2.resize(r,(new_demosic_width,new_demosic_height),interpolation=cv2.INTER_NEAREST)
r = resize(r,(new_demosic_height,new_demosic_width))#interpolation=cv.INTER_NEAREST
raw_new = np.empty((new_height,new_width), dtype=np.uint16)


raw_new[::2, ::2] =r[:,:,0]
raw_new[::2, 1::2] =r[:,:,1]
raw_new[1::2, ::2] =r[:,:,2]
raw_new[1::2, 1::2] = r[:,:,3]


output_folder = '/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2'
os.makedirs(output_folder,exist_ok=True)

print(50*'-' + '\nLoading RAW Image Done......\n')

plt.figure()
plt.imshow(raw/raw.max(), cmap='gray')
plt.title("raw")
plt.show()
raw=raw_new
plt.figure()
plt.imshow(raw_new/raw_new.max(), cmap='gray')
plt.title("raw_new")
plt.show()
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/LSC_raw.txt',raw,fmt="%x",delimiter=' ')
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/LSC_raw_dec.txt',raw,fmt="%d",delimiter=' ')
output = os.path.join(output_folder,'{}_raw.jpg'.format(filename))
cv2.imwrite(output,raw/raw.max()*255)
blc_raw = np.empty(raw.shape, dtype=np.uint16)
raw = np.float16(raw)

r_offset, gr_offset, gb_offset, b_offset = img.black_level_per_channel
r_sat, gr_sat, gb_sat, b_sat = img.camera_white_level_per_channel
blc_raw[0::2, 0::2] = np.clip(raw[0::2, 0::2] - r_offset, 0, r_sat)
blc_raw[0::2, 1::2] = np.clip(raw[0::2, 1::2] - gr_offset, 0, gr_sat)
blc_raw[1::2, 0::2] = np.clip(raw[1::2, 0::2] - gb_offset, 0, gb_sat)
blc_raw[1::2, 1::2] = np.clip(raw[1::2, 1::2] - b_offset, 0, b_sat)
blc_raw = np.uint16(blc_raw)

plt.figure()
plt.imshow(blc_raw/blc_raw.max(), cmap='gray')
plt.title("blc_raw")
plt.show()
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/LSC_blc.txt',blc_raw,fmt="%x",delimiter=' ')
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/LSC_blc_dec.txt',blc_raw,fmt="%d",delimiter=' ')
output = os.path.join(output_folder,'{}_blc_raw.jpeg'.format(filename))
cv2.imwrite(output,blc_raw/blc_raw.max()*255)
#  Dead pixels correction
dpc = DPC(blc_raw, sensor_info, parm_dpc, platform)
dpc_raw = dpc.execute()
plt.imshow(dpc_raw, cmap='gray')
plt.title("dpc_raw")
plt.show()
output = os.path.join(output_folder,'{}_dpc_raw.jpeg'.format(filename))
cv2.imwrite(output,dpc_raw/dpc_raw.max()*255)
# 6 Lens shading correction
'''
demosaic_out = np.empty([int(height/2),int(width/2),4],dtype=np.uint16)
demosaic_out[0:height, 0:width, 0] = raw[::2, ::2]
demosaic_out[0:height, 0:width, 1] = raw[::2, 1::2]
demosaic_out[0:height, 0:width, 2] = raw[1::2, ::2]
demosaic_out[0:height, 0:width, 3] = raw[1::2, 1::2]

demosaic_out = np.uint16(np.clip(demosaic_out, 0, sensor_range))
plt.imshow(demosaic_out[:,:, 0], cmap='gray')
plt.title("demosaic_out")
plt.show()

size_down = (int(height/32),int(width/32))
demosaic_out_down_cv2 = cv2.resize(demosaic_out,(size_down[1],size_down[0]),interpolation=cv2.INTER_NEAREST)
plt.imshow(demosaic_out_down_cv2[:,:, 0], cmap='gray')
plt.title("demosaic_out_down_cv2")
plt.show()

demosaic_out_down = resize(demosaic_out,size_down)
plt.imshow(demosaic_out_down[:,:, 0], cmap='gray')
plt.title("demosaic_out_down")
plt.show()

blc_down = np.empty([size_down[0]*2,size_down[1]*2],dtype=np.uint16)
blc_down[::2, ::2] = demosaic_out_down[:,:,0]
blc_down[::2, 1::2] = demosaic_out_down[:,:,1]
blc_down[1::2, ::2] = demosaic_out_down[:,:,2]
blc_down[1::2, 1::2] = demosaic_out_down[:,:,3]
plt.imshow(blc_down, cmap='gray')
plt.title("blc_down")
plt.show()'''
lsc = LSC(dpc_raw, sensor_info, parm_lsc, rGain, grGain, gbGain, bGain)
lsc_raw = lsc.mesh_shading_correction()

plt.imshow(lsc_raw, cmap='gray')
plt.title("lsc_raw")
plt.show()
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/LSC_lsc.txt',lsc_raw,fmt="%x",delimiter=' ')
np.savetxt('/home/project/xupf/Projects/AI_ISP/Infinite-ISP/CANON1D2/LSC_lsc_dec.txt',lsc_raw,fmt="%d",delimiter=' ')
output = os.path.join(output_folder,'{}_lsc_raw.jpeg'.format(filename))
cv2.imwrite(output,lsc_raw/lsc_raw.max()*255)
# 9 CFA demosaicing
#demos_img =np.stack(wb_raw[::2, ::2], (wb_raw[::2, 1::2] + wb_raw[1::2, ::2])/2.0,wb_raw[1::2, 1::2])
cfa_inter = CFA_I(lsc_raw, sensor_info)
demos_img = cfa_inter.execute()

plt.imshow(demos_img/demos_img.max(), cmap='gray')
plt.title("demos_img")
plt.show()
output = os.path.join(output_folder,'{}_demos_img.jpeg'.format(filename))
cv2.imwrite(output,demos_img/demos_img.max()*255)
plt.imshow(demos_img[:,:,0]/demos_img[:,:,0].max(), cmap='gray')
plt.title("demos_img")
plt.show()
if bayer=='bggr':
    b=demos_img[:,:,0]
    g=demos_img[:,:,1]
    r=demos_img[:,:,2]
    demos_img = np.stack((r,g,b),axis=2)

