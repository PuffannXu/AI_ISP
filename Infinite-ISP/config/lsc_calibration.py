"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/9/7 13:32
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import numpy as np
import cv2
import rawpy
from matplotlib import pyplot as plt
from AWB.code.core.utils import resize
img = rawpy.imread('/home/project/xupf/Databases/NUS8/CANON1D2/LSC/A95I4988.CR2')
raw = img.raw_image_visible
height,width = raw.shape
demosic_height = int(height/2)
demosic_width = int(width/2)
r = np.empty((demosic_height,demosic_width,4), dtype=np.uint16)

new_height = 60 #916
new_width = 80 #int(width/4)
new_demosic_height = int(new_height/2)
new_demosic_width = int(new_width/2)

r[:,:,0] = raw[::2, ::2]
r[:,:,1] = raw[::2, 1::2]
r[:,:,2] = raw[1::2, ::2]
r[:,:,3] = raw[1::2, 1::2]



#r = cv2.resize(r,(new_demosic_width,new_demosic_height),interpolation=cv2.INTER_NEAREST)
r = resize(r,(new_demosic_height,new_demosic_width))#interpolation=cv.INTER_NEAREST
raw_new = np.empty((new_height,new_width), dtype=np.uint16)
plt.figure()
plt.imshow(r[:,:,0], cmap='gray')
plt.title("r[:,:,0]")
plt.show()
plt.figure()
plt.imshow(r[:,:,1], cmap='gray')
plt.title("r[:,:,1]")
plt.show()
plt.figure()
plt.imshow(r[:,:,2], cmap='gray')
plt.title("r[:,:,2]")
plt.show()
plt.figure()
plt.imshow(r[:,:,3], cmap='gray')
plt.title("r[:,:,3]")
plt.show()
raw_new[::2, ::2] =r[:,:,0]
raw_new[::2, 1::2] =r[:,:,1]
raw_new[1::2, ::2] =r[:,:,2]
raw_new[1::2, 1::2] = r[:,:,3]
plt.figure()
plt.imshow(raw_new, cmap='gray')
plt.title("raw_new")
plt.show()

blc_raw = np.empty(raw_new.shape, dtype=np.uint16)
raw_new = np.float16(raw_new)
r_offset, gr_offset, \
    gb_offset, b_offset = img.black_level_per_channel
r_sat, gr_sat, \
    gb_sat, b_sat = img.camera_white_level_per_channel
blc_raw[0::2, 0::2] = np.clip(raw_new[0::2, 0::2] - r_offset, 0, r_sat)
blc_raw[0::2, 1::2] = np.clip(raw_new[0::2, 1::2] - gr_offset, 0, gr_sat)
blc_raw[1::2, 0::2] = np.clip(raw_new[1::2, 0::2] - gb_offset, 0, gb_sat)
blc_raw[1::2, 1::2] = np.clip(raw_new[1::2, 1::2] - b_offset, 0, b_sat)
blc_raw = np.uint16(blc_raw)

plt.figure()
plt.imshow(blc_raw, cmap='gray')
plt.title("blc_raw")
plt.show()

data = blc_raw
mesh_num = 16
image_r = data[::2, ::2]
image_gr = data[::2, 1::2]
image_gb = data[1::2, ::2]
image_b = data[1::2, 1::2]
height, width = np.shape(image_r)

mesh_height = np.floor(height/mesh_num)
mesh_width = np.floor(width/mesh_num)

mesh_r = np.zeros([mesh_num+1,mesh_num+1])
mesh_gr = np.zeros([mesh_num+1,mesh_num+1])
mesh_gb = np.zeros([mesh_num+1,mesh_num+1])
mesh_b = np.zeros([mesh_num+1,mesh_num+1])



image_point_r = np.zeros([mesh_num+1,mesh_num+1])
image_point_gr = np.zeros([mesh_num+1,mesh_num+1])
image_point_gb = np.zeros([mesh_num+1,mesh_num+1])
image_point_b = np.zeros([mesh_num+1,mesh_num+1])

for i in range(mesh_num+1):
    for j in range(mesh_num+1):
        w_clip = np.int16(np.floor([j * mesh_width - mesh_width / 2, j * mesh_width + mesh_width / 2]))
        h_clip = np.int16(np.floor([i * mesh_height - mesh_height / 2, i * mesh_height + mesh_height / 2]))
        if (i == mesh_num and h_clip[1] != height):
            h_clip[1] = height
        if (j == mesh_num and w_clip[1] != width):
            w_clip[1] = width
        h_clip[h_clip < 1] = 0
        h_clip[h_clip > height] = height
        w_clip[w_clip < 1] = 0
        w_clip[w_clip > width] = width
        if(h_clip[0]==h_clip[1]==0):
            h_clip[1]=1
        elif(h_clip[0]==h_clip[1]==height):
            h_clip[1]=height-1
        elif(h_clip[0]==h_clip[1]):
            h_clip[1] = h_clip[0]+1

        if (w_clip[0] == w_clip[1] == 0):
            w_clip[1] = 1
        elif (w_clip[0] == w_clip[1] == width):
            w_clip[1] = width - 1
        elif (w_clip[0] == w_clip[1]):
            w_clip[1] = w_clip[0] + 1
        data_r = image_r[h_clip[0]:h_clip[1], w_clip[0]: w_clip[1]]
        image_point_r[i, j] = np.mean(np.mean(data_r))
        data_gr = image_gr[h_clip[0]:h_clip[1], w_clip[0]: w_clip[1]]
        image_point_gr[i, j] = np.mean(np.mean(data_gr))
        data_gb = image_gb[h_clip[0]:h_clip[1], w_clip[0]: w_clip[1]]
        image_point_gb[i, j] = np.mean(np.mean(data_gb))
        data_b = image_r[h_clip[0]:h_clip[1], w_clip[0]: w_clip[1]]
        image_point_b[i, j] = np.mean(np.mean(data_b))


rGain = np.zeros((mesh_num + 1, mesh_num + 1),np.float16)
grGain = np.zeros((mesh_num + 1, mesh_num + 1),np.float16)
gbGain = np.zeros((mesh_num + 1, mesh_num + 1),np.float16)
bGain = np.zeros((mesh_num + 1, mesh_num + 1),np.float16)

for i in range(mesh_num+1):
    for j in range(mesh_num+1):
        rGain[i, j] = image_point_r[np.uint8(mesh_num / 2), np.uint8(mesh_num / 2)] / image_point_r[i, j]
        grGain[i, j] = image_point_gr[np.uint8(mesh_num / 2), np.uint8(mesh_num / 2)] / image_point_gr[i, j]
        gbGain[i, j] = image_point_gb[np.uint8(mesh_num / 2), np.uint8(mesh_num / 2)] / image_point_gb[i, j]
        bGain[i, j] = image_point_b[np.uint8(mesh_num / 2), np.uint8(mesh_num / 2)] / image_point_b[i, j]
np.save('rGain{}x{}.mat'.format(new_height,new_width), rGain)
np.save('grGain{}x{}.mat'.format(new_height,new_width), grGain)
np.save('bGain{}x{}.mat'.format(new_height,new_width), bGain)
np.save('gbGain{}x{}.mat'.format(new_height,new_width), gbGain)