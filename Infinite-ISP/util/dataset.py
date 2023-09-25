"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/9/9 13:30
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""

import scipy.io
import numpy as np
import os


ground_truth = scipy.io.loadmat('/home/project/xupf/Databases/NUS8/SonyA57_RAW/SonyA57_gt.mat')
illums = ground_truth['groundtruth_illuminants']
darkness_level = ground_truth['darkness_level']
saturation_level = ground_truth['saturation_level']
cc_coords = ground_truth['CC_coords']
illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]

extras = {
    'darkness_level': darkness_level,
    'saturation_level': saturation_level
}