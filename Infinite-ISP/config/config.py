"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2023/9/11 8:55
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""
import os
import math
import scipy
from matplotlib import pyplot as plt
import numpy as np
import yaml
import rawpy
from pathlib import Path
from string import Template

#not to jumble any tags
yaml.preserve_quotes = True
#config_path = 'NUS8_configs.yml'
#config_path = 'SonyA57_configs.yml'
config_path = 'temp_configs.yml'
with open(config_path, 'r', encoding='utf-8') as f:
    c_yaml = yaml.safe_load(f)


# Extract workspace info
platform = c_yaml['platform']
raw_file = platform['filename']
raw_folder = platform['folder']
# Pre-read sensor information
print(50*'-' + '\nPre-reading one RAW Image and re-write yaml......\n')
path_object =  Path(raw_folder, raw_file)
raw_path = str(path_object.resolve())
img = rawpy.imread(raw_path)
raw = img.raw_image


sensor_range =img.white_level - max(img.black_level_per_channel)
width = '{}'.format(img.sizes.raw_width)
height = img.sizes.raw_height
bitdep = int(math.log2(img.white_level+1))
r_offset, gr_offset, \
    gb_offset, b_offset = img.black_level_per_channel
r_sat, gr_sat, \
    gb_sat, b_sat = img.camera_white_level_per_channel

raw_bayer = img.raw_colors[0,0],img.raw_colors[0,1],img.raw_colors[1,0],img.raw_colors[1,1]
if raw_bayer == (2,3,1,0):
    bayer_pattern = "bggr"
elif raw_bayer == (0,1,3,2):
    bayer_pattern = "rggb"
else:
    bayer_pattern = 'none'

temp_c_yaml = Template(c_yaml)
temp_c_yaml.template = str(temp_c_yaml.template)
temp_c_yaml_new = temp_c_yaml.safe_substitute(
    {"bayer_pattern":str(bayer_pattern),"sensor_range": sensor_range,"bitdep":int(bitdep),"width":width,"height":height,
     "r_offset": str(r_offset),"gr_offset":gr_offset,"gb_offset":gb_offset,"b_offset":b_offset,
     "r_sat": r_sat,"gr_sat":gr_sat,"gb_sat":gb_sat,"b_sat":b_sat
     })
temp_c_yaml.template = eval(temp_c_yaml.template)
print(temp_c_yaml_new)
c_yaml_new=yaml.safe_load(temp_c_yaml_new)

config_path_new = 'SonyA57_configs.yml'
with open(config_path_new, 'w', encoding="utf-8") as f:
    # 不排序
    yaml.dump(c_yaml_new, f, sort_keys=False)

