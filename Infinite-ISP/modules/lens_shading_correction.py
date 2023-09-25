# File: lens_shading_correction.py
# Description: 
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np
from tqdm import tqdm


class LensShadingCorrection:
    'Lens Shading Correction'

    def __init__(self, img, sensor_info, parm_lsc, rGain, grGain, gbGain, bGain):
        self.img = img
        self.enable = parm_lsc['isEnable']
        self.sensor_info = sensor_info
        self.rGain = np.float16(rGain)
        self.grGain = np.float16(grGain)
        self.gbGain = np.float16(gbGain)
        self.bGain = np.float16(bGain)

    def mesh_shading_correction(self):
        mesh_num = 16

        image_r = self.img[::2, ::2]
        image_gr = self.img[::2, 1::2]
        image_gb = self.img[1::2, ::2]
        image_b = self.img[1::2, 1::2]
        height, width = np.shape(image_b)

        mesh_height = height >> 4
        mesh_width = width >> 4
        rGainTmp = np.zeros(image_r.shape,dtype=np.float16)
        grGainTmp = np.zeros(image_gr.shape,dtype=np.float16)
        gbGainTmp = np.zeros(image_gb.shape,dtype=np.float16)
        bGainTmp = np.zeros(image_b.shape,dtype=np.float16)
        dH = np.zeros(image_b.shape,dtype=np.float16)
        dW = np.zeros(image_b.shape,dtype=np.float16)
        a1 = np.zeros(image_b.shape,dtype=np.float16)
        a2 = np.zeros(image_b.shape,dtype=np.float16)
        a3 = np.zeros(image_b.shape,dtype=np.float16)
        a4 = np.zeros(image_b.shape,dtype=np.float16)
        b1 = np.zeros(image_b.shape,dtype=np.float16)
        b2 = np.zeros(image_b.shape,dtype=np.float16)
        b3 = np.zeros(image_b.shape,dtype=np.float16)
        b4 = np.zeros(image_b.shape,dtype=np.float16)
        c1 = np.zeros(image_b.shape,dtype=np.float16)
        c2 = np.zeros(image_b.shape,dtype=np.float16)
        c3 = np.zeros(image_b.shape,dtype=np.float16)
        c4 = np.zeros(image_b.shape,dtype=np.float16)
        d1 = np.zeros(image_b.shape,dtype=np.float16)
        d2 = np.zeros(image_b.shape,dtype=np.float16)
        d3 = np.zeros(image_b.shape,dtype=np.float16)
        d4 = np.zeros(image_b.shape,dtype=np.float16)
        H_div_sub = np.zeros(image_b.shape,dtype=np.float16)
        W_div_sub = np.zeros(image_b.shape,dtype=np.float16)
        output = np.empty(self.img.shape, np.uint16)

        for cnt_demosic_h in tqdm(range(height), disable=False, leave=False):
            print(cnt_demosic_h)
            for cnt_demosic_w in range(0,width,2):
                if (cnt_demosic_h < mesh_height):#dH = cnt_demosic_h // mesh_height
                    dH[cnt_demosic_h,cnt_demosic_w] = 0
                elif(cnt_demosic_h < 2 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 1
                elif (cnt_demosic_h < 3 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 2
                elif (cnt_demosic_h < 4 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 3
                elif (cnt_demosic_h < 5 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 4
                elif (cnt_demosic_h < 6 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 5
                elif (cnt_demosic_h < 7 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 6
                elif (cnt_demosic_h < 8 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 7
                elif (cnt_demosic_h < 9 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 8
                elif (cnt_demosic_h < 10 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 9
                elif (cnt_demosic_h < 11 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 10
                elif (cnt_demosic_h < 12 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 11
                elif (cnt_demosic_h < 13 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 12
                elif (cnt_demosic_h < 14 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 13
                elif (cnt_demosic_h < 15 * mesh_height):
                    dH[cnt_demosic_h,cnt_demosic_w] = 14
                else: dH[cnt_demosic_h,cnt_demosic_w] = 15
                if (cnt_demosic_w< mesh_width):  # dW = cnt_demosic_w// mesh_width
                    dW[cnt_demosic_h,cnt_demosic_w] = 0
                elif (cnt_demosic_w< 2 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 1
                elif (cnt_demosic_w< 3 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 2
                elif (cnt_demosic_w< 4 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 3
                elif (cnt_demosic_w< 5 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 4
                elif (cnt_demosic_w< 6 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 5
                elif (cnt_demosic_w< 7 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 6
                elif (cnt_demosic_w< 8 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 7
                elif (cnt_demosic_w< 9 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 8
                elif (cnt_demosic_w< 10 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 9
                elif (cnt_demosic_w< 11 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 10
                elif (cnt_demosic_w< 12 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 11
                elif (cnt_demosic_w< 13 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 12
                elif (cnt_demosic_w< 14 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 13
                elif (cnt_demosic_w< 15 * mesh_width):
                    dW[cnt_demosic_h,cnt_demosic_w] = 14
                else:
                    dW[cnt_demosic_h,cnt_demosic_w] = 15
                # f(x,y) = [f(1,0)-f(0,0)]*x+[f(0,1)-f(0,0)]*y+[f(1,1)+f(0,0)-f(1,0)-f(0,1)]*xy+f(0,0)
                # f(x,y) = a*cnt_demosic_h + b*cnt_demosic_w+ c*h*cnt_demosic_w+ d
                H_div_sub[cnt_demosic_h,cnt_demosic_w] = (cnt_demosic_h / mesh_height - dH[cnt_demosic_h,cnt_demosic_w])
                W_div_sub[cnt_demosic_h,cnt_demosic_w] = (cnt_demosic_w/ mesh_width- dW[cnt_demosic_h,cnt_demosic_w])
                index_h = int(dH[cnt_demosic_h,cnt_demosic_w])
                index_w = int(dW[cnt_demosic_h,cnt_demosic_w])
                a1[cnt_demosic_h,cnt_demosic_w] = (self.rGain[index_h + 1, index_w] - self.rGain[index_h, index_w])
                b1[cnt_demosic_h,cnt_demosic_w] = (self.rGain[index_h, index_w + 1] - self.rGain[index_h, index_w])
                c1[cnt_demosic_h,cnt_demosic_w] = (self.rGain[index_h + 1, index_w + 1] + self.rGain[index_h, index_w] - self.rGain[index_h + 1, index_w] - self.rGain[index_h, index_w + 1])
                d1[cnt_demosic_h,cnt_demosic_w] = self.rGain[index_h, index_w]
                a2[cnt_demosic_h,cnt_demosic_w] = (self.grGain[index_h + 1, index_w] - self.grGain[index_h, index_w])
                b2[cnt_demosic_h,cnt_demosic_w] = (self.grGain[index_h, index_w + 1] - self.grGain[index_h, index_w])
                c2[cnt_demosic_h,cnt_demosic_w] = (self.grGain[index_h + 1, index_w + 1] + self.grGain[index_h, index_w] - self.grGain[index_h + 1, index_w] - self.grGain[index_h, index_w + 1])
                d2[cnt_demosic_h,cnt_demosic_w] = self.grGain[index_h, index_w]
                a3[cnt_demosic_h,cnt_demosic_w] = (self.gbGain[index_h + 1, index_w] - self.gbGain[index_h, index_w])
                b3[cnt_demosic_h,cnt_demosic_w] = (self.gbGain[index_h, index_w + 1] - self.gbGain[index_h, index_w])
                c3[cnt_demosic_h,cnt_demosic_w] = (self.gbGain[index_h + 1, index_w + 1] + self.gbGain[index_h, index_w] - self.gbGain[index_h + 1, index_w] - self.gbGain[index_h, index_w + 1])
                d3[cnt_demosic_h,cnt_demosic_w] = self.gbGain[index_h, index_w]
                a4[cnt_demosic_h,cnt_demosic_w] = (self.bGain[index_h + 1, index_w] - self.bGain[index_h, index_w])
                b4[cnt_demosic_h,cnt_demosic_w] = (self.bGain[index_h, index_w + 1] - self.bGain[index_h, index_w])
                c4[cnt_demosic_h,cnt_demosic_w] = (self.bGain[index_h + 1, index_w + 1] + self.bGain[index_h, index_w] - self.bGain[index_h + 1, index_w] - self.bGain[index_h, index_w + 1])
                d4[cnt_demosic_h,cnt_demosic_w] = self.bGain[index_h, index_w]
                rGainTmp[cnt_demosic_h,cnt_demosic_w] = a1[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] + b1[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] +c1[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] + d1[cnt_demosic_h,cnt_demosic_w]
                grGainTmp[cnt_demosic_h,cnt_demosic_w] = a2[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] + b2[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] + c2[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] + d2[cnt_demosic_h,cnt_demosic_w]
                gbGainTmp[cnt_demosic_h,cnt_demosic_w] = a3[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] + b3[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] + c3[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] + d3[cnt_demosic_h,cnt_demosic_w]
                bGainTmp[cnt_demosic_h,cnt_demosic_w] = a4[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] + b4[cnt_demosic_h,cnt_demosic_w] * W_div_sub[cnt_demosic_h,cnt_demosic_w] + c4[cnt_demosic_h,cnt_demosic_w] * H_div_sub[cnt_demosic_h,cnt_demosic_w] *  W_div_sub[cnt_demosic_h,cnt_demosic_w] + d4[cnt_demosic_h,cnt_demosic_w]
                if(cnt_demosic_w+1!=width):
                    rGainTmp[cnt_demosic_h, cnt_demosic_w+1] = rGainTmp[cnt_demosic_h,cnt_demosic_w]
                    grGainTmp[cnt_demosic_h, cnt_demosic_w+ 1] = grGainTmp[cnt_demosic_h, cnt_demosic_w]
                    gbGainTmp[cnt_demosic_h, cnt_demosic_w+ 1] = gbGainTmp[cnt_demosic_h, cnt_demosic_w]
                    bGainTmp[cnt_demosic_h, cnt_demosic_w+ 1] = bGainTmp[cnt_demosic_h, cnt_demosic_w]

        output[::2, ::2] = image_r * rGainTmp
        output[::2, 1::2] = image_gr * grGainTmp
        output[1::2, ::2] = image_gb* gbGainTmp
        output[1::2, 1::2] = image_b* bGainTmp
        return output

    def execute(self):
        mesh_num = 16

        image_r = self.img[::2, ::2]
        image_gr = self.img[::2, 1::2]
        image_gb = self.img[1::2, ::2]
        image_b = self.img[1::2, 1::2]
        height, width = np.shape(image_b)

        mesh_height = np.floor(height / mesh_num)
        mesh_width = np.floor(width / mesh_num)
        rGainTmp = np.zeros(image_r.shape)
        grGainTmp = np.zeros(image_gr.shape)
        gbGainTmp = np.zeros(image_gb.shape)
        bGainTmp = np.zeros(image_b.shape)

        output = np.empty(self.img.shape, np.uint16)

        for i in tqdm(range(height)):
            print(i)
            for j in range(width):
                dH = np.int16(np.floor(i / mesh_height))
                if dH > 15:
                    dH = 15
                dW = np.int16(np.floor(j / mesh_width))
                if dW > 15:
                    dW = 15
                rGainTmp[i,j] = self.rGain[dH, dW]
                grGainTmp[i,j] = self.grGain[dH, dW]
                gbGainTmp[i,j] = self.gbGain[dH, dW]
                bGainTmp[i,j] = self.bGain[dH, dW]
        output[::2, ::2] = image_r * rGainTmp
        output[::2, 1::2] = image_gr * grGainTmp
        output[1::2, ::2] = image_gb* gbGainTmp
        output[1::2, 1::2] = image_b* bGainTmp
        return output

