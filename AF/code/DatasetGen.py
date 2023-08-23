import os

import cv2
import numpy as np
from core.utils import cal_ig, cal_la

def find_max_metrics(filename):
    max_so = 0
    max_la = 0
    so_id = 0
    la_id = 0

    for index in range(49):
        path_to_img = os.path.join(PATH_TO_SCALED_IMAGE, filename, "{}/result_scaled_image_center.jpg".format(index))
        img = cv2.imread(path_to_img)
        img = cv2.resize(img, (256, 256), fx=1, fy=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cur_so = cal_ig(img)
        cur_la = cal_la(img)

        with open(os.path.join(BASE_PATH_OUT, 'so_of_each_focus.csv'), 'a') as f:
            print(filename, index, cur_so, sep=',', end='\n', file=f)
        with open(os.path.join(BASE_PATH_OUT, 'la_of_each_focus.csv'), 'a') as f:
            print(filename, index, cur_la, sep=',', end='\n', file=f)
        if cur_so > max_so:
            max_so = cur_so
            so_id = index
        if cur_la > max_la:
            max_la = cur_la
            la_id = index
    return so_id, max_so, la_id, max_la
for TRAIN in [7]:

    BASE_PATH = "E:/Dataset/preprocessed/AF"
    BASE_PATH_OUT = "E:/Dataset/preprocessed/AF/train{}".format(TRAIN)
    PATH_TO_SCALED_IMAGE = "E:/Dataset/train{}/scaled_images/".format(TRAIN)
    PATH_TO_LEFT_PD = "E:/Dataset/train{}/raw_left_pd/".format(TRAIN)
    PATH_TO_RIGHT_PD = "E:/Dataset/train{}/raw_right_pd/".format(TRAIN)
    PATH_TO_NUMPY_DATA = os.path.join(BASE_PATH, "numpy_data")#"numpy_data")
    PATH_TO_NUMPY_LABELS = os.path.join(BASE_PATH, "numpy_labels")
    os.makedirs(os.path.join(BASE_PATH_OUT), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_DATA), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_0"), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_1"), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_2"), exist_ok=True)
    for filename in os.listdir(PATH_TO_SCALED_IMAGE):
        print("filename is:{}".format(filename))
        so_id, max_so, la_id, max_la = find_max_metrics(filename)
        dif_id = abs(so_id - la_id)
        print("so_id is:{}, la_id is: {}".format(so_id, la_id))
        with open(os.path.join(BASE_PATH_OUT, 'best_of_a_scene.csv'), 'a') as f:
            print(filename, so_id, la_id, dif_id, sep=',', end='\n', file=f)
        if dif_id == 0:
            for save_id in range(49):
                left_pd = np.array(cv2.imread(os.path.join(PATH_TO_LEFT_PD, filename, str(save_id), "result_pd_left_center.png"), -1), dtype='uint16')
                right_pd = np.array(cv2.imread(os.path.join(PATH_TO_RIGHT_PD, filename, str(save_id), "result_pd_right_center.png"), -1), dtype='uint16')
                stack_pd = np.stack((left_pd, right_pd), axis=2)
                stack_pd = cv2.resize(stack_pd, (256, 256), fx=1, fy=1)
                np.save(os.path.join(PATH_TO_NUMPY_DATA, "{}_{}_data".format(filename, save_id)), stack_pd)
                gt = so_id - save_id
                np.save(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_0", "{}_{}_label".format(filename, save_id)), gt)
                with open(os.path.join(BASE_PATH_OUT, 'ground_truth_{}.txt'.format(TRAIN)), 'a') as f:
                    print("{}_{}_data {}".format(filename, save_id, gt), end='\n',file=f)
                # print("img {}_{}: gt = {}".format(filename, save_id, gt))
        elif dif_id == 1:
            for save_id in range(49):
                left_pd = np.array(
                    cv2.imread(os.path.join(PATH_TO_LEFT_PD, filename, str(save_id), "result_pd_left_center.png"), -1),
                    dtype='uint16')
                right_pd = np.array(
                    cv2.imread(os.path.join(PATH_TO_RIGHT_PD, filename, str(save_id), "result_pd_right_center.png"), -1),
                    dtype='uint16')
                stack_pd = np.stack((left_pd, right_pd), axis=2)
                stack_pd = cv2.resize(stack_pd, (256, 256), fx=1, fy=1)
                np.save(os.path.join(PATH_TO_NUMPY_DATA, "{}_{}_data".format(filename, save_id)), stack_pd)
                gt = round((so_id+la_id)/2) - save_id
                np.save(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_1", "{}_{}_label".format(filename, save_id)), gt)
                with open(os.path.join(BASE_PATH_OUT, 'ground_truth_{}.txt'.format(TRAIN)), 'a') as f:
                    print("{}_{}_data {}".format(filename, save_id, gt), end='\n',file=f)
                # print("img {}_{}: gt = {}".format(filename, save_id, gt))
        elif dif_id == 2:
            for save_id in range(49):
                left_pd = np.array(
                    cv2.imread(os.path.join(PATH_TO_LEFT_PD, filename, str(save_id), "result_pd_left_center.png"), -1),
                    dtype='uint16')
                right_pd = np.array(
                    cv2.imread(os.path.join(PATH_TO_RIGHT_PD, filename, str(save_id), "result_pd_right_center.png"), -1),
                    dtype='uint16')
                stack_pd = np.stack((left_pd, right_pd), axis=2)
                stack_pd = cv2.resize(stack_pd, (256, 256), fx=1, fy=1)
                np.save(os.path.join(PATH_TO_NUMPY_DATA, "{}_{}_data".format(filename, save_id)), stack_pd)
                gt = round((so_id+la_id)/2) - save_id
                np.save(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_2", "{}_{}_label".format(filename, save_id)), gt)
                with open(os.path.join(BASE_PATH_OUT, 'ground_truth_{}.txt'.format(TRAIN)), 'a') as f:
                    print("{}_{}_data {}".format(filename, save_id, gt), end='\n',file=f)