import os
import cv2
import numpy as np
from core.utils import crop, depth2focus
from core.settings import ROI_SIZE
from matplotlib import pyplot as plt

cam_pos = "center"
x_pos = 0.4
y_pos = 0.4
for TRAIN in [0,1,2,5,6,7]:

    # input path
    BASE_PATH = "E:/Dataset/preprocessed/AF_{}_{}".format(cam_pos, (x_pos, y_pos))
    PATH_TO_SCALED_IMAGE = "E:/Dataset/train{}/scaled_images/".format(TRAIN)
    PATH_TO_MERGED_DEPTH = "E:/Dataset/train{}/merged_depth/".format(TRAIN)
    PATH_TO_LEFT_PD = "E:/Dataset/train{}/raw_up_left_pd/".format(TRAIN)
    PATH_TO_RIGHT_PD = "E:/Dataset/train{}/raw_up_right_pd/".format(TRAIN)
    # output path
    PATH_TO_NUMPY_DATA = os.path.join(BASE_PATH, "numpy_data")
    PATH_TO_NUMPY_LABELS = os.path.join(BASE_PATH, "numpy_labels")
    os.makedirs(os.path.join(PATH_TO_NUMPY_DATA), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS), exist_ok=True)

    with open(os.path.join(BASE_PATH, '{}_{}_{}.csv'.format(cam_pos, x_pos, y_pos)), 'a') as f:
        print("Train{}".format(TRAIN), "median_depth", "focus", "min_diff", sep=',', end='\n', file=f)

    for filename in os.listdir(PATH_TO_SCALED_IMAGE):

        # 读入scaled image 378*504
        print("filename is:{}".format(filename))
        # path_to_scaled_img = os.path.join(PATH_TO_SCALED_IMAGE, filename,
        #                                  "{}/result_scaled_image_center.jpg".format(20))
        # scaled_img = cv2.imread(path_to_scaled_img)
        # cropped_img = crop(scaled_img, 0.5, 0.5, ROI_SIZE / 4)  # 256/4 = 64

        # 读入merged_depth 378*504
        path_to_merged_depth = os.path.join(PATH_TO_MERGED_DEPTH, filename,
                                            "result_merged_depth_{}.png".format(cam_pos))
        merged_depth = cv2.imread(path_to_merged_depth)
        merged_depth = cv2.cvtColor(merged_depth, cv2.COLOR_BGR2GRAY)
        # 按照中心裁剪为64x64
        cropped_merged_depth = crop(merged_depth, x_pos, y_pos, ROI_SIZE / 4)  # 256/4 = 64
        depth = np.median(cropped_merged_depth)
        focus, min_diff = depth2focus(depth)
        print("depth: {}".format(depth))
        print("focus: {}".format(focus))

        with open(os.path.join(BASE_PATH, '{}_{}_{}.csv'.format(cam_pos, x_pos, y_pos, )), 'a') as f:
            print(filename, depth, focus, min_diff, sep=',', end='\n', file=f)

        if 102 < depth < 3911:
            for index in range(49):
                left_pd = np.array(
                    cv2.imread(
                        os.path.join(PATH_TO_LEFT_PD, filename, str(index), "result_up_pd_left_{}.png".format(cam_pos)),
                        -1),
                    dtype='uint16')
                right_pd = np.array(
                    cv2.imread(os.path.join(PATH_TO_RIGHT_PD, filename, str(index),
                                            "result_up_pd_right_{}.png".format(cam_pos)),
                               -1), dtype='uint16')
                stack_pd = np.stack((left_pd, right_pd), axis=2)
                cropped_stack_pd = crop(stack_pd, x_pos, y_pos, ROI_SIZE)
                np.save(os.path.join(PATH_TO_NUMPY_DATA,
                                     "{}_{}_{}_{}_{}_data".format(filename, index, cam_pos, x_pos, y_pos)),
                        cropped_stack_pd)
                gt = focus - index
                np.save(os.path.join(PATH_TO_NUMPY_LABELS,
                                     "{}_{}_{}_{}_{}_label".format(filename, index, cam_pos, x_pos, y_pos)), gt)
                with open(os.path.join(BASE_PATH,
                                       'ground_truth_{}_{}_{}_train{}.txt'.format(cam_pos, x_pos, y_pos, TRAIN)),
                          'a') as f:
                    print("{}_{}_data,{}".format(filename, index, gt), end='\n', file=f)
                # print("img {}_{}: gt = {}".format(filename, save_id, gt))
                '''
                fig, axs = plt.subplots(2, 4)
                axs[0, 0].imshow(scaled_img)
                axs[0, 0].set_title("scaled_img")
                axs[0, 1].imshow(cropped_img)
                axs[0, 1].set_title("cropped_img")
                axs[0, 2].imshow(merged_depth, cmap='gray')
                axs[0, 2].set_title("merged_depth")
                axs[0, 3].imshow(cropped_merged_depth, cmap='gray')
                axs[0, 3].set_title("cropped_merged_depth")
                axs[1, 0].imshow(left_pd, cmap='gray')
                axs[1, 0].set_title("left_pd")
                axs[1, 1].imshow(right_pd, cmap='gray')
                axs[1, 1].set_title("right_pd")
                axs[1, 2].imshow(stack_pd[:,:,0], cmap='gray')
                axs[1, 2].set_title("stack_pd")
                axs[1, 3].imshow(cropped_stack_pd[:,:,0], cmap='gray')
                axs[1, 3].set_title("cropped_stack_pd")

                fig.suptitle("Image ID: filename | gt: {},{}".format(depth, focus))
                fig.show()
                '''
