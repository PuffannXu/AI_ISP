import os
import cv2
import numpy as np
from AF.code.core.utils import crop, depth2focus
from AF.code.core.settings import ROI_SIZE
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
from AWB.code.core.utils import normalize, linear_to_nonlinear
from AF.code.core.utils import scale, cal_ig


BASE_PATH = os.path.join("/home/project/xupf/Databases/AF")
PATH_TO_LEFT = os.path.join(BASE_PATH, "test_data", "left_pd")
PATH_TO_RIGHT = os.path.join(BASE_PATH, "test_data", "right_pd")
PATH_TO_LABEL = os.path.join(BASE_PATH, "test_labels")


left_pd = np.array(np.load(os.path.join(PATH_TO_LEFT, "home1_0_0_" + 'data.npy')), dtype='uint16')
right_pd = np.array(np.load(os.path.join(PATH_TO_RIGHT, "home1_0_0_" + 'data.npy')), dtype='uint16')
gt = np.array(np.load(os.path.join(PATH_TO_LABEL, "home1_0_0_" + 'label.npy')), dtype='float32')


left_pd_1 = scale(left_pd)

left_pd_2 = normalize(left_pd)

ig = cal_ig(left_pd)
ig1 = cal_ig(left_pd_1)
ig2 = cal_ig(left_pd_2)


#path_to_save = os.path.join(self.__path_to_save, "fold_{}".format(0), file_name)
#os.makedirs(path_to_save, exist_ok=True)
fig, axs = plt.subplots(1, 3)
axs[0].imshow(left_pd,cmap='gray')
axs[0].set_title("raw,ig={}".format(ig))
axs[1].imshow(left_pd_1,cmap='gray')
axs[1].set_title("scale,ig={}".format(ig1))
axs[2].imshow(left_pd_2,cmap='gray')
axs[2].set_title("normalize,ig={}".format(ig2))

fig.suptitle("Image ID: home1_0_0 | gt: {}".format(gt))
fig.show()
#fig.savefig(os.path.join(path_to_save, "pre_img.png"), dpi=200)
for TRAIN in [7]:

    # input path
    BASE_PATH = "E:/Dataset/preprocessed/AF"
    PATH_TO_SCALED_IMAGE = "E:/Dataset/train{}/scaled_images/".format(TRAIN)
    PATH_TO_MERGED_DEPTH = "E:/Dataset/train{}/merged_depth/".format(TRAIN)
    PATH_TO_LEFT_PD = "E:/Dataset/train{}/raw_left_pd/".format(TRAIN)
    PATH_TO_RIGHT_PD = "E:/Dataset/train{}/raw_right_pd/".format(TRAIN)
    # output path
    BASE_PATH_OUT = "E:/Dataset/preprocessed/AF/train{}".format(TRAIN)
    PATH_TO_NUMPY_DATA = os.path.join(BASE_PATH, "numpy_data")
    PATH_TO_NUMPY_LABELS = os.path.join(BASE_PATH, "numpy_labels")
    os.makedirs(os.path.join(BASE_PATH_OUT), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_DATA), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_0"), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_1"), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NUMPY_LABELS, "dif_is_2"), exist_ok=True)
    for filename in os.listdir(PATH_TO_SCALED_IMAGE):
        # 读入scaled image 378*504
        print("filename is:{}".format(filename))
        path_to_scaled_img = os.path.join(PATH_TO_SCALED_IMAGE, filename,
                                          "{}/result_scaled_image_center.jpg".format(20))
        # scaled_img = Image.open(path_to_scaled_img).convert("RGB")
        scaled_img = cv2.imread(path_to_scaled_img)
        cropped_img = crop(scaled_img, 0.5, 0.5, ROI_SIZE / 4)  # 256/4 = 64
        path_to_merged_depth = os.path.join(PATH_TO_MERGED_DEPTH, filename,
                                            "result_merged_depth_center.png")
        merged_depth = cv2.imread(path_to_merged_depth)
        merged_depth = cv2.cvtColor(merged_depth, cv2.COLOR_BGR2GRAY)
        cropped_merged_depth = crop(merged_depth, 0.5, 0.5, ROI_SIZE / 4)  # 256/4 = 64
        depth = np.median(cropped_merged_depth)
        focus, min_diff = depth2focus(depth)
        print("depth: {}".format(depth))
        print("focus: {}".format(focus))
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(scaled_img)
        axs[0].set_title("scaled_img")
        axs[1].imshow(cropped_img)
        axs[1].set_title("cropped_img")
        axs[2].imshow(merged_depth, cmap='gray')
        axs[2].set_title("merged_depth")
        axs[3].imshow(cropped_merged_depth, cmap='gray')
        axs[3].set_title("cropped_merged_depth")

        fig.suptitle("Image ID: filename | gt: {},{}".format(depth, focus))
        fig.show()