import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from core.utils import normalize, linear_to_nonlinear, bgr_to_rgb
from core.DataAugmenter import DataAugmenter as da

PATH_TO_IMAGES = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/code/dataset/image")
PATH_TO_COORDINATES = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/code/dataset/coordinates")
PATH_TO_CC_METADATA = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/code/dataset/metadata.txt")

BASE_PATH = "/home/project/xupf/Projects/AI_ISP/AWB/code/dataset/preprocessed"
PATH_TO_NUMPY_DATA = os.path.join(BASE_PATH, "numpy_data")
PATH_TO_NUMPY_LABELS = os.path.join(BASE_PATH, "numpy_labels")
PATH_TO_LINEAR_IMAGES = os.path.join(BASE_PATH, "linear_images")
PATH_TO_GT_CORRECTED = os.path.join(BASE_PATH, "gt_corrected")


print("\n=================================================\n")
print("\t Masking MCC charts")
print("\n=================================================\n")
print("Paths: \n"
      "\t - Numpy data generated at ..... : {} \n"
      "\t - Numpy labels generated at ... : {} \n"
      "\t - Images fetched from ......... : {} \n"
      "\t - Coordinates fetched from .... : {} \n"
      .format(PATH_TO_NUMPY_DATA, PATH_TO_NUMPY_LABELS, PATH_TO_IMAGES, PATH_TO_COORDINATES))

os.makedirs(PATH_TO_NUMPY_DATA, exist_ok=True)
os.makedirs(PATH_TO_NUMPY_LABELS, exist_ok=True)
os.makedirs(PATH_TO_LINEAR_IMAGES, exist_ok=True)
os.makedirs(PATH_TO_GT_CORRECTED, exist_ok=True)

print("Processing images at {} \n".format(PATH_TO_CC_METADATA))




def load_image_without_mcc(file_name: str, mcc_coord: np.ndarray) -> np.ndarray:
    """ Masks the Macbeth Color Checker in the image with a black polygon """
    raw = load_image(file_name)

    # Clip the values between 0 and 1
    img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)

    # Get the vertices of polygon the
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)

    # Fill the polygon to img
    cv2.fillPoly(img, pts=[polygon], color=(1e-5,) * 3)

    return img


def load_image(file_name: str) -> np.ndarray:
    raw = np.array(cv2.imread(os.path.join(PATH_TO_IMAGES, file_name), -1), dtype='float32')

    # Handle pictures taken with Canon 5d Mark III
    black_point = 129 if file_name.startswith('IMG') else 1

    # Keep only the pixels such that raw - black_point > 0
    return np.maximum(raw - black_point, [0, 0, 0])


def get_mcc_coord(file_name: str) -> np.ndarray:
    """ Computes the relative MCC coordinates for the given image """

    lines = open(os.path.join(PATH_TO_COORDINATES, file_name.split('.')[0] + "_macbeth.txt"), 'r').readlines()
    width, height = map(float, lines[0].split())
    scale_x, scale_y = 1 / width, 1 / height

    polygon = []
    for line in [lines[1], lines[2], lines[4], lines[3]]:
        line = line.strip().split()
        x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
        polygon.append((x, y))

    return np.array(polygon, dtype='float32')

# Generate numpy dataset
for img_metadata in tqdm(open(PATH_TO_CC_METADATA, 'r').readlines(), desc="Preprocessing images"):
    _, file_name, r, g, b = img_metadata.strip().split(' ')

    img_without_mcc = load_image_without_mcc(file_name, get_mcc_coord(file_name))
    unaugment_linear_bgr_image = torch.from_numpy(normalize(img_without_mcc).copy())
    #np.save(os.path.join(PATH_TO_NUMPY_DATA, file_name), img_without_mcc)

    illuminant = [float(r), float(g), float(b)]
    #np.save(os.path.join(PATH_TO_NUMPY_LABELS, file_name), illuminant)

    vis_img = Image.fromarray((linear_to_nonlinear(bgr_to_rgb(normalize(img_without_mcc))) * 255).astype(np.uint8))
    #vis_img.save(os.path.join(PATH_TO_LINEAR_IMAGES, file_name))

    #img = Image.fromarray((bgr_to_rgb(normalize(img_without_mcc)) * 255).astype(np.uint8))
    #gt_corrected = correct(img, torch.from_numpy(np.array([illuminant])))
    #gt_corrected.save(os.path.join(PATH_TO_GT_CORRECTED, file_name))



    img, illuminant = da.augment(img_without_mcc, illuminant)
    #img = cv2.resize(img_without_mcc,(256, 256))

    cropped_linear_bgr_image = torch.from_numpy(normalize(img).copy())
    cropped_linear_rgb_image = torch.from_numpy(bgr_to_rgb(normalize(img)).copy())
    cropped_nonlinear_rgb_image = torch.from_numpy(linear_to_nonlinear(bgr_to_rgb(normalize(img))).copy())


    #path_to_save = os.path.join(self.__path_to_save, "fold_{}".format(0), file_name)
    #os.makedirs(path_to_save, exist_ok=True)
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(cropped_linear_bgr_image)
    axs[0].set_title("linear_bgr")
    axs[1].imshow(cropped_linear_rgb_image)
    axs[1].set_title("linear_rgb")
    axs[2].imshow(cropped_nonlinear_rgb_image)
    axs[2].set_title("nonlinear_rgb")
    axs[3].imshow(unaugment_linear_bgr_image)
    axs[3].set_title("unaugment")
    fig.suptitle("Image ID: {} | gt: {}".format(file_name, illuminant))
    fig.show()
    #fig.savefig(os.path.join(path_to_save, "pre_img.png"), dpi=200)