import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from AF.code.core.settings import DEVICE, make_deterministic
from AF.code.core.utils import print_metrics, log_metrics, scale
from AF.code.core.Evaluator import Evaluator
from AF.code.core.LossTracker import LossTracker
from AF.code.LoadDataset import Dataset
from AF.model.AFNet import Model
from RRAM import my_utils as my

# Set to -1 to process all the samples in the test set of the current fold
NUM_SAMPLES = -1

# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 1
isint = 0
qn_on = 1
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0.075
version = 2

PATH_TO_SAVED = os.path.join("/home/project/xupf/Projects/AI_ISP/AF/output/test", "{}".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time())))))

def main():
    evaluator = Evaluator()
    model = Model(qn_on = qn_on, version=version, weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, noise_scale=noise_scale)
    os.makedirs(PATH_TO_SAVED,exist_ok=True)

    test_set = Dataset("/home/project/xupf/Databases/AF_new/ground_truth_train0.txt")
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    path_to_pretrained = os.path.join("/home/project/xupf/Projects/AI_ISP/AF/output/train/I8W4O8_n0.075/model_V2.pth")
        #"/home/project/xupf/Projects/AI_ISP/AF/model/model_V2.pth")
    # path_to_pretrained = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/output/train/FULL_fold_0/model.pth")
    model.load(path_to_pretrained)
    model.evaluation_mode()

    print("\n *** Testing *** \n")
    print(" * Test set size: {}".format(len(test_set)))
    print(" * Using pretrained model stored at: {} \n".format(path_to_pretrained))
    with open(os.path.join(PATH_TO_SAVED, 'test.csv'), 'a') as f:
        print("num", "file_name", "gt", "pred", "loss", sep=',', end='\n', file=f)
    with torch.no_grad():
        for i, (img, label, file_name) in enumerate(dataloader):
            if NUM_SAMPLES > -1 and i > NUM_SAMPLES - 1:
                break

            img, label = img.to(DEVICE), label.to(DEVICE)
            if img_quant_flag == 1:
                img, _ = my.data_quantization_sym(img, half_level=2 ** input_bit / 2 - 1)
                # img = img / (2 ** input_bit)
                img = img.float()
            pred, a = model.predict(img, return_steps=True)
            loss = model.get_loss(pred, label).item()
            evaluator.add_error(loss)
            file_name = file_name[0].split(".")[0]
            print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name, i, loss))
            with open(os.path.join(PATH_TO_SAVED, 'test.csv'), 'a') as f:
                print(i, file_name, label, pred, loss, sep=',', end='\n', file=f)


    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["wst5"]))
    with open(os.path.join(PATH_TO_SAVED, 'test.csv'), 'a') as f:
        print(NUM_SAMPLES, metrics, sep=',', end='\n', file=f)


if __name__ == '__main__':
    main()