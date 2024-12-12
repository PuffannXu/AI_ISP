import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn.functional import normalize
from AWB.code.core.settings import DEVICE
from AWB.code.core.utils import correct, rescale, scale, linear_to_nonlinear
from AWB.code.core.Evaluator import Evaluator
from AWB.code.ColorCheckerDataset import ColorCheckerDataset
from AWB.model.Alex_FC4 import Model
from RRAM import my_utils as my

# Set to -1 to process all the samples in the test set of the current fold
NUM_SAMPLES = -1

# The number of folds to be processed (either 1, 2 or 3)
NUM_FOLDS = 3
model_name = "AWB_fp8_w_hw_group288_epoch5"
# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 0
isint = 0
qn_on = False
fp_on = 2 #0:off 1:wo hw 2:hw
quant_type = "group" #"layer" "channel" "group"
group_number = 288
input_bit = 4
weight_bit = 4
output_bit = 4
clamp_std = 0
noise_scale = 0
channel_number = 4
height = 224
width = 224
# Where to save the generated visualizations
if fp_on == 0:
    PATH_TO_SAVED = os.path.join(f"/home/project/xupf/Projects/AI_ISP/AWB/output/vis/{model_name}_qn{qn_on}isint{isint}I{input_bit}W{weight_bit}_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
elif fp_on==1:
    PATH_TO_SAVED = os.path.join(f"/home/project/xupf/Projects/AI_ISP/AWB/output/vis/{model_name}_fp8_wo_hw_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
elif fp_on==2:
    if quant_type == "group":
        PATH_TO_SAVED = os.path.join(f"/home/project/xupf/Projects/AI_ISP/AWB/output/vis/{model_name}_fp8_w_hw_{quant_type}{group_number}_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")
    else:
        PATH_TO_SAVED = os.path.join(f"/home/project/xupf/Projects/AI_ISP/AWB/output/vis/{model_name}_fp8_w_hw_{quant_type}_{str(time.strftime('%Y%m%d_%H%M', time.localtime(time.time())))}")

def main():
    evaluator = Evaluator()
    print(f"group_num is {group_number}")
    model = Model(channel_number, qn_on, fp_on, weight_bit, output_bit, isint, clamp_std, noise_scale, quant_type, group_number )
    os.makedirs(PATH_TO_SAVED,exist_ok=True)

    # 设置日志
    logger = my.setup_logger(name='FP8Logger', log_file=f'{PATH_TO_SAVED}/vis.log')

    for num_fold in range(NUM_FOLDS):
        test_set = ColorCheckerDataset(train=False, folds_num=num_fold)
        dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
        #path_to_pretrained = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/model/I8W4O8_n0.075_AWB_V2.pth")
        path_to_pretrained = os.path.join(f"/home/project/xupf/Projects/AI_ISP/model_save/{model_name}.pth")
        model.load(path_to_pretrained)
        model.evaluation_mode()

        print("\n *** Generating visualizations for FOLD {} *** \n".format(num_fold))
        print(" * Test set size: {}".format(len(test_set)))
        print(" * Using pretrained model stored at: {} \n".format(path_to_pretrained))
        img_rgb = np.empty((1,width,height,3),dtype = np.uint8)
        with torch.no_grad():
            for i, (img, label, file_name) in enumerate(dataloader):
                if NUM_SAMPLES > -1 and i > NUM_SAMPLES - 1:
                    break

                img, label = img.to(DEVICE), label.to(DEVICE)
                img_rgb[:,:,:,0] = img[:,0,:,:].cpu()
                img_rgb[:,:,:,1] = img[:,1,:,:].cpu()
                img_rgb[:,:,:,2] = img[:,3,:,:].cpu()
                original_fp32 = transforms.ToPILImage()(img_rgb.squeeze()).convert("RGB")
                if img_quant_flag == 1:
                    img, _ = my.data_quantization_sym(img, half_level=2 ** input_bit / 2 - 1)
                    # img = img / (2 ** input_bit)
                img = img.float()
                pred, rgb, confidence,a = model.predict(img, return_steps=True)

                loss = model.get_loss(pred, label).item()
                evaluator.add_error(loss)
                file_name = file_name[0].split(".")[0]
                print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name, i, loss))
                saveloss = [[file_name,i,loss]]
                save = pd.DataFrame(data=saveloss)

                path_to_save = os.path.join(PATH_TO_SAVED, "loss.csv")
                save.to_csv(path_to_save, mode='a', header=False,index=False)

                original = transforms.ToPILImage()(img_rgb.squeeze())#.convert("RGB")
                original_gamma = linear_to_nonlinear(original)
                gt_corrected, est_corrected = correct(original_fp32, label), correct(original, pred)
                gt_corrected_gamma = linear_to_nonlinear(gt_corrected)
                est_corrected_gamma = linear_to_nonlinear(est_corrected)
                size = original.size[::-1]
                rgb = normalize(rgb, dim = 1)
                scaled_rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
                scaled_confidence = rescale(confidence, size).squeeze(0).permute(1, 2, 0)

                weighted_est = scale(rgb * confidence)
                scaled_weighted_est = rescale(weighted_est, size).squeeze().permute(1, 2, 0)
                original_tensor = F.to_tensor(original)
                original_tensor = original_tensor.to(DEVICE)
                masked_original = scale(original_tensor.permute(1, 2, 0) * scaled_confidence)
                if file_name == "IMG_0822" or file_name == "IMG_0371":
                    fig, axs = plt.subplots(2, 4)

                    axs[0, 0].imshow(original)
                    axs[0, 0].set_title("Original_int8")
                    axs[0, 0].axis("off")

                    axs[0, 1].imshow(original_gamma)
                    axs[0, 1].set_title("Originl_gamma")
                    axs[0, 1].axis("off")

                    axs[0, 2].imshow(gt_corrected_gamma)
                    axs[0, 2].set_title("GT_gamma")
                    axs[0, 2].axis("off")

                    axs[0, 3].imshow(est_corrected_gamma)
                    axs[0, 3].set_title("Predict_gamma")
                    axs[0, 3].axis("off")

                    axs[1, 0].imshow(scaled_rgb.cpu())
                    axs[1, 0].set_title("RGB Mask")
                    axs[1, 0].axis("off")

                    axs[1, 1].imshow(scaled_confidence.cpu(), cmap="gray")
                    axs[1, 1].set_title("Confidence")
                    axs[1, 1].axis("off")

                    axs[1, 2].imshow(scaled_weighted_est.cpu())
                    axs[1, 2].set_title("RGB+Confidence")
                    axs[1, 2].axis("off")

                    axs[1, 3].imshow(est_corrected)
                    axs[1, 3].set_title("Predict")
                    axs[1, 3].axis("off")
                    if fp_on==2 and quant_type=="group":
                        fig.suptitle(f"Image ID: {file_name} | Error: {loss:.4f} | fp8_w_hw_{quant_type}{group_number} \n "
                                     f"ill_gt:{label.cpu().numpy()[0]}\n "
                                     f"pred_gt: {pred.cpu().numpy()[0]}")
                    elif fp_on == 2:
                        fig.suptitle(f"Image ID: {file_name} | Error: {loss:.4f} | fp8_w_hw_{quant_type} \n "
                                     f"ill_gt:{label.cpu().numpy()[0]}\n "
                                     f"pred_gt: {pred.cpu().numpy()[0]}")
                    elif fp_on == 1:
                        fig.suptitle(f"Image ID: {file_name} | Error: {loss:.4f} | fp8_wo_hw \n "
                                     f"ill_gt:{label.cpu().numpy()[0]}\n "
                                     f"pred_gt: {pred.cpu().numpy()[0]}")
                    elif qn_on == 1:
                        fig.suptitle(f"Image ID: {file_name} | Error: {loss:.4f} | I{input_bit}W{weight_bit} \n "
                                     f"ill_gt:{label.cpu().numpy()[0]}\n "
                                     f"pred_gt: {pred.cpu().numpy()[0]}")
                    else:
                        fig.suptitle(f"Image ID: {file_name} | Error: {loss:.4f} | fp32 \n "
                                     f"ill_gt:{label.cpu().numpy()[0]}\n "
                                     f"pred_gt: {pred.cpu().numpy()[0]}")
                    fig.tight_layout(pad=1.2)

                    if loss > 0:
                        fig.show()

                    path_to_save = os.path.join(PATH_TO_SAVED, "fold_{}".format(num_fold), file_name)
                    os.makedirs(path_to_save, exist_ok=True)
                    fig.savefig(os.path.join(path_to_save, "stages.png"), dpi=200)
                    # original.save(os.path.join(path_to_save, "original.png"))
                    # est_corrected.save(os.path.join(path_to_save, "est_corrected.png"))
                    # gt_corrected.save(os.path.join(path_to_save, "gt_corrected.png"))

                    plt.clf()






    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    # if fp_on == 2 and quant_type == "group":
    #     for group_number in [1,9,72,288]:
    #         print(f'==================== group_number is {group_number} ====================')
    #         main()
    # else:
    main()