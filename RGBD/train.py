import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from core.settings import DEVICE, make_deterministic
from core.utils import print_metrics, log_metrics
from core.Evaluator import Evaluator
from core.LossTracker import LossTracker
from RGBD.code.LoadDataset import Dataset
from RGBD.RGBDNet import Model
from RRAM import my_utils as my

# --------------------------------------------------------------------------------------------------------------------
# ======================================== #
# 训练参数
# ======================================== #
RANDOM_SEED = 0
EPOCHS = 10
EVAL_EPOCH = 1
VIS_EPOCH = EPOCHS / 10

BATCH_SIZE = 16
LEARNING_RATE = 0.0001



# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 1
in_range = 1
qn_on = True
isint = 0
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0.075

version = 5
RELOAD_CHECKPOINT = True
DOWN_SCALE = 16

# ======================================== #
# 保存设置参数
# ======================================== #

SAVE_TB = False
SAVE_LOG = False
if qn_on:
    model_name = "I{}W{}O{}_n{}".format(str(input_bit),str(weight_bit),str(output_bit),noise_scale)
    PATH_TO_PTH_CHECKPOINT = os.path.join("/home/project/xupf/Projects/AI_ISP/RGBD/output/train/I8W4O8_n0.075/model_down{}_{}_V{}.pth".format(DOWN_SCALE,in_range,version))#/model/I8W4O8n0.075_model_V2.pth")#I8W4O8_n0.075
else:
    model_name = "FULL"
    PATH_TO_PTH_CHECKPOINT = os.path.join("/home/project/xupf/Projects/AI_ISP/RGBD/output/train/{}/model_down{}_{}_V{}.pth".format(model_name,DOWN_SCALE,in_range,version))


# --------------------------------------------------------------------------------------------------------------------

def main(opt):
    epochs, batch_size, lr = opt.epochs, opt.batch_size, opt.lr
    if SAVE_TB is True:
        # ======================================== #
        # 实例化SummaryWriter对象
        # ======================================== #
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir='/'.join(["output","tensorboard", model_name]))

    # ======================================== #
    # 定义路径
    # ======================================== #
    # log path
    path_to_log = os.path.join("output", "train",  "{}".format(model_name))
    os.makedirs(path_to_log, exist_ok=True)
    #model path
    #path_to_model = os.path.join("output","trained_models", model_name)
    #os.makedirs(path_to_model, exist_ok=True)
    if SAVE_LOG is True:
        # metrics path
        path_to_metrics_log = os.path.join(path_to_log, "metrics{}.csv".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time())))))
        cfg = ("\n *** Training configuration ***")\
            +("\n Epochs .......... : {}".format(epochs))\
            +("\n Batch size ...... : {}".format(batch_size))\
            +("\n Learning rate ... : {}".format(lr))\
            +("\n Random seed ..... : {}".format(RANDOM_SEED))\
            +("\n *** RRAM configuration ***")\
            +("\n img_quant_flag .. : {}".format(img_quant_flag))\
            +("\n isint ........... : {}".format(isint))\
            +("\n weight_bit ...... : {}".format(weight_bit))\
            +("\n output_bit ...... : {}".format(output_bit))\
            +("\n noise_scale ..... : {}".format(noise_scale))\
            +("\n clamp_std ....... : {}\n".format(clamp_std))
        open(os.path.join(path_to_log, "network{}.txt".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time()))))), 'a+').write(str(cfg))
    # ======================================== #
    # 模型初始化
    # ======================================== #
    model = Model(qn_on = qn_on, version=version, rgb_channel = 3,weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, noise_scale=noise_scale)

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)


    model.print_network()
    #if SAVE_LOG is True:
    # model.log_network((3,256,256),path_to_log)
    model.set_optimizer(lr)
    model.set_scheduler(EPOCHS)
    # ======================================== #
    # 加载数据
    # ======================================== #
    dataset = Dataset(rgb_channel=3)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    training_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=True)
    print("\n Training set size ... : {}".format(len(training_set)))
    print(" Test set size ....... : {}\n".format(len(test_set)))

    # ======================================== #
    # 定义训练和测试函数
    # ======================================== #
    print("\n**************************************************************")
    print("\t\t\t Training FC4")
    print("**************************************************************\n")

    evaluator = Evaluator()
    best_val_loss, best_metrics = 10000.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()


    # ======================================== #
    # 开始训练
    # ======================================== #
    plt_train_loss = []
    plt_valid_loss = []
    for epoch in range(epochs):

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (rgb_image, depth, rawDepth, file_name) in enumerate(training_loader):
            rgb_image, depth, rawDepth = rgb_image.to(DEVICE), depth.to(DEVICE), rawDepth.to(DEVICE)
            # quant
            if img_quant_flag == 1:
                rgb_image, _ = my.data_quantization_sym(rgb_image, half_level=2 ** input_bit / 2 - 1)
                rgb_image = rgb_image.float()*in_range

                depth, _ = my.data_quantization_sym(depth, half_level=2 ** input_bit / 2 - 1)
                depth = depth.float()*in_range
                rawDepth =torch.nn.functional.interpolate(rawDepth, scale_factor=1/DOWN_SCALE)
                rawDepth = torch.nn.functional.interpolate(rawDepth, scale_factor=DOWN_SCALE)

                rawDepth, _ = my.data_quantization_sym(rawDepth, half_level=2 ** input_bit / 2 - 1)
                rawDepth = rawDepth.float()*in_range

            #print(rgb_image.size())
            #print(rawDepth.size())
            pred_depth, loss = model.optimize(rgb_image, rawDepth, depth)
            train_loss.update(loss)
            raw_loss = model.get_loss(rawDepth, depth).cpu().detach().numpy()
            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} | [ improve loss: {:.4f} ]".format(epoch, epochs, i, loss, raw_loss-loss))
        if epoch % VIS_EPOCH  == 0:
            fig, axs = plt.subplots(1, 4)
            showrgb = rgb_image.cpu()[0, :, :, :].squeeze()
            axs[0].imshow(np.transpose(showrgb, (1, 2, 0))/showrgb.max(), cmap='gray')
            showraw = rawDepth.cpu()[0, :, :, :].squeeze()
            showpred = pred_depth.cpu()[0, :, :, :].squeeze()
            axs[1].imshow(showraw, cmap='gray')
            showdepth = depth.cpu()[0, :, :, :].squeeze()
            axs[2].imshow(showdepth, cmap='gray')#, cmap='gray')
            axs[3].imshow(showpred.detach().numpy(), cmap='gray')#, cmap='gray')
            fig.suptitle("Epoch: {} | Train Loss: {:.4f} | Improve: {:.4f} ".format(epoch, loss, raw_loss-loss))
            fig.show()
        train_time = time.time() - start
        val_loss.reset()
        start = time.time()

        if epoch % EVAL_EPOCH == 0:
            evaluator.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")
            with torch.no_grad():
                for i, (rgb_image, depth, rawDepth, file_name) in enumerate(test_loader):
                    rgb_image, depth, rawDepth = rgb_image.to(DEVICE), depth.to(DEVICE), rawDepth.to(DEVICE)
                    # quant
                    if img_quant_flag == 1:
                        rgb_image, _ = my.data_quantization_sym(rgb_image, half_level=2 ** input_bit / 2 - 1)
                        rgb_image = rgb_image.float()*in_range
                        depth, _ = my.data_quantization_sym(depth, half_level=2 ** input_bit / 2 - 1)
                        depth = depth.float()*in_range
                        down_rawDepth = torch.nn.functional.interpolate(rawDepth, scale_factor=1 / DOWN_SCALE)
                        rawDepth = torch.nn.functional.interpolate(down_rawDepth, scale_factor=DOWN_SCALE)

                        rawDepth, _ = my.data_quantization_sym(rawDepth, half_level=2 ** input_bit / 2 - 1)
                        rawDepth = rawDepth.float()*in_range
                    pred_depth, a = model.predict(rgb_image, rawDepth)
                    loss = model.get_loss(pred_depth, depth).cpu().detach().numpy()
                    raw_loss = model.get_loss(rawDepth, depth).cpu().detach().numpy()
                    val_loss.update(loss)
                    evaluator.add_error(loss)

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} | improve loss: {:.4f}]".format(epoch, epochs, i, loss, raw_loss-loss))
        if epoch % VIS_EPOCH == 0:
            fig, axs = plt.subplots(1, 4)
            showrgb = rgb_image.cpu()[0, :, :, :].squeeze()
            axs[0].imshow(np.transpose(showrgb, (1, 2, 0))/showrgb.max(), cmap='gray')
            showraw = down_rawDepth.cpu()[0, :, :, :].squeeze()
            showpred = pred_depth.cpu()[0, :, :, :].squeeze()
            axs[1].imshow(showraw, cmap='gray')
            showdepth = depth.cpu()[0, :, :, :].squeeze()
            axs[2].imshow(showdepth, cmap='gray')  # , cmap='gray')
            axs[3].imshow(showpred.detach().numpy(), cmap='gray')  # , cmap='gray')
            fig.suptitle("Epoch: {} | Valid Loss: {:.4f} | Improve: {:.4f} ".format(epoch, loss, raw_loss-loss))
            fig.show()



            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start
        plt_train_loss.append(train_loss.avg)
        plt_valid_loss.append(val_loss.avg)
        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        tags = ["train_loss", "val_loss", "best_mean", "best_bst25"]

        if SAVE_TB is True:
            tb_writer.add_scalar(tags[0], train_loss.avg, epoch)
            tb_writer.add_scalar(tags[1], val_loss.avg, epoch)
            tb_writer.add_scalar(tags[2], best_metrics["mean"], epoch)
            tb_writer.add_scalar(tags[3], best_metrics["bst25"], epoch)

            weights_keys = model._network.state_dict().keys()
            for key in weights_keys:
                weight_t = model._network.state_dict()[key].cpu().numpy()
                tb_writer.add_histogram(tag=key,
                                        values=weight_t,
                                        global_step=epoch)
            for key in a:
                im = np.squeeze(a[key].cpu().detach().numpy())
                # [C, H, W] -> [H, W, C]
                # im = np.transpose(im, [1, 2, 0])
                tb_writer.add_histogram(tag=key,
                                        values=im,
                                        global_step=epoch)

        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
        print("********************************************************************\n")
        #model.schedule()
        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(path_to_log, "model_down{}_{}_V{}.pth".format(DOWN_SCALE,in_range,version))


        if SAVE_LOG is True:
            log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)
    plt.figure()
    plt.title('train_loss')
    plt.xlabel('Epoch')
    plt.plot(plt_train_loss, color='b', linestyle='-')
    plt.show()
    plt.savefig(os.path.join(path_to_log, "train_loss.png"))

    plt.figure()
    plt.title('valid_loss')
    plt.xlabel('Epoch')
    plt.plot(plt_valid_loss, color='r', linestyle='-')
    plt.show()
    plt.savefig(os.path.join(path_to_log, "valid_loss.png"))
    plt.plot(plt_train_loss, color='b', linestyle='-')
    plt.show()
    plt.savefig(os.path.join(path_to_log, "both_loss.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    print("\n *** Training configuration ***")
    print("\t Epochs .......... : {}".format(opt.epochs))
    print("\t Batch size ...... : {}".format(opt.batch_size))
    print("\t Learning rate ... : {}".format(opt.lr))
    print("\t Random seed ..... : {}".format(opt.random_seed))

    print("\n *** RRAM configuration ***")
    print("\t img_quant_flag .. : {}".format(img_quant_flag))
    print("\t isint ........... : {}".format(isint))
    print("\t input_bit ...... : {}".format(input_bit))
    print("\t weight_bit ...... : {}".format(weight_bit))
    print("\t output_bit ...... : {}".format(output_bit))
    print("\t noise_scale ..... : {}".format(noise_scale))
    print("\t clamp_std ....... : {}".format(clamp_std))

    main(opt)

