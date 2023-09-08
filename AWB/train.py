import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from AWB.code.core.settings import DEVICE, USE_CONFIDENCE_WEIGHTED_POOLING, make_deterministic
from AWB.code.core.utils import print_metrics, log_metrics
from AWB.code.core.Evaluator import Evaluator
from AWB.code.core.LossTracker import LossTracker
from AWB.code.ColorCheckerDataset import ColorCheckerDataset
from AWB.model.Alex_FC4 import Model
from RRAM import my_utils as my

# --------------------------------------------------------------------------------------------------------------------
# ======================================== #
# 训练参数
# ======================================== #
RANDOM_SEED = 0
EPOCHS = 51
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
FOLD_NUM = 0

RELOAD_CHECKPOINT = True
# ======================================== #
# 量化训练参数
# ======================================== #
img_quant_flag = 1
qn_on = True
isint = 0
input_bit = 8
weight_bit = 4
output_bit = 8
clamp_std = 0
noise_scale = 0.075

if qn_on:
    model_name = "I{}W{}O{}_n{}".format(str(input_bit),str(weight_bit),str(output_bit),noise_scale)
    PATH_TO_PTH_CHECKPOINT = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/output/train/I8W4O8_n0.075_fold_0/model_VAlex.pth")
else:
    model_name = "FULL"
    PATH_TO_PTH_CHECKPOINT = os.path.join("/home/project/xupf/Projects/AI_ISP/AWB/output/train/FULL_fold_0")
# The subset of test images to be monitored (set to empty list to skip saving visualizations and speed up training)
# For example: TEST_VIS_IMG = ["IMG_0753", "IMG_0438", "IMG_0397"]
TEST_VIS_IMG = []





# --------------------------------------------------------------------------------------------------------------------

def main(opt):
    fold_num, epochs, batch_size, lr = opt.fold_num, opt.epochs, opt.batch_size, opt.lr
    # ======================================== #
    # 实例化SummaryWriter对象
    # ======================================== #
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir='/'.join(["output","tensorboard", model_name]))

    # ======================================== #
    # 定义路径
    # ======================================== #
    # log path
    path_to_log = os.path.join("output", "train",  "{}_fold_{}".format(model_name,str(fold_num)))
    os.makedirs(path_to_log, exist_ok=True)
    #model path
    #path_to_model = os.path.join("output","trained_models", model_name)
    #os.makedirs(path_to_model, exist_ok=True)
    # metrics path
    path_to_metrics_log = os.path.join(path_to_log, "metrics{}.csv".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time())))))
    cfg = ("\n *** Training configuration ***")\
        +("\n Fold num ........ : {}".format(fold_num))\
        +("\n Epochs .......... : {}".format(epochs))\
        +("\n Batch size ...... : {}".format(batch_size))\
        +("\n Learning rate ... : {}".format(lr))\
        +("\n Random seed ..... : {}".format(RANDOM_SEED))\
        +("\n *** RRAM configuration ***")\
        +("\n img_quant_flag .. : {}".format(img_quant_flag)) \
        + ("\n qn_on ........... : {}".format(qn_on)) \
         +("\n isint ........... : {}".format(isint))\
        +("\n weight_bit ...... : {}".format(weight_bit))\
        +("\n output_bit ...... : {}".format(output_bit))\
        +("\n noise_scale ..... : {}".format(noise_scale))\
        +("\n clamp_std ....... : {}\n".format(clamp_std))
    open(os.path.join(path_to_log, "network{}.txt".format(str(time.strftime('%Y%m%d_%H%M',time.localtime(time.time()))))), 'a+').write(str(cfg))
    # vis path
    path_to_vis = os.path.join(path_to_log, "test_vis")
    if TEST_VIS_IMG:
        print("Test vis for monitored image {} will be saved at {}\n".format(TEST_VIS_IMG, path_to_vis))
        os.makedirs(path_to_vis)

    # ======================================== #
    # 模型初始化
    # ======================================== #
    model = Model(qn_on, weight_bit, output_bit, isint, clamp_std, noise_scale)
    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network((3,256,256),path_to_log)
    model.set_optimizer(lr)
    # ======================================== #
    # 加载数据
    # ======================================== #
    training_set = ColorCheckerDataset(train=True, folds_num=fold_num)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    print("\n Training set size ... : {}".format(len(training_set)))

    test_set = ColorCheckerDataset(train=False, folds_num=fold_num)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=True)
    print(" Test set size ....... : {}\n".format(len(test_set)))

    # ======================================== #
    # 定义训练和测试函数
    # ======================================== #
    print("\n**************************************************************")
    print("\t\t\t Training FC4 - Fold {}".format(fold_num))
    print("**************************************************************\n")

    evaluator = Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
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

        for i, (img, label, file_name) in enumerate(training_loader):
            label = label[:,1].unsqueeze(1) / label
            img, label = img.to(DEVICE), label.to(DEVICE)
            # quant
            if img_quant_flag == 1:
                img, _ =  my.data_quantization_sym(img, half_level=2 ** input_bit / 2 - 1)
                #img = img / (2 ** input_bit)
                img = img.float()
            loss = model.optimize(img, label)
            train_loss.update(loss)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} ]".format(epoch, epochs, i, loss))
        train_time = time.time() - start
        val_loss.reset()
        start = time.time()
        plt_train_loss.append(train_loss.avg)

        if epoch % 5 == 0:
            evaluator.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")
            plt_valid_loss.append(val_loss.avg)
            with torch.no_grad():
                for i, (img, label, file_name) in enumerate(test_loader):
                    label = label[:,1].unsqueeze(1)/label
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    # quant
                    if img_quant_flag == 1:
                        img, _ = my.data_quantization_sym(img, scale=1, half_level=2 ** input_bit / 2 - 1)
                        #img = img / (2 ** input_bit)
                        img = img.float()
                    pred, rgb, confidence, a = model.predict(img, return_steps=True)
                    loss = model.get_loss(pred, label).item()
                    val_loss.update(loss)

                    evaluator.add_error(model.get_loss(pred, label).item())

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} ]".format(epoch, epochs, i, loss))

                    img_id = file_name[0].split(".")[0]
                    if USE_CONFIDENCE_WEIGHTED_POOLING:
                        if img_id in TEST_VIS_IMG:
                            model.save_vis({"img": img, "label": label, "pred": pred, "rgb": rgb, "c": confidence},
                                           os.path.join(path_to_vis, img_id, "epoch_{}.png".format(epoch)))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        tags = ["train_loss", "val_loss", "best_mean", "best_bst25"]
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

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(path_to_log, "model_V{}.pth".format("Alex"))

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)
    plt.figure()
    plt.title('train_loss')
    plt.xlabel('Epoch')

    plt.plot(plt_train_loss, color='b', linestyle='-')
    plt.show()
    plt.figure()
    plt.title('valid_loss')
    plt.xlabel('Epoch')

    plt.plot(plt_valid_loss, color='r', linestyle='-')
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_num", type=int, default=FOLD_NUM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    print("\n *** Training configuration ***")
    print("\t Fold num ........ : {}".format(opt.fold_num))
    print("\t Epochs .......... : {}".format(opt.epochs))
    print("\t Batch size ...... : {}".format(opt.batch_size))
    print("\t Learning rate ... : {}".format(opt.lr))
    print("\t Random seed ..... : {}".format(opt.random_seed))

    print("\n *** RRAM configuration ***")
    print("\t img_quant_flag .. : {}".format(img_quant_flag))
    print("\t qn_on ........... : {}".format(qn_on))
    print("\t isint ........... : {}".format(isint))
    print("\t input_bit ...... : {}".format(input_bit))
    print("\t weight_bit ...... : {}".format(weight_bit))
    print("\t output_bit ...... : {}".format(output_bit))
    print("\t noise_scale ..... : {}".format(noise_scale))
    print("\t clamp_std ....... : {}".format(clamp_std))

    main(opt)

