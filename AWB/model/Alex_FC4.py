import os
from typing import Union, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.nn.functional import normalize
import torchvision.transforms.functional as F

from torchvision.transforms import transforms
from torch.quantization import QuantStub, DeQuantStub

from core.settings import USE_CONFIDENCE_WEIGHTED_POOLING
from core.utils import correct, rescale, scale
from core.Model import Model
from RRAM import my_utils as my
from core.AngularLoss import AngularLoss

class Model(Model):
    def __init__(self,channel_number: int,
                 qn_on: bool,
                 fp_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float,):
        super().__init__()
        self._network = Net_DCIM(channel_number, qn_on, fp_on, weight_bit, output_bit, isint, clamp_std).to(self._device)
        # if qn_on:
        #     self._network = Net_DCIM_QAT(channel_number, qn_on, weight_bit,output_bit,isint,clamp_std).to(self._device)
        # else:
        #     self._network = Net_DCIM(channel_number, qn_on, weight_bit, output_bit, isint, clamp_std).to(self._device)
        self._criterion = AngularLoss(self._device)
    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self._criterion(pred, label)

    def predict(self, img: Tensor, return_steps: bool = False) -> Union[Tensor, Tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, rgb, confidence, a = self._network(img)
            if return_steps:
                return pred, rgb, confidence, a
            return pred
        return self._network(img)

    def optimize(self, img: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(img)
        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def save_vis(self, model_output: dict, path_to_plot: str):
        model_output = {k: v.clone().detach().to(self._device) for k, v in model_output.items()}

        img, label, pred = model_output["img"], model_output["label"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]
        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)
        masked_original = scale(F.to_tensor(original).to(self._device).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_loss(pred, label)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')
class Net_V1(torch.nn.Module):

    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float,
                 ):
        super().__init__()

        def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        self.conv1 = conv2d(in_channels=3,   out_channels=64,   kernel_size=7, stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = conv2d(in_channels=64,  out_channels=64,   kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = conv2d(in_channels=64,  out_channels=128,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv6 = conv2d(in_channels=256, out_channels=64,   kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = conv2d(in_channels=64,  out_channels=4 if USE_CONFIDENCE_WEIGHTED_POOLING else 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        a = {}
        # 第1层
        a['input_img'] = x
        x = self.conv1(x)
        a['conv1_no_relu'] = x
        x = self.relu1(x)
        a['conv1_relu'] = x
        x = self.maxpool1(x)
        a['max_pool2d'] = x
        # 第2层CNN
        x = self.conv2(x)
        a['conv2_no_relu'] = x
        x = self.relu2(x)
        a['conv2_relu'] = x
        x = self.maxpool2(x)
        a['max_pool2d'] = x
        # 第3层
        x = self.conv3(x)
        a['conv3_no_relu'] = x
        x = self.relu3(x)
        a['conv3_relu'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = self.relu4(x)
        a['conv4_relu'] = x
        # 第5层
        x = self.conv5(x)
        a['conv5_no_relu'] = x
        x = self.relu5(x)
        a['conv5_relu'] = x
        x = self.maxpool5(x)
        a['max_pool2d'] = x
        # 第6层
        x = self.conv6(x)
        a['conv6_no_relu'] = x
        x = self.relu6(x)
        a['conv6_relu'] = x
        # 第7层
        x = self.conv7(x)
        a['conv7_no_relu'] = x
        out = self.relu7(x)
        a['conv7_relu'] = out

        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps

        if USE_CONFIDENCE_WEIGHTED_POOLING:
            # Per-patch color estimates (first 3 dimensions)
            #rgb = normalize(out[:, :3, :, :], dim=1)
            rgb = out[:, :3, :, :]
            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]
            #if torch.sum(confidence):
                # Confidence-weighted pooling
            #    pred = normalize(torch.sum(torch.sum(rgb * (confidence+1), 2), 2), dim=1)
            #else:
            #    pred = normalize(torch.sum(torch.sum(rgb, 2), 2), dim=1)
            pred = normalize(torch.sum(torch.sum(rgb * (confidence + 1), 2), 2), dim=1)
            return pred, rgb, confidence, a

        pred = normalize(torch.sum(torch.sum(out, 2), 2), dim=1)
        return pred, a

class Net_in_channel(torch.nn.Module):

    def __init__(self,
                 channel_number: int = 4,
                 qn_on: bool = False,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0,
                 noise_scale: float = 0.075,
                 fp_on: bool = False,):
        super().__init__()
        if(fp_on):
            def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    my.Conv2d_fp8(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias),)
        elif(qn_on):
            def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),)
        else:
            def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),)
        self.conv1 = conv2d(in_channels=channel_number,   out_channels=32,   kernel_size=3, stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = conv2d(in_channels=32,  out_channels=64,   kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = conv2d(in_channels=64,  out_channels=128,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv6 = conv2d(in_channels=256, out_channels=64,   kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = conv2d(in_channels=64,  out_channels=4 if USE_CONFIDENCE_WEIGHTED_POOLING else 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)

    def post_easy(self, out):
        # Per-patch color estimates (last 3 dimensions)
        # rgb = normalize(out[:, 1:4, :, :], dim=1)
        rgb = out[:, 1:4, :, :]
        # Confidence (first dimension)
        confidence = out[:, 0:1, :, :]

        # pred = 1/normalize(torch.sum(torch.sum(rgb * confidence + 1, 2), 2),p=1, dim=1)
        pred = torch.sum(torch.sum(rgb * confidence + 1, 2), 2)
        # pred = pred.sum()/pred
        g=torch.repeat_interleave(pred[:,1].unsqueeze(dim=1), repeats=3, dim=1)
        pred = g / pred
        return pred, rgb, confidence

    def post_norm(self, out):
        # Per-patch color estimates (last 3 dimensions)
        rgb = normalize(out[:, 1:4, :, :], dim=1)

        # Confidence (first dimension)
        confidence = out[:, 0:1, :, :]

        pred = 1 / normalize(torch.sum(torch.sum(rgb * confidence + 1, 2), 2), p=1, dim=1)
        # pred = torch.sum(torch.sum(rgb * confidence + 1, 2), 2)
        # pred = pred.sum()/pred
        # pred = pred[:, 1] / pred
        return pred, rgb, confidence
    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        a = {}
        # 第1层
        a['input_img'] = x
        x = self.conv1(x)
        a['conv1_no_relu'] = x
        x = self.relu1(x)
        a['conv1_relu'] = x
        x = self.maxpool1(x)
        a['max_pool2d'] = x
        # 第2层CNN
        x = self.conv2(x)
        a['conv2_no_relu'] = x
        x = self.relu2(x)
        a['conv2_relu'] = x
        x = self.maxpool2(x)
        a['max_pool2d'] = x
        # 第3层
        x = self.conv3(x)
        a['conv3_no_relu'] = x
        x = self.relu3(x)
        a['conv3_relu'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = self.relu4(x)
        a['conv4_relu'] = x
        # 第5层
        x = self.conv5(x)
        a['conv5_no_relu'] = x
        x = self.relu5(x)
        a['conv5_relu'] = x
        x = self.maxpool5(x)
        a['max_pool2d'] = x
        # 第6层
        x = self.conv6(x)
        a['conv6_no_relu'] = x
        x = self.relu6(x)
        a['conv6_relu'] = x
        # 第7层
        x = self.conv7(x)
        a['conv7_no_relu'] = x
        out = self.relu7(x)
        a['conv7_relu'] = out

        pred, rgb, confidence = self.post_easy(out)

        return pred, rgb, confidence, a

class Net_DCIM(torch.nn.Module):

    def __init__(self,
                 channel_number: int = 4,
                 qn_on: bool = False,
                 fp_on: bool = False,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0):
        super().__init__()
        if (fp_on):
            def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    my.Conv2d_fp8(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias), )
        elif (qn_on):
            def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    my.Conv2d_quant(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      bias=bias),)
        else:
            def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),)
        self.conv1 = conv2d(in_channels=channel_number,   out_channels=32,   kernel_size=3, stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = conv2d(in_channels=32,  out_channels=64,   kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = conv2d(in_channels=64,  out_channels=128,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv6 = conv2d(in_channels=256, out_channels=64,   kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = conv2d(in_channels=64,  out_channels=4 if USE_CONFIDENCE_WEIGHTED_POOLING else 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)

    def post_norm(self, out):
        # Per-patch color estimates (last 3 dimensions)
        rgb = normalize(out[:, 1:4, :, :], dim=1)

        # Confidence (first dimension)
        confidence = out[:, 0:1, :, :]

        # Confidence-weighted pooling
        pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

        return pred, rgb, confidence
    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        a = {}
        # 第1层
        a['input_img'] = x
        x = self.conv1(x)
        a['conv1_no_relu'] = x
        x = self.relu1(x)
        a['conv1_relu'] = x
        x = self.maxpool1(x)
        a['max_pool2d'] = x
        # 第2层CNN
        x = self.conv2(x)
        a['conv2_no_relu'] = x
        x = self.relu2(x)
        a['conv2_relu'] = x
        x = self.maxpool2(x)
        a['max_pool2d'] = x
        # 第3层
        x = self.conv3(x)
        a['conv3_no_relu'] = x
        x = self.relu3(x)
        a['conv3_relu'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = self.relu4(x)
        a['conv4_relu'] = x
        # 第5层
        x = self.conv5(x)
        a['conv5_no_relu'] = x
        x = self.relu5(x)
        a['conv5_relu'] = x
        x = self.maxpool5(x)
        a['max_pool2d'] = x
        # 第6层
        x = self.conv6(x)
        a['conv6_no_relu'] = x
        x = self.relu6(x)
        a['conv6_relu'] = x
        # 第7层
        x = self.conv7(x)
        a['conv7_no_relu'] = x
        out = self.relu7(x)
        a['conv7_relu'] = out

        pred, rgb, confidence = self.post_norm(out)

        return pred, rgb, confidence, a

class Net_DCIM_QAT(torch.nn.Module):

    def __init__(self,
                 channel_number: int = 4,
                 qn_on: bool = False,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0):
        super().__init__()
        def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = conv2d(in_channels=channel_number,   out_channels=32,   kernel_size=3, stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = conv2d(in_channels=32,  out_channels=64,   kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = conv2d(in_channels=64,  out_channels=128,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv6 = conv2d(in_channels=256, out_channels=64,   kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = conv2d(in_channels=64,  out_channels=4 if USE_CONFIDENCE_WEIGHTED_POOLING else 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)

    def post_norm(self, out):
        # Per-patch color estimates (last 3 dimensions)
        rgb = normalize(out[:, 1:4, :, :], dim=1)

        # Confidence (first dimension)
        confidence = out[:, 0:1, :, :]

        # Confidence-weighted pooling
        pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

        return pred, rgb, confidence
    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        a = {}
        # 第1层
        a['input_img'] = x
        x = self.quant(x)
        x = self.conv1(x)
        a['conv1_no_relu'] = x
        x = self.relu1(x)
        a['conv1_relu'] = x
        x = self.maxpool1(x)
        a['max_pool2d'] = x
        # 第2层CNN
        x = self.conv2(x)
        a['conv2_no_relu'] = x
        x = self.relu2(x)
        a['conv2_relu'] = x
        x = self.maxpool2(x)
        a['max_pool2d'] = x
        # 第3层
        x = self.conv3(x)
        a['conv3_no_relu'] = x
        x = self.relu3(x)
        a['conv3_relu'] = x
        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = self.relu4(x)
        a['conv4_relu'] = x
        # 第5层
        x = self.conv5(x)
        a['conv5_no_relu'] = x
        x = self.relu5(x)
        a['conv5_relu'] = x
        x = self.maxpool5(x)
        a['max_pool2d'] = x
        # 第6层
        x = self.conv6(x)
        a['conv6_no_relu'] = x
        x = self.relu6(x)
        a['conv6_relu'] = x
        # 第7层
        x = self.conv7(x)
        a['conv7_no_relu'] = x
        x = self.relu7(x)
        out = self.dequant(x)
        a['conv7_relu'] = out

        pred, rgb, confidence = self.post_norm(out)

        return pred, rgb, confidence, a