from typing import Union, Tuple

import torch
from torch import nn, Tensor

from core.Model import Model
from AF.code.core.Network import Network
from RRAM import my_utils as my
import torch.nn.functional as F


class Model(Model):
    def __init__(self,
                 qn_on: bool, version: int,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                ):
        super().__init__()
        if version == 1:
            self._network = Net_V1(qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)
        elif version == 2:
            self._network = Net_V2(qn_on=qn_on, weight_bit=weight_bit, output_bit=output_bit,
                                   isint=isint, clamp_std=clamp_std, noise_scale=noise_scale).to(self._device)

    def predict(self, img: Tensor, return_steps: bool = False) -> Union[Tensor, Tuple]:
        return self._network(img)
    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        #loss = nn.MSELoss()
        loss = nn.SmoothL1Loss()
        pred = pred.to(torch.float32)
        label = label.to(torch.float32)
        pred = pred.squeeze()
        label = label.squeeze()
        mse = loss(pred.to(self._device), label.to(self._device))
        #rmse = mse**0.5
        return mse
    def optimize(self, img: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred, _ = self.predict(img)
        #pred = torch.stack(pred)
        #label = torch.stack(label)
        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()

class Net_V1(Network):

    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()
        def conv2d(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )

        self.conv1 = conv2d(in_channels=1,   out_channels=8,   kernel_size=5, stride=2, padding=2, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv2d(in_channels=8,   out_channels=32,  kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv2d(in_channels=32,  out_channels=64,  kernel_size=3, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d(in_channels=64,  out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = conv2d(in_channels=128, out_channels=64,  kernel_size=3, stride=2, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = conv2d(in_channels=64,  out_channels=32,  kernel_size=3, stride=2, padding=1, bias=False)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = conv2d(in_channels=32,  out_channels=1,   kernel_size=3, stride=2, padding=1, bias=False)
        self.relu9 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x: Tensor) ->Union[Tensor,tuple]:
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
        # 第2层CNN
        x = self.conv2(x)
        a['conv2_no_relu'] = x
        x = self.relu2(x)
        a['conv2_relu'] = x
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
        # 第6层
        x = self.conv6(x)
        a['conv6_no_relu'] = x
        # 第7层
        x = self.conv7(x)
        a['conv7_no_relu'] = x
        # 第8层
        x = self.conv8(x)
        a['conv8_no_relu'] = x
        # 第9层
        x = self.conv9(x)
        a['conv9_no_relu'] = x
        out = x.squeeze() / 2
        return out, a


class Net_V2(Network):

    def __init__(self,
                 qn_on: bool,
                 weight_bit: int, output_bit: int,
                 isint: int,
                 clamp_std: int, noise_scale: float
                 ):
        super().__init__()

        def conv2d(in_channels, out_channels, kernel_size, stride, padding, bias):
            return nn.Sequential(
                my.Conv2d_quant_noise(qn_on=qn_on, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std,
                                      noise_scale=noise_scale,
                                      bias=bias),
            )
        def fc(in_channels, out_channels, bias):
            return nn.Sequential(
                my.Linear_quant_noise(qn_on=qn_on, in_features=in_channels, out_features=out_channels, weight_bit=weight_bit, output_bit=output_bit, isint=isint,
                                      clamp_std=clamp_std, noise_scale=noise_scale, bias=bias),
            )
        self.conv1 = conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv2 = conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5 = conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = fc(512, 256, bias=True)
        self.fc2 = fc(256, 64, bias=True)
        self.fc3 = fc(64, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Union[Tensor, tuple]:
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
        x = F.relu(x)
        a['conv1_relu'] = x
        x = F.max_pool2d(x, kernel_size=(2, 2))
        a['max_pool2d'] = x

        # 第2层CNN
        x = self.conv2(x)
        a['conv2_no_relu'] = x
        x = F.relu(x)
        a['conv2_relu'] = x
        x = F.max_pool2d(x, kernel_size=(2, 2))
        a['max_pool2d'] = x

        # 第3层
        x = self.conv3(x)
        a['conv3_no_relu'] = x
        x = F.relu(x)
        a['conv3_relu'] = x
        x = F.max_pool2d(x, kernel_size=(2, 2))
        a['max_pool2d'] = x

        # 第4层
        x = self.conv4(x)
        a['conv4_no_relu'] = x
        x = F.relu(x)
        a['conv4_relu'] = x

        # 第5层
        x = self.conv5(x)
        a['conv5_no_relu'] = x
        x = F.relu(x)
        a['conv5_relu'] = x

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        a['fc1_no_relu'] = x
        x = F.relu(x)
        a['fc1_relu'] = x
        #x = self.dropout(x)
        #a['dropout'] = x

        x = self.fc2(x)
        a['fc2_no_relu'] = x
        x = F.relu(x)
        a['fc2_relu'] = x
        #x = self.dropout(x)
        #a['dropout'] = x

        out = self.fc3(x)
        a['fc3_no_relu'] = out
        #x = self.dropout(x)
        #a['dropout'] = x

        #out = out.squeeze() / 2
        return out, a