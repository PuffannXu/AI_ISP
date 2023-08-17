# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: kidwz
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torch.nn import init
from torch.nn.parameter import Parameter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_quantization_sym(data_float, half_level=15, scale=None,
                          isint=0, clamp_std=0, boundary_refine=True,
                          reg_shift_mode=False, reg_shift_bits=None):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    # reg_shift_mode -> force half_level to be exponent of 2, i.e., half_level = 2^n (n is integer)

    if half_level <= 0:
        return data_float, 1

    if boundary_refine:
        half_level += 0.4999

    if clamp_std:
        std = data_float.std()
        data_float[data_float < (clamp_std * -std)] = (clamp_std * -std)
        data_float[data_float > (clamp_std * std)] = (clamp_std * std)

    if scale == None or scale == 0:
        scale = abs(data_float).max()

    if scale == 0:
        return data_float, 1

    if reg_shift_mode:
        if reg_shift_bits != None:
            quant_scale = 2 ** reg_shift_bits
        else:
            shift_bits = round(math.log(1 / scale * half_level, 2))
            quant_scale = 2 ** shift_bits
        data_quantized = (data_float * quant_scale).round()
        #print(f'quant_scale = {quant_scale}')
        #print(f'reg_shift_bits = {reg_shift_bits}')
    else:
        data_quantized = (data_float / scale * half_level).round()
        quant_scale = 1 / scale * half_level

    if isint == 0:
        data_quantized = data_quantized * scale / half_level
        quant_scale = 1

    return data_quantized, quant_scale


# Add noise to input data
def add_noise(weight, method='add', n_scale=0.074, n_range='max'):
    # weight -> input data, usually a weight
    # method ->
    #   'add' -> add a Gaussian noise to the weight, preferred method
    #   'mul' -> multiply a noise factor to the weight, rarely used
    # n_scale -> noise factor
    # n_range ->
    #   'max' -> use maximum range of weight times the n_scale as the noise std, preferred method
    #   'std' -> use weight std times the n_scale as the noise std, rarely used
    std = weight.std()
    w_range = weight.max() - weight.min()

    if n_range == 'max':
        factor = w_range
    if n_range == 'std':
        factor = std

    if method == 'add':
        w_noise = factor * n_scale * torch.randn_like(weight)
        weight_noise = weight + w_noise
    if method == 'mul':
        w_noise = torch.randn_like(weight) * n_scale + 1
        weight_noise = weight * w_noise
    return weight_noise


# ================================== #
# Autograd Functions
# ================================== #
class Round_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.round()
        # ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Quantize weight and add noise
class Weight_Quant_Noise(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, bias, half_level, isint, clamp_std, noise_scale):
        # weight -> input weight
        # bias -> input bias
        # half_level -> quantization level
        # isint -> return int (will result in scaling) or float (same scale)
        # clamp_std -> clamp weight to [- std * clamp_std, std * clamp_std]
        # noise_scale -> noise scale, equantion can be found in add_noise()
        ctx.save_for_backward()

        std = weight.std()
        if clamp_std != 0:
            weight = torch.clamp(weight, min=-clamp_std * std, max=clamp_std * std)

        # log down the max scale for input weight
        scale_in = abs(weight).max()

        # log down the max scale for input weight
        weight, scale = data_quantization_sym(weight, half_level, scale=scale_in,
                                              isint=isint, clamp_std=0)
        # add noise to weight
        weight = add_noise(weight, n_scale=noise_scale)

        # No need for bias quantization, since the bias is added to the feature map on CPU (or GPU)
        if bias != None:
            # bias = bias / scale
            bias, _ = data_quantization_sym(bias, 127,
                                            isint=isint, clamp_std=0)
            # bias = add_noise(bias, n_scale=noise_scale)

        return weight, bias

    # Use default gradiant to train the network
    # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
    # number of return in def forward() (return weight, bias)
    @staticmethod
    def backward(ctx, weight_grad, bias_grad):
        return weight_grad, bias_grad, None, None, None, None


class Feature_Quant_noise(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, half_level, scale, isint, noise_scale):
        # feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        feature_q = add_noise(feature, n_scale=noise_scale)
        feature_q, _ = data_quantization_sym(feature_q, half_level=half_level, scale = scale, isint = isint)

        return feature_q

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None, None


class Feature_Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, half_level, isint):
        # feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        feature_q, _ = data_quantization_sym(feature, half_level,  isint=isint)
        return feature_q

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
# A quantization layer
class Layer_Quant(nn.Module):
    def __init__(self, bit_level, isint, clamp_std):
        super().__init__()
        self.isint = isint
        self.output_half_level = 2 ** bit_level / 2 - 1
        self.clamp_std = clamp_std

    def forward(self, x):
        x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        return x


class Layer_Quant_noise(nn.Module):
    def __init__(self, bit_level, isint, clamp_std, noise_scale):
        super().__init__()
        self.isint = isint
        self.output_half_level = 2 ** bit_level / 2 - 1
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

    def forward(self, x):
        x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        return x


# A convolution layer which adds noise and quantize the weight and output feature map
class Conv2d_quant_noise(nn.Conv2d):
    def __init__(self,
                 qn_on,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 noise_scale,
                 bias,
                 ):
        # weight_bit -> bit level for weight
        # output_bit -> bit level for output feature map
        # isint, clamp_std, noise_scale -> same arguments as Weight_Quant_Noise()
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias
                         )
        self.qn_on = qn_on
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

    def forward(self, x):
        # quantize weight and add noise first
        if self.qn_on:
            weight_q, bias_q = Weight_Quant_Noise.apply(self.weight, self.bias,
                                                        self.weight_half_level, self.isint, self.clamp_std,
                                                        self.noise_scale)
            # calculate the convolution next
            x = self._conv_forward(x, weight_q, bias_q)

            # quantize the output feature map at last
            x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        else:
            x = self._conv_forward(x, self.weight, self.bias)

        return x


# BN融合
class BNFold_Conv2d_Q(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight_bit,
            output_bit,
            isint,
            clamp_std,
            noise_scale,
            bias,
            eps=1e-5,
            momentum=0.01, ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=True,
                         )
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def forward(self, input):
        # 训练态
        output = self._conv_forward(input, self.weight, self.bias)
        # 先做普通卷积得到A，以取得BN参数

        # 更新BN统计参数（batch和running）
        dims = [dim for dim in range(4) if dim != 1]
        batch_mean = torch.mean(output, dim=dims)
        batch_var = torch.var(output, dim=dims)
        with torch.no_grad():
            if self.first_bn == 0:
                self.first_bn.add_(1)
                self.running_mean.add_(batch_mean)
                self.running_var.add_(batch_var)
            else:
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
        # BN融合
        if self.bias is not None:
            bias = reshape_to_bias(
                self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
        else:
            bias = reshape_to_bias(
                self.beta - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps)))  # b融batch
        weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running

        # 量化A和bn融合后的W
        if qn_on:
            weight_q, bias_q = Weight_Quant_Noise.apply(weight, bias,
                                                        self.weight_half_level, self.isint, self.clamp_std,
                                                        self.noise_scale)
        else:
            weight_q = weight
            bias_q = bias
        # 量化卷积
        output = self._conv_forward(input, weight_q, bias_q)
        # output = F.conv2d(
        #     input=input,
        #     weight=weight_q,
        #     bias=self.bias,  # 注意，这里不加bias（self.bias为None）
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.groups
        # )
        # # # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
        # output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
        # output += reshape_to_activation(bias_q)
        # 量化输出
        if qnon:
            output = Feature_Quant.apply(output, self.output_half_level, self.isint)

        return output


# A fully connected layer which adds noise and quantize the weight and output feature map
# See notes in Conv2d_quant_noise
class Linear_quant_noise(nn.Linear):
    def __init__(self, qn_on, in_features, out_features,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 noise_scale,
                 bias=False, ):
        super().__init__(in_features, out_features, bias)
        self.qn_on = qn_on
        self.weight_bit = weight_bit
        self.output_bit = output_bit
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1

    def forward(self, x):
        if self.qn_on:
            weight_q, bias_q = Weight_Quant_Noise.apply(self.weight, self.bias,
                                                        self.weight_half_level, self.isint, self.clamp_std,
                                                        self.noise_scale)
            x = F.linear(x, weight_q, bias_q)
            x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        else:
            x = F.linear(x, self.weight, self.bias)

        return x


# ================================== #
# Other Functions, rarely used
# ================================== #
def plt_weight_dist(weight, name, bins):
    num_ele = weight.numel()
    weight_np = weight.cpu().numpy().reshape(num_ele, -1).squeeze()
    plt.figure()
    plt.hist(weight_np, density=True, bins=bins)
    plot_name = f"saved_best_examples/weight_dist_{name}.png"
    plt.savefig(plot_name)
    plt.close()


# Similar to Conv2d_quant_noise, only add noise without quantization
class Conv2d_noise(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=False,
                 ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias=False,
                         )

    def forward(self, x):
        weight_n = add_noise(self.weight)
        x = self._conv_forward(x, weight_n, self.bias)
        return x
