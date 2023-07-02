from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed as distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):
    def __init__(self, features):
        """
        初始化VGG模型

        Args:
            features (nn.Module): VGG模型的特征提取部分
        """
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '17': "relu3_3",
            '26': "relu4_3"
        }  # 定义VGG模型中各层的名称映射
        for p in self.parameters():
            p.requires_grad = False  # 冻结模型的参数，使其不可训练

    def forward(self, x):
        """
        VGG模型的前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            List[torch.Tensor]: 指定层的输出列表
        """
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)  # 提取指定层的输出
        return outs


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """
        初始化自定义的二维卷积层

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int or tuple): 卷积核大小，默认为3
            stride (int or tuple): 步长，默认为1
        """
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros(
            (out_channels,
             in_channels,
             kernel_size,
             kernel_size)).to(device)  # 创建卷积核权重
        self.bias = torch.zeros(out_channels).to(device)  # 创建卷积核偏置

        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = (kernel_size, kernel_size)  # 卷积核大小
        self.stride = (stride, stride)  # 步长

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)  # 执行卷积运算

    def extra_repr(self):
        """
        返回额外的模块信息字符串

        Returns:
            str: 模块信息字符串
        """
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')  # 打印额外的模块信息
        return s.format(**self.__dict__)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        初始化残差块

        Args:
            channels (int): 输入和输出通道数
        """
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )  # 定义残差块的卷积层序列

    def forward(self, x):
        """
        执行残差块的前向传播

        Args:
            x (tensor): 输入张量

        Returns:
            tensor: 输出张量
        """
        return self.conv(x) + x  # 执行残差块的前向传播，加上输入张量


def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
              upsample=None, instance_norm=True, relu=True, trainable=False):
    """
    构建卷积层的序列

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        upsample (int or None): 上采样因子，默认为None
        instance_norm (bool): 是否使用实例归一化层，默认为True
        relu (bool): 是否使用ReLU激活函数，默认为True
        trainable (bool): 是否使用可训练的卷积层，默认为False

    Returns:
        list: 卷积层序列
    """
    layers = []
    if upsample:
        layers.append(
            nn.Upsample(
                mode='nearest',
                scale_factor=upsample))  # 上采样操作
    layers.append(nn.ReflectionPad2d(kernel_size // 2))  # 反射填充
    if trainable:
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride))  # 可训练的卷积层
    else:
        layers.append(
            MyConv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride))  # 自定义的卷积层
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))  # 实例归一化层
    if relu:
        layers.append(nn.ReLU())  # ReLU激活函数
    return layers  # 返回卷积层序列


class TransformNet(nn.Module):
    def __init__(self, base=8):
        super(TransformNet, self).__init__()
        self.base = base
        self.weights = []  # 权重列表
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9, trainable=True),  # 下采样层
            *ConvLayer(base, base * 2, kernel_size=3, stride=2),
            *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(base * 4) for i in range(5)])  # 残差块
        self.upsampling = nn.Sequential(
            *
            ConvLayer(
                base *
                4,
                base *
                2,
                kernel_size=3,
                upsample=2),  # 上采样层
            *
            ConvLayer(
                base *
                2,
                base,
                kernel_size=3,
                upsample=2),
            *
            ConvLayer(
                base,
                3,
                kernel_size=9,
                instance_norm=False,
                relu=False,
                trainable=True),
        )
        self.get_param_dict()

    def forward(self, X):
        y = self.downsampling(X)  # 下采样
        y = self.residuals(y)  # 残差块
        y = self.upsampling(y)  # 上采样
        return y

    def get_param_dict(self):
        param_dict = defaultdict(int)  # 参数字典

        def dfs(module, name):
            for name2, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
            if module.__class__ == MyConv2D:
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))
        dfs(self, '')
        return param_dict

    def set_my_attr(self, name, value):
        target = self
        for x in name.split('.'):
            if x.isnumeric():
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)

        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)  # 设置权重
        target.bias = value[n_weight:].view(target.bias.shape)  # 设置偏置项

    def set_weights(self, weights, i=0):
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])  # 设置权重


class MetaNet(nn.Module):
    def __init__(self, param_dict):
        """
        MetaNet模型的构造函数

        Args:
            param_dict (dict): 参数字典，包含线性层名称和对应的输出维度
        """
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)  # 参数数量
        self.hidden = nn.Linear(1920, 128 * self.param_num)  # 隐藏层

        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

    def forward(self, mean_std_features):
        """
        MetaNet模型的前向传播

        Args:
            mean_std_features (torch.Tensor): 均值和标准差特征向量，形状为 (batch_size, 1920)

        Returns:
            dict: 线性层输出字典，包含线性层名称和对应的输出张量
        """
        hidden = F.relu(self.hidden(mean_std_features))  # 隐藏层

        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])  # 线性层

        return filters
