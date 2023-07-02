import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed as distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt

cnn_normalization_mean = [0.485, 0.456, 0.406]  # 图像归一化的均值
cnn_normalization_std = [0.229, 0.224, 0.225]  # 图像归一化的标准差
tensor_normalizer = transforms.Normalize(
    mean=cnn_normalization_mean,
    std=cnn_normalization_std)  # 创建图像张量的标准化转换器
epsilon = 1e-5  # 避免除以零的小量


def preprocess_image(
        image,
        target_width=None,
        resize=True,
        center_crop=True,
        normalize=True):
    """
    预处理图像，包括调整大小、剪裁、转换为张量并进行标准化

    Args:
        image (PIL.Image.Image): 输入的图像
        target_width (int): 目标宽度 (default: None)
        resize (bool): 是否调整大小 (default: True)
        center_crop (bool): 是否居中裁剪 (default: True)
        normalize (bool): 是否标准化 (default: True)

    Returns:
        torch.Tensor or None: 预处理后的图像张量，如果发生异常则返回 None

    Raises:
        RuntimeError: 当图像尺寸过大导致内存溢出时
    """
    transforms_list = []

    if resize and target_width:
        transforms_list.append(transforms.Resize(target_width))  # 调整图像大小

    if center_crop and target_width:
        transforms_list.append(transforms.CenterCrop(target_width))  # 居中裁剪

    transforms_list.append(transforms.ToTensor())  # 转换为张量

    if normalize:
        transforms_list.append(tensor_normalizer)  # 标准化

    transform = transforms.Compose(transforms_list)

    try:
        return transform(image).unsqueeze(0)  # 添加一维作为批处理维度
    except RuntimeError:
        # 处理图像尺寸过大导致的内存溢出异常
        print("Error: Image size too large.")
        return None


def image_to_tensor(image, target_width=None):
    """
    将图像转换为张量

    Args:
        image (numpy.ndarray): 输入的图像数组
        target_width (int): 目标宽度 (default: None)

    Returns:
        torch.Tensor or None: 预处理后的图像张量，如果发生异常则返回 None

    Raises:
        ValueError: 当输入图像为无效类型或尺寸时
    """
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序为RGB
        image = Image.fromarray(image)  # 将数组转换为图像
        return preprocess_image(image, target_width)  # 预处理图像
    except ValueError as e:
        print(f"Error: Invalid input image. {str(e)}")
        return None


def read_image(path, target_width=None):
    """
    从文件路径读取图像并进行预处理

    Args:
        path (str): 图像文件的路径
        target_width (int): 目标宽度 (default: None)

    Returns:
        torch.Tensor or None: 预处理后的图像张量，如果发生异常则返回 None

    Raises:
        FileNotFoundError: 当文件路径指定的图像文件不存在
        ValueError: 当输入图像尺寸无效或预处理过程中发生异常
    """
    try:
        image = Image.open(path)  # 打开图像文件
        return preprocess_image(image, target_width)  # 预处理图像
    except FileNotFoundError:
        print(f"Error: Image file not found at path: {path}")
        return None
    except ValueError as e:
        print(f"Error: Invalid input image. {str(e)}")
        return None


def recover_image(tensor):
    """
    将张量转换回图像

    Args:
        tensor (torch.Tensor): 输入的张量

    Returns:
        numpy.ndarray: 恢复后的图像数组
    """
    image = tensor.to(torch.uint8).detach(
    ).cpu().numpy()  # 将张量转移到CPU并转换为NumPy数组
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
        np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (
        image.transpose(
            0,
            2,
            3,
            1) *
        255.).clip(
            0,
            255).astype(
                np.uint8)[0]


def recover_tensor(tensor):
    """
    将张量进行反标准化

    Args:
        tensor (torch.Tensor): 输入的张量

    Returns:
        torch.Tensor: 反标准化后的张量，数值在0和1之间
    """
    mean = torch.tensor(cnn_normalization_mean).reshape(
        (1, 3, 1, 1)).to(
        tensor.device)
    std = torch.tensor(cnn_normalization_std).reshape(
        (1, 3, 1, 1)).to(
        tensor.device)

    tensor = tensor * std + mean  # 反标准化
    tensor = torch.clamp(tensor, 0, 1)  # 对张量进行裁剪，确保数值在0和1之间

    return tensor


def imshow(tensor, title=None):
    """
    显示图像张量

    Args:
        tensor (torch.Tensor): 输入的张量
        title (str or None): 图像的标题 (default: None)
    """
    image = recover_image(tensor)  # 将张量转换为图像
    image = image.astype(np.uint8)  # 将图像数组转换为无符号整数类型
    image = np.clip(image, 0, 255)  # 对图像进行裁剪，确保数值在0和255之间
    image = image[..., ::-1]  # 调整图像的颜色通道顺序为RGB
    plt.imshow(image)  # 显示图像
    if title is not None:
        plt.title(title)  # 设置标题
    plt.show()  # 显示图像


def mean_std(features, epsilon=1e-5):
    """
    计算特征的均值和标准差

    Args:
        features (List[torch.Tensor]): 特征列表
        epsilon (float): 避免除以零的小值 (default: 1e-5)

    Returns:
        torch.Tensor: 均值和标准差的张量
    """
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)  # 重塑形状以便计算均值和方差
        x = torch.cat([x.mean(-1),
                       torch.sqrt(x.var(-1) + epsilon)],
                      dim=-1)  # 计算均值和标准差
        n = x.shape[0]
        # 【mean, ..., std, ...] to [mean, std, ...]
        x2 = x.view(n, 2, -1).transpose(2,
                                        1).contiguous().view(n, -1)  # 重排张量的顺序
        mean_std_features.append(x2)  # 添加到列表中
    mean_std_features = torch.cat(mean_std_features, dim=-1)  # 按维度拼接张量
    return mean_std_features  # 返回均值和标准差的张量


class Smooth:
    def __init__(self, window_size=100):
        """
        初始化Smooth类

        Args:
            window_size (int): 平滑窗口的大小 (default: 100)
        """
        self.window_size = window_size  # 平滑窗口的大小
        self.data = np.zeros(
            (self.window_size, 1), dtype=np.float32)  # 存储数据的数组
        self.index = 0  # 当前索引

    def __iadd__(self, x):
        """
        实现+=运算符重载

        Args:
            x (float): 要添加到数据数组中的值

        Returns:
            Smooth: 更新后的Smooth对象
        """
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x  # 更新数据数组中的值
        self.index += 1  # 增加索引
        return self

    def __float__(self):
        """
        实现float()函数转换

        Returns:
            float: 数据的平均值
        """
        return float(self.data.mean())  # 返回数据的平均值

    def __format__(self, f):
        """
        实现格式化输出

        Args:
            f (str): 格式字符串

        Returns:
            str: 格式化后的平均值字符串
        """
        return self.__float__().__format__(f)  # 格式化输出平均值
