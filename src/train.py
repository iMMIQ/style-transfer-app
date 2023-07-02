import random
from glob import glob
import multiprocessing
import pathlib
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm
from pprint import pprint

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

import horovod.torch as hvd

from models import *
from utils import *

display = pprint  # 定义pprint别名为display

hvd.init()  # 初始化Horovod
torch.cuda.set_device(hvd.local_rank())  # 设置当前设备为Horovod本地排名对应的GPU设备

device = torch.device(
    "cuda:%s" %
    hvd.local_rank() if torch.cuda.is_available() else "cpu")  # 根据是否有可用的GPU设备选择设备类型

torch.multiprocessing.set_sharing_strategy(
    'file_system')  # 设置PyTorch多进程共享策略为文件系统


is_hvd = False  # 是否使用Horovod进行分布式训练
tag = 'nohvd'  # 标签
base = 32  # TransformNet的基础通道数
style_weight = 50  # 风格损失的权重
content_weight = 1  # 内容损失的权重
tv_weight = 1e-6  # 总变差损失的权重
epochs = 22  # 训练的轮数

batch_size = 32  # 批处理大小
width = 256  # 图像的宽度

verbose_hist_batch = 100  # 打印训练过程中的历史损失的批次间隔
verbose_image_batch = 800  # 打印生成图像的批次间隔

# 模型名称
model_name = 'metanet_base{}_style{}_tv{}_tag{}'.format(
    base, style_weight, tv_weight, tag)
rank = hvd.rank() if hasattr(hvd, 'rank') else None  # 获取当前进程的排名

# 打印模型名称和当前进程的排名
print('model_name: {}, rank: {}'.format(model_name, rank))


def remove_directory(path):
    # 删除目录及其内容
    try:
        path = pathlib.Path(path)
        path.unlink(missing_ok=True)  # 删除文件或空目录
        if path.is_dir():
            for child in path.iterdir():
                remove_directory(child)  # 递归删除子目录或文件
        path.rmdir()  # 删除目录
    except Exception as e:
        # 处理异常情况
        print(f"Failed to remove {path}: {e}")


# 删除隐藏文件
hidden_files = pathlib.Path('runs').glob('*/.AppleDouble')
for file in hidden_files:
    remove_directory(file)

# 删除模型保存目录
model_path = pathlib.Path('runs') / model_name
remove_directory(model_path)

# 创建预训练的VGG19模型
vgg19 = models.vgg19(pretrained=True)
vgg_features = vgg19.features[:36]  # 选择要提取特征的VGG层次
vgg = VGG(vgg_features).to(device).eval()  # 构建截断的VGG网络，用于提取特征

# 创建TransformNet模型
transform_net = TransformNet(base).to(device)
transform_net_param_dict = transform_net.get_param_dict()  # 获取TransformNet模型的参数字典

# 创建MetaNet模型
metanet = MetaNet(transform_net_param_dict).to(device)

# 数据预处理的转换操作
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(
        256 / 480, 1), ratio=(1, 1)),  # 随机裁剪图像
    transforms.ToTensor(),  # 转换为张量
    tensor_normalizer  # 标准化
])

style_dataset = torchvision.datasets.ImageFolder(
    '/root/autodl-tmp/wikiart/wikiart/images',  # 风格图像数据集路径
    transform=data_transform)  # 数据集的变换操作
content_dataset = torchvision.datasets.ImageFolder(
    '/root/autodl-tmp/coco', transform=data_transform)  # 内容图像数据集路径

if is_hvd:
    # 创建分布式训练的采样器
    train_sampler = distributed.DistributedSampler(
        content_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    # 创建数据加载器，用于加载内容数据集
    content_data_loader = torch.utils.data.DataLoader(
        content_dataset,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count(),
        sampler=train_sampler)
else:
    # 创建数据加载器，用于加载内容数据集
    content_data_loader = torch.utils.data.DataLoader(
        content_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count())

if not is_hvd or hvd.rank() == 0:
    # 打印风格数据集信息和内容数据集信息
    print(style_dataset)
    print('-' * 20)
    print(content_dataset)

# 设置模型为评估模式
metanet.eval()
transform_net.eval()

# 创建随机张量
rands = torch.rand(4, 3, 256, 256).to(device)

# 提取随机张量的特征
features = vgg(rands)

# 使用MetaNet计算特征的权重
weights = metanet(mean_std(features))

# 将权重设置到TransformNet模型中
transform_net.set_weights(weights)

# 使用TransformNet对随机张量进行变换
with torch.no_grad():
    transformed_images = transform_net(torch.rand(4, 3, 256, 256).to(device))

# 打印特征的形状
if not is_hvd or hvd.rank() == 0:
    print('Features:')
    display([x.shape for x in features])

    # 打印权重的形状
    print('Weights:')
    display([x.shape for x in weights.values()])

    # 打印变换后图像的形状
    print('Transformed Images:')
    display(transformed_images.shape)

visualization_style_image = random.choice(
    style_dataset)[0].unsqueeze(0).to(device)  # 随机选择一个风格图像进行可视化
visualization_content_images = torch.stack([random.choice(
    content_dataset)[0] for i in range(4)]).to(device)  # 随机选择4个内容图像进行可视化

if not is_hvd or hvd.rank() == 0:
    for f in glob('runs/*/.AppleDouble'):
        remove_directory(f)  # 删除隐藏文件

    remove_directory('runs/' + model_name)  # 删除模型保存目录

visualization_style_image = random.choice(
    style_dataset)[0].unsqueeze(0).to(device)  # 随机选择一个风格图像，并将其转换为张量，并移到指定设备上
visualization_content_images = torch.stack(
    [random.choice(content_dataset)[0] for i in range(4)]).to(device)  # 随机选择4个内容图像，并将它们转换为张量，并移到指定设备上

del rands, features, weights, transformed_images  # 删除不再需要的变量，释放内存

trainable_params = {}  # 可训练参数的字典
trainable_param_shapes = {}  # 可训练参数的形状字典
for model in [vgg, transform_net, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param  # 将可训练参数添加到字典中
            trainable_param_shapes[name] = param.shape  # 记录可训练参数的形状

# 创建Adam优化器，并传入可训练参数的值列表和学习率
optimizer = optim.Adam(trainable_params.values(), lr=1e-3)

if is_hvd:
    # 使用Horovod分布式优化器
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=trainable_params.items())

    # 合并TransformNet和MetaNet的状态字典
    params = transform_net.state_dict()
    params.update(metanet.state_dict())

    # 使用Horovod广播模型参数
    hvd.broadcast_parameters(params, root_rank=0)

# 内容数据加载器的批次数
n_batch = len(content_data_loader)

# 设置MetaNet和TransformNet为训练模式
metanet.train()
transform_net.train()

for epoch in range(epochs):
    smoother = defaultdict(Smooth)  # 平滑器，用于计算平均损失
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:  # 使用tqdm显示进度条
        for batch, (content_images, _) in pbar:
            n_iter = epoch * n_batch + batch  # 当前迭代次数

            if batch % 20 == 0:
                style_image = random.choice(style_dataset)[0].unsqueeze(
                    0).to(device)  # 每20个批次随机选择一个风格图像，并将其转换为张量，并移到指定设备上
                style_features = vgg(style_image)  # 提取风格图像的特征
                style_mean_std = mean_std(style_features)  # 计算风格特征的均值和标准差

            x = content_images.cpu().numpy()  # 将内容图像转移到CPU并转换为NumPy数组
            if (x.min(-1).min(-1) == x.max(-1).max(-1)
                    ).any():  # 如果存在像素值全相等的图像，则跳过该批次
                continue

            optimizer.zero_grad()  # 梯度清零

            weights = metanet(mean_std(style_features))  # 使用meta网络计算权重
            transform_net.set_weights(weights, 0)  # 将权重设置到转换网络中的第一个模块

            content_images = content_images.to(device)  # 将内容图像移动到指定设备上
            transformed_images = transform_net(
                content_images)  # 使用转换网络对内容图像进行转换

            content_features = vgg(content_images)  # 提取内容图像的特征
            transformed_features = vgg(transformed_images)  # 提取转换后图像的特征
            transformed_mean_std = mean_std(
                transformed_features)  # 计算转换后特征的均值和标准差

            content_loss = content_weight * \
                F.mse_loss(
                    transformed_features[2],
                    content_features[2])  # 计算内容损失

            style_loss = style_weight * \
                F.mse_loss(
                    transformed_mean_std,
                    style_mean_std.expand_as(transformed_mean_std))  # 计算风格损失

            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                   torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))  # 计算总变差损失

            loss = content_loss + style_loss + tv_loss  # 总损失

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            smoother['content_loss'] += content_loss.item()  # 累加内容损失
            smoother['style_loss'] += style_loss.item()  # 累加风格损失
            smoother['tv_loss'] += tv_loss.item()  # 累加总变差损失
            smoother['loss'] += loss.item()  # 累加总损失

            max_value = max([x.max().item()
                            for x in weights.values()])  # 计算权重的最大值

            s = 'Epoch: {} '.format(epoch + 1)  # 当前的epoch
            # 平滑后的内容损失
            s += 'Content: {:.2f} '.format(smoother['content_loss'])
            s += 'Style: {:.1f} '.format(smoother['style_loss'])  # 平滑后的风格损失
            s += 'Loss: {:.2f} '.format(smoother['loss'])  # 平滑后的总损失
            s += 'Max: {:.2f}'.format(max_value)  # 权重的最大值

            if (batch + 1) % verbose_image_batch == 0:
                transform_net.eval()  # 设置转换网络为评估模式
                visualization_transformed_images = transform_net(
                    visualization_content_images)  # 使用转换网络对可视化内容图像进行转换
                transform_net.train()  # 设置转换网络为训练模式
                visualization_transformed_images = torch.cat(
                    [style_image, visualization_transformed_images])  # 将风格图像与可视化转换图像拼接起来
                del visualization_transformed_images  # 删除变量，释放内存

            pbar.set_description(s)  # 更新进度条描述

            del transformed_images, weights  # 删除变量，释放内存

    if not is_hvd or hvd.rank() == 0:
        torch.save(metanet.state_dict(),
                   '/root/autodl-tmp/improve/checkpoints/{}_{}.pth'.format(model_name,
                                                                           epoch + 1))  # 保存meta网络的模型参数
        torch.save(
            transform_net.state_dict(),
            '/root/autodl-tmp/improve/checkpoints/{}_transform_net_{}.pth'.format(
                model_name,
                epoch + 1))  # 保存转换网络的模型参数

        torch.save(
            metanet.state_dict(),
            '/root/autodl-tmp/improve/models/{}.pth'.format(model_name))  # 保存meta网络的模型参数
        torch.save(
            transform_net.state_dict(),
            '/root/autodl-tmp/improve/models/{}_transform_net.pth'.format(model_name))  # 保存转换网络的模型参数
