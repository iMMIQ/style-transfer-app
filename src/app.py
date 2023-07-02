from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from models import *
from utils import *

import subprocess
from flask import Flask, request, send_file, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 添加CORS支持

# 定义MetaNet类，继承自nn.Module


class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128 * self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
        return list(filters.values())

    def forward2(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
        return filters


width = 256

# 定义数据转换的操作
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256 / 480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])


@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    # 获取上传的文件
    style_image = request.files['style_image']
    content_image = request.files['content_image']

    # 保存上传的文件
    style_path = 'style.jpg'
    content_path = 'content.jpg'
    style_image.save(style_path)
    content_image.save(content_path)

    # 读取style_image，并使用vgg_model提取其特征
    style_image = read_image(
        style_path,
        target_width=256).to(device)
    style_features = vgg_model(style_image)
    style_mean_std = mean_std(style_features)

    num_batches = 50

    # 使用tqdm显示训练进度
    with tqdm(enumerate(content_data_loader), total=num_batches) as pbar:
        for batch, (content_images, _) in pbar:
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue

            optimizer.zero_grad()

            # 使用metanet获取权重
            weights = metanet.forward2(mean_std(style_features))
            transform_net.set_weights(weights, 0)

            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)

            content_features = vgg_model(content_images)
            transformed_features = vgg_model(transformed_images)
            transformed_mean_std = mean_std(transformed_features)

            # 计算内容损失
            content_loss = content_weight * \
                F.mse_loss(transformed_features[2], content_features[2])

            # 计算风格损失
            style_loss = style_weight * \
                F.mse_loss(transformed_mean_std, style_mean_std.expand_as(transformed_mean_std))

            y = transformed_images
            # 计算总变差损失
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
                torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            loss = content_loss + style_loss + tv_loss

            loss.backward()
            optimizer.step()

            if batch > num_batches:
                break

    content_image = read_image(content_path)
    content_image = tuple(content_image)  # Convert the tensor to a tuple
    content_image = torch.stack(content_image).to(device)

    transformed_image = transform_net(content_image)

    torchvision.utils.save_image(transformed_image, 'pics/result.jpg')

    # 返回输出图片给前端
    result_path = 'pics/result.jpg'
    return send_file(result_path, mimetype='image/jpeg')


if __name__ == '__main__':
    # 加载预训练的vgg19模型，并将其部分特征提取层赋值给vgg_model
    vgg19_model = models.vgg19(pretrained=True)
    vgg_model = VGG(vgg19_model.features[:36]).to(device).eval()

    base = 32
    # 创建一个TransformNet实例，base参数为32
    transform_net = TransformNet(base).to(device)
    # 获取transform_net的参数字典
    transform_net.get_param_dict()

    # 创建MetaNet实例，使用transform_net的参数字典作为输入
    metanet = MetaNet(transform_net.get_param_dict()).to(device)

    # 创建content_dataset数据集
    content_dataset = torchvision.datasets.ImageFolder(
        '/root/autodl-tmp/coco', transform=data_transform)

    style_weight = 50
    content_weight = 1
    tv_weight = 1e-6
    batch_size = 4

    trainable_params = {}
    trainable_param_shapes = {}

    # 遍历vgg_model、transform_net和metanet的参数，将需要训练的参数添加到trainable_params字典中
    for model in [vgg_model, transform_net, metanet]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
                trainable_param_shapes[name] = param.shape

    # 使用Adam优化器，学习率为1e-3，对trainable_params中的参数进行优化
    optimizer = optim.Adam(trainable_params.values(), 1e-3)

    # 创建content_data_loader，用于加载content_dataset的数据
    content_data_loader = torch.utils.data.DataLoader(
        content_dataset, batch_size=batch_size, shuffle=True)

    # 加载预训练的metanet和transform_net模型的权重
    metanet.load_state_dict(torch.load(
        'models-std/metanet_base32_style50_tv1e-06_tagnohvd.pth'))
    transform_net.load_state_dict(torch.load(
        'models-std/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth'))

    app.run()
