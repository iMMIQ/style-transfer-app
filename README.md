# 风格转换应用

该项目是一个风格转换应用，使用Vue.js和axios构建前端，后端使用Python的Flask框架。该应用允许用户上传两张图片，一张作为风格图片，另一张作为内容图片，并生成一张风格转换后的图片。

## 依赖安装

在运行该应用程序之前，请确保已安装以下软件和库：

- Python 3.x
- PyTorch （仅在1.11版本测试）
- torchvision
- Flask
- flask_cors

你可以使用pip命令安装它们：

```
pip install torch torchvision Flask flask_cors tqdm
```

安装前端依赖：

```bash
cd src/static
npm install
```

## 运行应用程序

要运行该应用程序，请按照以下步骤操作：

1. 在应用程序所在的目录中打开终端或命令提示符，确保已下载好相关u数据集及模型。
2. 运行以下命令启动应用程序：

```
python app.py
```

3. 应用程序将在本地主机上的默认端口（通常是http://127.0.0.1:5000/）上启动。

4. 安装前端依赖：

```bash
cd static
npm install
```

5. 启动前端应用：

```bash
npm run serve
```

## 使用方法

1. 打开浏览器并访问前端应用地址：`http://localhost:8080/`。
2. 页面会显示一个标题和两个图片上传框，分别对应风格图片和内容图片。
3. 点击第一个图片上传框，选择一个图片作为风格图片，支持JPEG和PNG格式。
4. 点击第二个图片上传框，选择一个图片作为内容图片，同样支持JPEG和PNG格式。
5. 在选择完两张图片后，点击“生成”按钮。
6. 等待一段时间，风格转换后的图片会在页面中显示出来。

## 注意事项

- 确保您的电脑上已经安装了Node.js和Python。
- 该应用仅支持JPEG和PNG格式的图片。
- 图片上传可能需要一定时间，取决于图片大小和网络速度。
- 风格转换后的图片会覆盖之前生成的结果，刷新页面将清除之前的结果。

## 技术实现

- 前端使用Vue.js框架构建，通过axios进行与后端的HTTP通信。
- 后端使用Python的Flask框架，接收前端上传的图片，进行风格转换并返回结果。
- 风格转换算法可能需要一定时间，请耐心等待。

## 下载链接

- [coco数据集](http://images.cocodataset.org/zips/train2017.zip)
- [wikiart数据集](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan)
- [基于vgg19训练的模型](https://pan.immiq.top/s/ZgSi9kY4yYiLWzb)

## 训练模型

1. 环境设置

要运行该代码，你需要满足以下环境要求：

- Python 3.6+
- PyTorch （仅在1.11下测试）
- torchvision
- numpy
- OpenCV
- tqdm
- matplotlib
- horovod

可以使用以下命令安装所需的依赖项：

```shell
pip install torch torchvision numpy opencv-python tqdm matplotlib horovod
```

2. 将内容图像和风格图像分别放置文件夹中，参考注释修改train.py中的数据集路径。

3. 运行以下命令开始训练模型：

```shell
python train.py
```

训练过程将持续多个周期（epochs），可以在代码中设置训练参数来控制训练的细节。

4. 训练完成后，模型将被保存在以下路径：

- MetaNet模型：'/root/autodl-tmp/improve/models/model_name.pth'
- TransformNet模型：'/root/autodl-tmp/improve/models/model_name_transform_net.pth'

请将上述路径中的'model_name'替换为你自己的模型名称。