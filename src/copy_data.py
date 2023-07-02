import os
import random
import shutil

def copy_images(dataset_dir, target_dir, percentage):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if random.random() <= percentage:
                # 构建源文件路径和目标文件路径
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, dataset_dir)
                target_path = os.path.join(target_dir, relative_path)

                # 创建目标文件所在的文件夹
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # 复制文件
                shutil.copy2(source_path, target_path)

# 指定数据集目录和目标目录
dataset_dir_1 = '/root/autodl-tmp/wikiart/wikiart/images'
target_dir_1 = '/root/autodl-tmp/wikiart-small/wikiart/images'

dataset_dir_2 = '/root/autodl-tmp/coco'
target_dir_2 = '/root/autodl-tmp/coco-small'

# 指定要复制的图像百分比（1%）
percentage = 0.01

# 复制图像到目标目录
copy_images(dataset_dir_1, target_dir_1, percentage)
copy_images(dataset_dir_2, target_dir_2, percentage)
