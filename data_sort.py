# 对mini-imagenet-raw目录的数据进行分类

import os
import csv
from PIL import Image

# 初始化csv标签文件路径
train_csv_path = "mini-imagenet-raw/train.csv"
val_csv_path = "mini-imagenet-raw/val.csv"
test_csv_path = "mini-imagenet-raw/test.csv"

# 初始化标签存储字典
train_label = {}
val_label = {}
test_label = {}

with open(train_csv_path) as csvfile:
    # csv.reader()返回一个列表，文件中的一行对应列表中的一个元素
    csv_reader = csv.reader(csvfile)
    # 将标头迭代掉
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[0]] = row[1]

with open(val_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[0]] = row[1]

with open(test_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        test_label[row[0]] = row[1]

img_path = "mini-imagenet-raw/images"
new_img_path = "mini-imagenet-raw/partitioned_images"
for jpg in os.listdir(img_path):
    path = img_path + '/' + jpg
    img = Image.open(path)
    if jpg in train_label.keys():
        # 读取标签名用于创建类别文件夹名
        label_name = train_label[jpg]
        # 类别文件夹路径
        temp_path = new_img_path + '/train' + '/' + label_name
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        # 划分后的数据集图片保存路径
        temp_img_path = temp_path + '/' + jpg
        img.save(temp_img_path)

    elif jpg in val_label.keys():
        label_name = val_label[jpg]
        temp_path = new_img_path + '/val' + '/' + label_name
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        temp_img_path = temp_path + '/' + jpg
        img.save(temp_img_path)

    elif jpg in test_label.keys():
        label_name = test_label[jpg]
        temp_path = new_img_path + '/test' + '/' + label_name
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        temp_img_path = temp_path + '/' + jpg
        img.save(temp_img_path)
