# 将mini-imagenet-sorted中的数据随机划分为训练集和测试集

import os
import random
# 安装shutil指令：pip install shutilwhich
import shutil


# 创建目录
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder has been created  ---")
    else:
        print("---  There is this folder!  ---")


# 确保文件目录存在
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# 将source_dir目录的文件复制到target_dir目录
def cover_files(source_dir, target_dir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_dir)


# 将source_dir目录下的文件按一定比例剪切到target_dir目录
def pick_file(source_dir, target_dir, rate):
    ensure_dir_exists(target_dir)
    files_dir = os.listdir(source_dir)
    file_number = len(files_dir)
    pick_rate = rate
    pick_number = int(file_number * pick_rate)
    sample = random.sample(files_dir, pick_number)
    for name in sample:
        shutil.move(source_dir+'/'+name, target_dir+'/'+name)


if __name__ == '__main__':
    data_path = "./mini-imagenet-sorted"
    sort_dir = os.listdir(data_path)
    train_dataset_path = "./data/Mini-ImageNet/train"
    test_dataset_path = "./data/Mini-ImageNet/test"
    for sort in sort_dir:
        source_path = "./mini-imagenet-sorted" + '/' + sort
        target_path = "./data/Mini-ImageNet/test" + '/' + sort
        # 将当前类别目录中的图片文件随机抽取16.67%移动至测试集目录
        pick_file(source_path, target_path, rate=0.1667)
        # 将当前类别目录及目录下剩余的图片移动至训练集目录下
        shutil.move(source_path, train_dataset_path)
