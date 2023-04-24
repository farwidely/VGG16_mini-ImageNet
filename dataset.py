# 将dataset_constructing.py中划分好的数据集导入pytorch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 设置数据预处理
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 创建数据集
train_dataset = datasets.ImageFolder("./data/Mini-ImageNet/train", transform=transform)
test_dataset = datasets.ImageFolder("./data/Mini-ImageNet/test", transform=transform)

# 查看数据集长度和标签
print(len(train_dataset))
print(len(test_dataset))
print(train_dataset.class_to_idx)
print(test_dataset.class_to_idx)

# # 设置DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# for data in train_dataloader:
#     imgs, targets = data
#     print(imgs)
#     print(targets)
