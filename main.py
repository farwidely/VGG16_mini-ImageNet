import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
# 设置计算硬件为cpu或cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置数据预处理
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# 创建数据集
train_dataset = datasets.ImageFolder("./data/Mini-ImageNet/train", transform=transform)
test_dataset = datasets.ImageFolder("./data/Mini-ImageNet/test", transform=transform)

# 查看数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")
print("学号：221115194    姓名：邓广远")

# 设置DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 初始化模型，并载入预训练权重
model = torchvision.models.vgg16(weights='IMAGENET1K_V1')

# 冻结除全连接层外的模型参数
num = 0
for param in model.parameters():
    param.requires_grad = False

# 修改模型最后一层
model.classifier[6] = nn.Linear(4096, 100, bias=True)
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
momentum = 5e-1
optimizer = torch.optim.SGD(model.classifier[6].parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=9e-1)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 20

# 添加tensorboard
writer = SummaryWriter("./logs_train")

start = time.time()

for i in range(epoch):
    print(f"------第 {i + 1} 轮训练开始------")

    start1 = time.time()

    # 训练步骤开始
    model.train()
    for data in tqdm(train_dataloader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}，Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end1 = time.time()
    print(f"本轮训练时长为{end1 - start1}秒")
    start2 = time.time()

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    end2 = time.time()
    print(f"本轮测试时长为{end2 - start2}秒\n")

    total_test_step += 1

    if i == 19:
        torch.save(model, f"./trained_models/MyVGG16_gpu_20.pth")
        print("模型已保存")

end = time.time()
print(f"训练+测试总时长为{end - start}秒\n")
print("学号：221115194    姓名：邓广远")
