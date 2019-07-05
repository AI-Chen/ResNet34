import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from ResNet34 import  ResNet
from ResNet34 import  BasicBlock
from utils.utils import MyDataset, validate, show_confMat

train_txt_path = 'Data/train.txt'
valid_txt_path = 'Data/valid.txt'
classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
lr_init = 0.0001
max_epoch = 20
# log
result_dir = 'Result/'
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# -------------------------------------------- step 1/5 : 加载数据 -------------------------------------------#

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
    ])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
    ])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# -------------------------------------------- step 2/5 : 构建网络、加载权值 -------------------------------------------#
# ================================ #
#           构建一个网络
net = ResNet(BasicBlock, [3, 4, 6, 3])
# ================================ #

# ================================ #
#           初始化权值
pretrained_dict = torch.load('ResNet34_pretrained/resnet34-333f7ec4.pth')
pretrained_dict_keys = [name for name in pretrained_dict.keys()]
print(pretrained_dict_keys)
net_state_dict = net.state_dict()
net_state_dict_keys = [name for name in net_state_dict.keys()]
print(net_state_dict_keys)

net_state_dict = net.state_dict()
#model是自己定义好的新网络模型，将pretrained_dict和model_dict中命名一致的层加入pretrained_dict（包括参数)。
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
net_state_dict.update(pretrained_dict_1)
net.load_state_dict(net_state_dict)
# ================================ #

# ================================ #
#   以下是初始化权值时自己的尝试。在调试代码过程中熟悉了load函数、update函数、load_dict函数
#   模型的state_dict属性以及模型参数以键值对储存的形式
# pretrained_dict = torch.load('ResNet34_pretrained/resnet34-333f7ec4.pth')
# pretrained_dict_keys = [name for name in pretrained_dict.keys()]
# print(len(pretrained_dict_keys))
# net_state_dict = net.state_dict()
# net_state_dict_keys = [name for name in net_state_dict.keys()]
# print(len(net_state_dict_keys))
# for i , name in enumerate(net_state_dict_keys):
#     if name != 'fc.weight' and  name !="fc.bias":
#         print(i)
#         net_state_dict[name].copy_(pretrained_dict[pretrained_dict_keys[i]])
# print(pretrained_dict[pretrained_dict_keys[0]],net_state_dict_keys[net_state_dict_keys[0]])
# ================================ #

# -------------------------------------------- step 3/5 : 选择损失函数、优化器；设置超参数 -------------------------------------------#
# ================================ #
#     为不同层设置不同的学习率

# 将fc_out层的参数从原始网络参数中剔除
ignored_params = list(map(id, net.fc_out.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

# 为fc_out层设置需要的学习率
optimizer = optim.SGD([
    {'params': base_params},
    {'params': net.fc_out.parameters(), 'lr': lr_init*10}],  lr_init, momentum=0.9, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数(交叉熵损失函数）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)     # 设置学习率下降策略
# ================================ #

# ================================ #
#                训练
writer = SummaryWriter(log_dir)
for epoch in range(max_epoch):
    loss_sigma = 0.0  # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))
            print('参数组1的学习率:{}, 参数组2的学习率:{}'.format(scheduler.get_lr()[0], scheduler.get_lr()[1]))

    # 每个损失的下降情况
    writer.add_scalar('loss', loss_avg, epoch, walltime=epoch)
    writer.add_scalar('acc', correct / total, epoch, walltime=epoch)

    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

            # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    loss_sigma = 0.0
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
    net.eval()
    for i, data in enumerate(valid_loader):

        # 获取图片和标签
        images, labels = data
        images, labels = Variable(images), Variable(labels)

        # forward
        outputs = net(images)
        outputs.detach_()

        # 计算loss
        loss = criterion(outputs, labels)
        loss_sigma += loss.item()

        # 统计
        _, predicted = torch.max(outputs.data, 1)
        # labels = labels.data    # Variable --> tensor

        # 统计混淆矩阵
        for j in range(len(labels)):
            cate_i = labels[j].numpy()
            pre_i = predicted[j].numpy()
            conf_mat[cate_i, pre_i] += 1.0
    print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
writer.close()
print('Finished Training')


# ------------------------------------ step5: 绘制混淆矩阵图 ------------------------------------

conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)