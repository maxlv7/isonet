import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO 网络结构优化
# TODO 改变学习率 OK
# TODO cuda支持 OK
# TODO 测试集支持 OK
# TODO 断点恢复训练
# TODO tensorboard支持

class ISONet(nn.Module):
    def __init__(self):
        super().__init__()
        # in 3 out 32
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv6 = nn.Conv2d(64, 1, 3)

        self.fc = nn.Linear(121, 11)
        self.fc2 = nn.Linear(11, 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x: torch.Tensor):
        # 64*64 -> 60*60
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 60*60->30*30
        x = F.max_pool2d(x, (2, 2))
        # x = nn.BatchNorm2d(x,)

        # 30*30->24*24
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # 24*24->22*22
        x = F.relu(self.conv6(x))

        # 22*22->11*11

        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc(x)

        x = self.fc2(x)

        return x


class ISONet_Lenet(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3136, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
