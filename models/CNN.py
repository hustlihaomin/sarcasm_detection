from abc import ABC

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CNNModule']


class CNNModule(nn.Module, ABC):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(CNNModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 10), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 10), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 10), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 10), stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        # TODO 64 * 1 * 1 需要改成相对应的输出
        # for GEN, is 64 * 11 * 41
        # for HYP, is 64 * 13 * 41
        # for RQ, is 64 * 14 * 41
        self.fc1 = nn.Linear(64 * 14 * 41, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)

        # TODO 添加position embedding，防止全部为0

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        # TODO 64 * 1 * 1 需要改成相对应的输出
        x = x.view(-1, 64 * 14 * 41)

        # x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(self.fc4(x))
        # x = F.relu((self.fc1(x)))
        # x = F.relu((self.fc2(x)))
        # x = F.relu((self.fc3(x)))
        x = self.fc4(x)
        # x = self.softmax(x)
        return x
