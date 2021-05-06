from abc import ABC

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DNNModule']


class DNNModule(nn.Module, ABC):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(DNNModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions
        self.dropout = nn.Dropout(0.1)
        # TODO 64 * 1 * 1 需要改成相对应的输出
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,16)
        self.fc7 = nn.Linear(16,8)
        self.fc8 = nn.Linear(8,2)
        self.fc = nn.Linear(256,2)
        # TODO 添加position embedding，防止全部为0

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(16)
        self.bn7 = nn.BatchNorm1d(8)
    def forward(self, x):
        x = x[:, -1, :]
        # TODO 64 * 1 * 1 需要改成相对应的输出
        # x = x.view(x.size(0), -1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        # x = self.bn3(F.relu(self.fc3(x)))
        # x = self.bn4(F.relu(self.fc4(x)))
        # x = self.bn5(F.relu(self.fc5(x)))
        # x = self.bn6(F.relu(self.fc6(x)))
        # x = self.bn7(F.relu(self.fc7(x)))
        # x = self.dropout(F.relu(self.fc8(x)))
        # x = self.softmax(x)
        x = self.dropout(self.fc(x))
        return x
