from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TextCNNModule']


class TextCNNModule(nn.Module, ABC):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(TextCNNModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(10, self.width))
        self.conv2 = nn.Conv2d(1, 32, kernel_size=(20, self.width))
        self.conv3 = nn.Conv2d(1, 32, kernel_size=(30, self.width))
        self.conv4 = nn.Conv2d(1, 32, kernel_size=(40, self.width))

        self.pool1 = nn.MaxPool1d(4)
        self.pool2 = nn.MaxPool1d(3)
        self.pool3 = nn.MaxPool1d(2)
        self.pool4 = nn.MaxPool1d(1)

        # TODO 64 * 1 * 1 需要改成相对应的输出
        # for GEN, is 32 * 315
        # for HYP, is 32 * 386
        self.fc1 = nn.Linear(32 * 386, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
        # TODO 添加position embedding，防止全部为0

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # the shape of x is: <batch, max_length, embed_size>
        # =====> <batch, 1 max_length, embed_size>
        x = x.unsqueeze(1)
        x_1 = F.relu(self.conv1(x))
        x_2 = F.relu(self.conv2(x))
        x_3 = F.relu(self.conv3(x))
        x_4 = F.relu(self.conv4(x))

        # =====> <batch, 32, max_length, 1>
        x_1 = self.pool1(torch.squeeze(x_1, dim=3))
        x_2 = self.pool2(torch.squeeze(x_2, dim=3))
        x_3 = self.pool3(torch.squeeze(x_3, dim=3))
        x_4 = self.pool4(torch.squeeze(x_4, dim=3))

        x = torch.cat((x_1, x_2), dim=2)
        x = torch.cat((x, x_3), dim=2)
        x = torch.cat((x, x_4), dim=2)

        # TODO 64 * 1 * 1 需要改成相对应的输出
        x = x.view(-1, 32 * 386)
        # x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        # x = self.softmax(x)
        return x
