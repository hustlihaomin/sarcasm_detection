from abc import ABC

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DNNModule']


class DNNModule(nn.Module, ABC):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(DNNModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions

        # TODO 64 * 1 * 1 需要改成相对应的输出
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        # TODO 添加position embedding，防止全部为0

    def forward(self, x):
        x = x[:, -1, :]
        # TODO 64 * 1 * 1 需要改成相对应的输出
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = self.softmax(x)
        return x
