import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CNNLSTMModule']


class CNNLSTMModule(nn.Module):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(CNNLSTMModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions

        self.hidden_dim = int(input_dimensions/2)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.17)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,128)
        self.fc3 = nn.Linear(128,self.output_classes)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(128)
        # self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 10), stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 10), stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(128, 64, kernel_size=(3, 10), stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(2)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 10), stride=1, padding=1)
        # self.pool4 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv1d(input_dimensions,self.hidden_dim,kernel_size=2)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2)

    def forward(self,x):
        # x = x.unsqueeze(1)
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = self.pool4(F.relu(self.conv4(x)))
        #print(x.shape)
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0,2,1)
        x,(hidden_last,cn_last) = self.lstm(x)

        hidden_last_L = hidden_last[-2]
        hidden_last_R = hidden_last[-1]

        hidden_last_out = torch.cat([hidden_last_L,hidden_last_R],dim=-1)

        out = self.dropout(hidden_last_out)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(self.fc3(out))
        return out
