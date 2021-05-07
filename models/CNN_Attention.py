import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttentionModule(nn.Module):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(CNNAttentionModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions

        self.hidden_dim = int(input_dimensions / 2)
        self.dropout = nn.Dropout(0.17)
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,self.output_classes)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(input_dimensions,self.hidden_dim,kernel_size=2)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2)
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    # CNN_output : [batch_size, seq_len, hidden_dim)], F matrix
    def attentino_net(self, cnn_output):
        u = torch.tanh(torch.matmul(cnn_output, self.w_omega))  # (batch,seq_len,hidden_dim)
        att = torch.matmul(u, self.u_omega)  # [batch,seq_len,1]
        att_score = F.softmax(att, dim=1)

        scored_x = cnn_output * att_score  # [batch,seq_len,hidden_dim]

        context = torch.sum(scored_x, dim=1)  # [batch,hidden_dim]
        return context#[batch,hidden_dim]

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        attn_output = self.attentino_net(x)

        out = F.relu(self.bn1(self.fc1(attn_output)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(self.fc3(out))
        return out