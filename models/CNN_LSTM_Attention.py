import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CNNLSTMAttentionModule']


class CNNLSTMAttentionModule(nn.Module):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(CNNLSTMAttentionModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions

        self.hidden_dim = int(input_dimensions/2)
        self.bilstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.17)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,128)
        self.fc3 = nn.Linear(128,self.output_classes)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(input_dimensions,self.hidden_dim,kernel_size=2)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2)

        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_dim * 2, self.hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    # lstm_output : [batch_size, seq_len, n_hidden * num_directions(=2)], F matrix
    def attentino_net(self, lstm_output):
        u = torch.tanh(torch.matmul(lstm_output, self.w_omega))  # (batch,seq_len,hidden_dim*2)
        att = torch.matmul(u, self.u_omega)  # [batch,seq_len,1]
        att_score = F.softmax(att, dim=1)

        scored_x = lstm_output * att_score  # [batch,seq_len,hidden_dim*2]

        context = torch.sum(scored_x, dim=1)  # [batch,hidden_dim*2]
        return context#[batch,hidden_dim*2]

    def forward(self,x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0,2,1)
        x,(hidden_last,cn_last) = self.bilstm(x)

        attn_output = self.attentino_net(x)# model : [batch_size, num_classes], attention : [batch_size, n_step]
        # hidden_last_L = hidden_last[-2]
        # hidden_last_R = hidden_last[-1]
        #
        # hidden_last_out = torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        #
        # out = self.dropout(hidden_last_out)
        # out = F.relu(self.bn1(self.fc1(attn_output)))
        # out = F.relu(self.bn2(self.fc2(out)))
        # out = self.dropout(self.fc3(out))
        out = F.relu(self.fc1(attn_output))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
