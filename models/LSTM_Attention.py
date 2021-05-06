import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttentionModule(nn.Module):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(LSTMAttentionModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions
        self.hidden_dim = int(input_dimensions / 2)
        self.bilstm = nn.LSTM(input_dimensions, self.hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_classes)

        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_dim*2,self.hidden_dim*2))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_dim*2,1))

        nn.init.uniform_(self.w_omega,-0.1,0.1)
        nn.init.uniform_(self.u_omega,-0.1,0.1)

    # lstm_output : [batch_size, seq_len, n_hidden * num_directions(=2)], F matrix
    def attentino_net(self, lstm_output):
        u = torch.tanh(torch.matmul(lstm_output,self.w_omega))#(batch,seq_len,hidden_dim*2)
        att = torch.matmul(u,self.u_omega) #[batch,seq_len,1]
        att_score = F.softmax(att,dim=1)

        scored_x = lstm_output *att_score #[batch,seq_len,hidden_dim*2]

        context = torch.sum(scored_x,dim=1)#[batch,hidden_dim*2]
        return context

    def forward(self, x):
        x, (hidden_last, cn_last) = self.bilstm(x) # hidden_last, cn_last : [num_layers(=1) * num_directions(=2), batch_size, hidden_dim]
        # x = x.permute(1, 0, 2)
        attn_output = self.attentino_net(x)
        out = self.fc(attn_output)  # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return out
