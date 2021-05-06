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

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attentino_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_dim * 2,
                                  1)  # hidden:[batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights:[batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] =
        # [batch_size, n_hidden *num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, x):
        # hidden_last, cn_last : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        x, (hidden_last, cn_last) = self.bilstm(x)
        x = x.permute(1, 0, 2)
        attn_output, attention = self.attentino_net(x, hidden_last)
        out = self.fc(attn_output)  # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return out
