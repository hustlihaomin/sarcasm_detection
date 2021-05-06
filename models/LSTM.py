import torch
import torch.nn as nn
# import torch.nn.functional as F


# __all__=['LSTMModule']
#BiLSTM
class LSTMModule(nn.Module):
    def __init__(self, input_dimensions, input_length, output_classes):
        super(LSTMModule, self).__init__()
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions
        self.hidden_dim = int(input_dimensions/2)
        self.bilstm = nn.LSTM(input_dimensions,self.hidden_dim,num_layers=2,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim*2,self.output_classes)

        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     #开始时刻，无隐状态
    #     #(Sequence*2,batch_size,hidden_dim)
    #     return (torch.zeros(1*2,24,self.hidden_dim).cuda(1),
    #             torch.zeros(1*2,24,self.hidden_dim).cuda(1))

    def forward(self,x):
        # print("hiddenshape: ",len(self.hidden))
        # print("x: ",x.shape),self.hidden
        #print(x.shape)
        x,(hidden_last,cn_last) = self.bilstm(x)

        hidden_last_L = hidden_last[-2]
        hidden_last_R = hidden_last[-1]

        hidden_last_out = torch.cat([hidden_last_L,hidden_last_R],dim=-1)

        output = self.dropout(hidden_last_out)
        out = self.fc(output)
        return out