import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from d2l.torch import MultiHeadAttention

class CTR_FeatureExtr(nn.Module):
    """
    Encoder to embed jmf sequences using transformer
    """
    def __init__(self):
        super(CTR_FeatureExtr, self).__init__()
        self.con1 = nn.Conv2d(3,6,3)
        self.con2 = nn.Conv2d(6,12,5,2)
        self.con3 = nn.Conv2d(12,24,5,2)
        self.con4 = nn.Conv2d(24,24,5,2)
        self.pool = nn.MaxPool2d(3,2)
        self.fc1 = nn.Linear(3456,1024)
        self.fc2 = nn.Linear(1024,512)
        self.act = nn.LeakyReLU()

        
        
    def forward(self, input_jmf):
        
        
        out = self.act(self.con1(input_jmf))
        out = self.act(self.con2(out))
        out = self.act(self.con3(out))
        out = self.act(self.con4(out))
        out = self.pool(out)
        out = out.view(out.shape[0],-1)
        out = self.act(self.fc1(out))
        out = self.act(self.fc2(out))
        

        return out
        
       
        


# model = CTR_JmfEncoder(1024, 512,5)
# x = torch.randn(20, 10, 1024)

# # 前向传播
# output = model(x)
# print(output.shape)