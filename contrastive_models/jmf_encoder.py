import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from d2l.torch import MultiHeadAttention
from models.utils import get_params

class CTR_JmfEncoder(nn.Module):
    """
    Encoder to embed jmf sequences using transformer
    """
    def __init__(self,input_size,hidden_size,num_layers):
        super(CTR_JmfEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.para = get_params()
        self.fc = nn.Linear(hidden_size,self.para["num_node"])
        
        
    def forward(self, input_jmf,effective_lengths):
        
        packed_out,(h_n,c_n) = self.lstm(input_jmf)
        
        if effective_lengths != None:
            # unpack sequences
            unpacked_seq, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)



            # stack effective embeddings to (batch_size,effective_emb)
            effective_embeddings = []

            for episode in range(len(effective_lengths)):
                effective_embeddings.append(unpacked_seq[episode][effective_lengths[episode] - 1].unsqueeze(0))

            effective_embeddings = torch.stack(effective_embeddings, dim=1).squeeze(0)

            

            out = F.leaky_relu(self.fc(effective_embeddings))
            return out
        
        else: # evaluation

            out = F.leaky_relu(self.fc(packed_out))
            out = out[:, -1, :] # resturn last timestep feature
            return out
        


# model = CTR_JmfEncoder(1024, 512,5)
# x = torch.randn(20, 10, 1024)

# # 前向传播
# output = model(x)
# print(output.shape)