import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.utils import contrastive_dataloader,get_params
from contrastive_models.jmf_encoder import CTR_JmfEncoder
from contrastive_models.language_encoder import CTR_LanguageEncoder

from torch.utils.tensorboard import SummaryWriter

run_name = "image"

writer = SummaryWriter('runs/cl_tril_' + run_name)

# initializing parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 15
lr = 0.0001
update_step = 10000


para = get_params()


feature_dim = para["f_d"]
hidden_size = para["h_d"]
layers = para["layers"]

data_loader = contrastive_dataloader()
jmf_encoder = CTR_JmfEncoder(feature_dim,hidden_size,layers).to(device)
language_encoder = CTR_LanguageEncoder().to(device)
all_params = (list(jmf_encoder.parameters()) +
              list(language_encoder.parameters()))
optimizer = optim.Adam(all_params, lr)

print("Training started..")

# start training
for update_step in range(10000):

    sampled_states, sampled_language, effective_lengths = data_loader.sample(batch_size,feature_dim)
    language_embeddings = language_encoder(sampled_language)
    effective_states_embeddings = jmf_encoder(sampled_states.to(device),effective_lengths)

    optimizer.zero_grad()

    # normalizing
    language_embeddings = F.normalize(language_embeddings, p=2, dim=1)
    effective_states_embeddings = F.normalize(effective_states_embeddings, p=2, dim=1)

    labels = torch.arange(0, language_embeddings.size(0)).to(language_embeddings.device)

    # compute lan emb loss
    similarity_matrix_lan = torch.matmul(language_embeddings, effective_states_embeddings.T) / 0.1 # cos 
    lan_loss = F.cross_entropy(similarity_matrix_lan, labels)

    # compute jmf emb loss
    similarity_matrix_jmf = torch.matmul(effective_states_embeddings, language_embeddings.T) / 0.1 # cos 
    jmf_loss = F.cross_entropy(similarity_matrix_jmf, labels)

    total_loss = (lan_loss + jmf_loss) * 0.5

    writer.add_scalar('Training Loss', total_loss.item(), update_step)

    total_loss.backward()


    optimizer.step()

    print('update_step: %3d loss: %.5f'
        % (update_step + 1,
        total_loss.item()))
    
    if update_step % 500 == 0 and update_step != 0:
        torch.save(jmf_encoder.state_dict(), "jmf_encoder_" + run_name + "_" + str(update_step) +  ".pth")
        torch.save(language_encoder.state_dict(), "language_encoder_" + run_name + "_" + str(update_step) + ".pth")

writer.close()
torch.save(jmf_encoder.state_dict(), "jmf_encoder_" + run_name + ".pth")
torch.save(language_encoder.state_dict(), "language_encoder_" + run_name + ".pth")