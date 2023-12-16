import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from models.utils import get_params

class CTR_LanguageEncoder(nn.Module):
    """
    Encoder to embed languages using pre-trained bert
    """
    def __init__(self):
        super(CTR_LanguageEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        for name, param in self.text_model.named_parameters():
            # if 'layer' in name and int(name.split('.')[2]) < 10:  # 冻结前5层
            param.requires_grad = False

        for name, param in self.text_model.named_parameters():
            # if 'layer' in name and int(name.split('.')[2]) < 10:  # 冻结前5层
            param.requires_grad = False
        
        self.para = get_params()


        self.fc1 = nn.Linear(768,self.para["num_node"])
        
    def forward(self, input_language):
        
        # tokenlizing
        max_length = 25
        encoded_languages = [self.tokenizer(language, return_tensors="pt", padding="max_length", truncation=True, max_length=512) for language in input_language]
       
        input_ids = torch.cat([e['input_ids'] for e in encoded_languages], dim=0).to("cuda")
        attention_mask = torch.cat([e['attention_mask'] for e in encoded_languages], dim=0).to("cuda")
        
        hidden = self.text_model(input_ids,attention_mask).pooler_output
        out = F.leaky_relu(self.fc1(hidden))
        return out
        


