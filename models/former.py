import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.RevIN import RevIN
from models.attn import MultiAttentionLayer
from models.embedding import Patch_embedding

from math import ceil

class former(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_lens, 
                d_model=512, d_ff = 1024, n_heads=8, a_layers=3, 
                dropout=0.0, device=torch.device('cuda:0')):
        super(former, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_lens = [int(seg_len) for seg_len in seg_lens.split(',')]
        self.patch_num = len(seg_lens)
        self.device = device

        self.revin = True
        if self.revin: self.revin_layer = RevIN(data_dim, affine=True, subtract_last=False)

        self.attentions = nn.ModuleList()
        for i in range(a_layers):
            self.attentions.append(
                MultiAttentionLayer(
                    d_model=d_model, 
                    n_heads=n_heads, 
                    d_ff=d_ff, 
                    dropout=dropout,
                    norm='')
            )
        
        self.enc_value_embeddings = nn.ModuleList()
        self.enc_pos_embedding = []
        self.pre_norms = nn.ModuleList()
        self.in_len_add = []
        self.total_seg_num = 0

        for seg_len in self.seg_lens:
            # The padding operation to handle invisible sgemnet length
            seg_num = ceil(1.0 * in_len / seg_len)
            self.total_seg_num += seg_num
            pad_in_len = seg_num * seg_len
            self.in_len_add.append(pad_in_len - in_len)

            # Embedding
            self.enc_value_embeddings.append(Patch_embedding(seg_len, d_model))
            self.enc_pos_embedding.append(nn.Parameter(torch.randn(1, data_dim, (pad_in_len // seg_len), d_model)).to(device))
            # self.enc_pos_embedding.append(positional_encoding("zeros", True, seg_num, d_model).to(device))
            self.pre_norms.append(nn.LayerNorm(d_model))

        # Predict
        self.Predict = nn.Linear(d_model * self.total_seg_num, out_len)

        
    def forward(self, x_seq):
        # norm
        if self.revin: 
            x_seq = self.revin_layer(x_seq, 'norm')

        x = 0
        for i, embedding, pre_norm in zip(range(self.patch_num),self.enc_value_embeddings,self.pre_norms):

            if (self.in_len_add[i] != 0):
                x_temp = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add[i], -1), x_seq), dim = 1)
            else:
                x_temp = x_seq

            x_temp = embedding(x_temp)
            x_temp += self.enc_pos_embedding[i]
            x_temp = pre_norm(x_temp)

            for attention in self.attentions:
                x_temp = attention(x_temp)

            if i == 0:
                x = x_temp
            else:
                x = torch.cat((x, x_temp),dim=2)

        for attention in self.attentions:
            x = attention(x)
        x = rearrange(x,'b d s dd -> b d (s dd)')

        final_y = self.Predict(x)

        predict = rearrange(final_y,'b d out -> b out d')
        
        # denorm
        if self.revin: 
            predict = self.revin_layer(predict, 'denorm')
        
        return predict