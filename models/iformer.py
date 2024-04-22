import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.attn import MultiAttentionLayer
from models.embedding import iPatch_embedding

from math import ceil

class iformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, patch_nums, 
                d_model=512, d_ff = 1024, n_heads=8, a_layers=3, 
                dropout=0.0, device=torch.device('cuda:0')):
        super(iformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.patch_nums = [int(patch_num) for patch_num in patch_nums.split(',')]
        self.patchs_num = len(patch_nums)
        self.device = device

        # Attention
        # self.inner_attentions = nn.ModuleList()
        # for i in range(self.patchs_num):#a_layers):
        #     self.inner_attentions.append(
        #         MultiAttentionLayer(
        #             d_model=d_model, 
        #             n_heads=n_heads, 
        #             d_ff=d_ff, 
        #             dropout=dropout)
        #     )

        self.outer_attentions = nn.ModuleList()
        for i in range(a_layers):
            self.outer_attentions.append(
                MultiAttentionLayer(
                    d_model=d_model, 
                    n_heads=n_heads, 
                    d_ff=d_ff, 
                    dropout=dropout)
            )
        
        self.enc_value_embeddings = nn.ModuleList()
        self.enc_pos_embedding = []
        self.pre_norms = nn.ModuleList()
        self.in_len_add = []
        self.total_patch_num = 0

        for patch_num in self.patch_nums:
            # The padding operation to handle invisible sgemnet length
            patch_len = ceil(1.0 * in_len / patch_num)
            self.total_patch_num += patch_num
            pad_in_len = patch_num * patch_len
            self.in_len_add.append(pad_in_len - in_len)

            # Embedding
            self.enc_value_embeddings.append(iPatch_embedding(patch_len, d_model))
            self.enc_pos_embedding.append(nn.Parameter(torch.randn(1, data_dim, patch_num, d_model)).to(device))
            self.pre_norms.append(nn.LayerNorm(d_model))

        # Predict
        self.Predict = nn.Linear(self.total_patch_num*d_model, out_len)#

        
    def forward(self, x_seq):
        x = 0
        for i, embedding, pre_norm in zip(range(self.patchs_num),self.enc_value_embeddings,self.pre_norms):

            if (self.in_len_add[i] != 0):
                x_temp = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add[i], -1), x_seq), dim = 1)
            else:
                x_temp = x_seq

            x_temp = embedding(x_temp)
            x_temp += self.enc_pos_embedding[i]
            x_temp = pre_norm(x_temp)

            # x_temp = self.inner_attentions[i](x_temp)

            # for attention in self.outer_attentions:
            #     x_temp = attention(x_temp)

            if i == 0:
                x = x_temp
            else:
                x = torch.cat((x, x_temp),dim=2)

        for attention in self.outer_attentions:
            x = attention(x)
            
        x = rearrange(x,'b d s dd -> b d (s dd)')

        final_y = self.Predict(x)

        # final_y = rearrange(final_y,'b d s out -> b d out s')
        # final_y = torch.sum(final_y , dim= 3)

        predict = rearrange(final_y,'b d out -> b out d')
        
        return predict