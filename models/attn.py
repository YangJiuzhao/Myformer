import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)

class MultiAttentionLayer(nn.Module):
    '''
    The Multi Stage Attention Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, d_model, n_heads, d_ff = None, dropout=0.1):
        super(MultiAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        # self.patch_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        # self.norm5 = nn.LayerNorm(d_model)
        # self.norm6 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        # self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
        #                         nn.GELU(),
        #                         nn.Linear(d_ff, d_model))

    def forward(self, x):
        # x : batch patch_num dim_num seg_num d_model
        b, d, s, dd = x.shape
        # Cross Time Attention
        time_in = rearrange(x, 'b d s dd -> (b d) s dd')
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Attention
        dim_send = rearrange(dim_in, '(b d) s dd -> (b s) d dd', d = d)
        dim_receive = self.dim_attention(dim_send, dim_send, dim_send)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        # dim_enc = rearrange(dim_in, '(b d) s dd -> (b s) d dd', d = d)

        # Cross Patch Attention
        # patch_in = rearrange(dim_enc, '(b p s) d dd -> (b d s) p dd', p = p)
        # patch_enc = self.patch_attention(patch_in, patch_in, patch_in)
        # patch_enc = patch_in + self.dropout(patch_enc)
        # patch_enc = self.norm5(patch_enc)
        # patch_enc = patch_enc + self.dropout(self.MLP3(patch_enc))
        # patch_enc = self.norm6(patch_enc)

        final_out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)

        return final_out
