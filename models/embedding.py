import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class Patch_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(Patch_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        return x_embed
    
class iPatch_embedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super(iPatch_embedding, self).__init__()
        self.patch_len = patch_len
        self.linear = nn.Linear(patch_len, d_model)

    def forward(self, x):

        batch, ts_len, ts_dim = x.shape
        iPatch_num = ts_len // self.patch_len

        x_segment = rearrange(x, 'b (pl pn) d -> b d pn pl',pn = iPatch_num)

        x_embed = self.linear(x_segment)

        return x_embed