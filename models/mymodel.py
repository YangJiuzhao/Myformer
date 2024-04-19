import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.former import former

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args,device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
        self.Former_Seasonal = former(self.args.data_dim, self.args.in_len, self.args.out_len, self.args.seg_lens, 
                self.args.d_model, self.args.d_ff, self.args.n_heads, self.args.a_layers, 
                self.args.dropout, self.device)
        self.Former_Trend = former(self.args.data_dim, self.args.in_len, self.args.out_len, self.args.seg_lens, 
                self.args.d_model, self.args.d_ff, self.args.n_heads, self.args.a_layers, 
                self.args.dropout, self.device)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        
        seasonal_output = self.Former_Seasonal(seasonal_init)
        trend_output = self.Former_Trend(trend_init)

        x = seasonal_output + trend_output
        return x  # to [Batch, Output length, Channel]
