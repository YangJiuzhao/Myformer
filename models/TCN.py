import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, dilation=dilation, padding=0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        # 第二个卷积层
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, stride=stride, dilation=dilation, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        # 组合两个卷积层
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        # 如果输入通道数与输出通道数不一致，则添加一个1x1卷积层进行降维或升维
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x的形状: (batch_size, input_size, sequence_length)
        out = self.net(x)  # 通过两个卷积层进行特征提取，out的形状: (batch_size, output_size, sequence_length - kernel_size + 1)
        res = x if self.downsample is None else self.downsample(x)  # 如果需要降维或升维，则使用1x1卷积层，res的形状: (batch_size, output_size, sequence_length)
        return self.relu(out + res)  # 残差连接并使用ReLU激活函数，返回值的形状: (batch_size, output_size, sequence_length)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # 添加TemporalBlock
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # x的形状: (batch_size, input_size, sequence_length)
        out = self.network(x.transpose(1, 2))  # 将序列维度转置，以适应卷积层的输入要求，out的形状: (batch_size, num_channels[-1], sequence_length - kernel_size + 1)
        out = out.transpose(1, 2).contiguous()  # 将序列维度转置回来，out的形状: (batch_size, sequence_length - kernel_size + 1, num_channels[-1])
        out = self.fc(out[:, -1, :])  # 最后一个时间步的输出进行预测，out的形状: (batch_size, output_size)
        return out