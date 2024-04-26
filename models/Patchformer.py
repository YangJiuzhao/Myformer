import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from math import ceil

from models.RevIN import RevIN
from models.attn import AttentionLayer
from models.embedding import Patch_embedding, iPatch_embedding

class Patchformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_lens, 
                d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.2, device=torch.device('cuda:0')):
        super(Patchformer, self).__init__()

        self.seg_lens = [int(seg_len) for seg_len in seg_lens.split(',')]
        self.patch_num = len(seg_lens)
        self.device = device


        self.revin = True
        if self.revin: self.revin_layer = RevIN(data_dim, affine=True, subtract_last=False)

        self.embeddings = nn.ModuleList()
        self.enc_pos_embedding = []
        self.dec_pos_embedding = []
        self.pre_norms = nn.ModuleList()
        self.predicts = nn.ModuleList()
        self.in_len_add = []
        self.out_len_add = []
        self.total_out_seg_num = 0
        for seg_len in self.seg_lens:
            # The padding operation to handle invisible sgemnet length
            in_seg_num = ceil(1.0 * in_len / seg_len)
            pad_in_len = in_seg_num * seg_len
            self.in_len_add.append(pad_in_len - in_len)


            out_seg_num = ceil(1.0 * out_len / seg_len)
            self.total_out_seg_num += out_seg_num
            pad_out_len = out_seg_num * seg_len
            self.out_len_add.append(pad_out_len - out_len)

            # Embedding
            self.embeddings.append(Patch_embedding(seg_len, d_model))
            # self.enc_pos_embedding.append(nn.Parameter(torch.randn(1, data_dim, (pad_in_len // seg_len), d_model)).to(device))
            # self.dec_pos_embedding.append(nn.Parameter(torch.randn(1, data_dim, (pad_out_len // seg_len), d_model)).to(device))
            self.enc_pos_embedding.append(positional_encoding("zeros", True, in_seg_num, d_model).to(device))
            self.dec_pos_embedding.append(positional_encoding("zeros", True, out_seg_num, d_model).to(device))
            self.pre_norms.append(nn.LayerNorm(d_model))
            self.predicts.append(nn.Linear(d_model, seg_len))
        # Encoder
        self.encoder = Encoder(
            patch_num = self.patch_num,
            embeddings = self.embeddings,
            enc_pos_embedding = self.enc_pos_embedding,
            pre_norms = self.pre_norms,
            in_len_add = self.in_len_add,
            d_model=d_model,
            d_ff = d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout, 
            device=device)
        # Trend Predictor
        self.trend_predictor = iformer(data_dim, in_len, out_len, seg_lens, 
                    d_model, d_ff, n_heads, e_layers, 
                    dropout, device)
        
        # Decoder
        self.decoder = Decoder(
            out_len=out_len, 
            patch_num=self.patch_num, 
            d_layers = e_layers + 1, 
            d_model=d_model, 
            n_heads=n_heads, 
            d_ff=d_ff, 
            dropout=dropout, 
            embeddings=self.embeddings, 
            dec_pos_embeddings = self.dec_pos_embedding,
            pre_norms=self.pre_norms, 
            out_len_add=self.out_len_add, 
            predicts=self.predicts
        )
        
    def forward(self, x): 
        # norm
        if self.revin: 
            x = self.revin_layer(x, 'norm')
        
        enc_out = self.encoder(x)

        dec_in = self.trend_predictor(x)

        predict_y = self.decoder(dec_in, enc_out)
        # predict_y = dec_in

        # denorm
        if self.revin: 
            predict_y = self.revin_layer(predict_y, 'denorm')

        return predict_y
    

class Encoder(nn.Module):
    '''
    The Encoder of Patchformer.
    '''
    def __init__(self, patch_num, embeddings, enc_pos_embedding, pre_norms, in_len_add,
                d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.2, device=torch.device('cuda:0')):
        super(Encoder, self).__init__()
        self.patch_num = patch_num
        self.embeddings = embeddings
        self.enc_pos_embedding = enc_pos_embedding
        self.pre_norms = pre_norms
        self.in_len_add = in_len_add

        self.encode_layers = nn.ModuleList()
        for i in range(0, e_layers):
            self.encode_layers.append(EncoderLayer(d_model=d_model, d_ff = d_ff, n_heads=n_heads, a_layers=1, 
                dropout=dropout, device=device))

    def forward(self, x):
        x_seq = []
        x_cat = 0
        for i, embedding, pre_norm in zip(range(self.patch_num),self.embeddings, self.pre_norms):
            if (self.in_len_add[i] != 0):
                x_temp = torch.cat((x[:, :1, :].expand(-1, self.in_len_add[i], -1), x), dim = 1)
            else:
                x_temp = x
            x_temp = embedding(x_temp)
            x_temp += self.enc_pos_embedding[i]
            x_temp = pre_norm(x_temp)
            x_seq.append(x_temp)
            if i == 0:
                x_cat = x_temp
            else:
                x_cat = torch.concat((x_cat, x_temp),dim=2)

        encode_x = []
        encode_x.append(x_cat)
        
        for layer in self.encode_layers:
            x_seq, x_cat = layer(x_seq)
            encode_x.append(x_cat)

        return encode_x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff = 1024, n_heads=8, a_layers=1, 
                dropout=0.0, device=torch.device('cuda:0')):
        super(EncoderLayer, self).__init__()
        self.device = device

        self.attention_layers = nn.ModuleList()
        for i in range(a_layers):
            self.attention_layers.append(
                MultiAttentionLayer(
                    d_model=d_model, 
                    n_heads=n_heads, 
                    d_ff=d_ff, 
                    dropout=dropout)
            )
        
    def forward(self, x_seq):

        for attention in self.attention_layers:
            x_seq, x_cat = attention(x_seq)
    
        return x_seq, x_cat
    
class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, out_len, d_model, n_heads, d_ff, dropout, predicts):
        super(DecoderLayer, self).__init__()
        self.out_len = out_len

        self.self_attention = MultiAttentionLayer(d_model, n_heads, d_ff, dropout)    
        self.patch_attention = self.self_attention.time_attention
        self.cross_attention = self.self_attention.dim_attention
        self.norm1 = self.self_attention.norm1
        self.norm2 = self.self_attention.norm2
        self.norm3 = self.self_attention.norm3
        self.norm4 = self.self_attention.norm4        
        self.dropout_attn1 = self.self_attention.dropout_attn1
        self.dropout_attn2 = self.self_attention.dropout_attn2
        self.dropout_ffn1 = self.self_attention.dropout_ffn1
        self.dropout_ffn2 = self.self_attention.dropout_ffn2
        self.MLP1 = self.self_attention.MLP1
        self.MLP2 = self.self_attention.MLP2
        self.linear_pred = predicts

    def forward(self, x_seq, cross):

        x_list, _ = self.self_attention(x_seq)
        dec_output = []
        i = 0
        # out_cat = 0
        final_predict = 0
        for x in x_list:
            b, d, s, dd = x.shape
            # Cross Patch Attention
            patch_in = rearrange(x, 'b d s dd -> (b d) s dd')
            cross_send = rearrange(cross, 'b d s dd -> (b d) s dd')
            patch_enc = self.patch_attention(patch_in, cross_send, cross_send)
            dim_in = patch_in + self.dropout_attn1(patch_enc)
            dim_in = self.norm1(dim_in)
            dim_in = dim_in + self.dropout_ffn1(self.MLP1(dim_in))
            dim_in = self.norm2(dim_in)
            
            # Cross Dimension Attention
            dim_send = rearrange(dim_in, '(b d) s dd -> (b s) d dd', d = d)
            dim_receive = self.cross_attention(dim_send, dim_send, dim_send)
            dim_enc = dim_send + self.dropout_attn2(dim_receive)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout_ffn2(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)
            
            # output
            out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)
            predict = self.linear_pred[i](out)
            predict = rearrange(predict,'b d s dd -> b (s dd) d')
            final_predict = final_predict + predict[:,:self.out_len,:]

            dec_output.append(out)
            # if i == 0:
            #     out_cat = out
            # else:
            #     out_cat = torch.concat((out_cat,out), dim=2)
            i += 1

        # out_cat = rearrange(out_cat, 'b d s dd -> b d (s dd)', d = d)
        # final_predict = self.linear_pred(out_cat)
        # final_predict = rearrange(final_predict, 'b d s dd -> b d (s dd)', d = d)
        # final_predict = rearrange(final_predict,'b d out -> b out d')[:,:self.out_len,:]

        return dec_output, final_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, out_len, patch_num, d_layers, d_model, n_heads, d_ff, dropout, embeddings, dec_pos_embeddings, pre_norms, out_len_add, predicts
                 ):
        super(Decoder, self).__init__()        
        self.out_len = out_len
        self.patch_num = patch_num
        self.embeddings = embeddings
        self.dec_pos_embeddings = dec_pos_embeddings
        self.pre_norms = pre_norms
        self.out_len_add = out_len_add
        self.d_layers = d_layers

        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(out_len, d_model, n_heads, d_ff, dropout, predicts))
        
        self.moving_avg = moving_avg(25,1)

    def forward(self, x, cross):
        dec_in = []

        for i, embedding, pre_norm in zip(range(self.patch_num),self.embeddings,self.pre_norms):
            if (self.out_len_add[i] != 0):
                x_temp = torch.cat((x[:, :1, :].expand(-1, self.out_len_add[i], -1), x), dim = 1)
            else:
                x_temp = x

            x_temp = embedding(x_temp)
            x_temp += self.dec_pos_embeddings[i]
            x_temp = pre_norm(x_temp)
            dec_in.append(x_temp)

        final_predict = self.moving_avg(x)
        i = 0

        for layer in self.decode_layers:
            cross_enc = cross[i]
            dec_in, layer_predict = layer(dec_in,  cross_enc)
            if i < self.d_layers - 1:
                final_predict = final_predict + self.moving_avg(layer_predict)
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        return final_predict

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
        self.patch_attention = self.time_attention
        

        norm = 'batch_norm'
        
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        self.dropout_ffn1 = nn.Dropout(dropout)
        self.dropout_ffn2 = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm4 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))

    def forward(self, x_seq):
        # x_seq : patch_number batch dim_num seg_num d_model
        x_list = []
        x_cat = 0
        i = 0
        for x in x_seq:
            b, d, s, dd = x.shape
            # Cross Time Attention
            time_in = rearrange(x, 'b d s dd -> (b d) s dd')
            time_enc = self.time_attention(time_in, time_in, time_in)
            dim_in = time_in + self.dropout_attn1(time_enc)
            dim_in = self.norm1(dim_in)
            dim_in = dim_in + self.dropout_ffn1(self.MLP1(dim_in))
            dim_in = self.norm2(dim_in)

            # Cross Dimension Attention
            dim_send = rearrange(dim_in, '(b d) s dd -> (b s) d dd', d = d)
            dim_receive = self.dim_attention(dim_send, dim_send, dim_send)
            dim_enc = dim_send + self.dropout_attn2(dim_receive)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout_ffn2(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)

            # output
            out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)
            x_list.append(out)
            if i == 0:
                x_cat = out
            else:
                x_cat = torch.concat((x_cat,out),dim=2)
            i += 1

        patch_send = rearrange(x_cat, 'b d s dd -> (b d) s dd', d = d)

        final_out = []
        final_cat = 0
        i = 0
        for x in x_list:
            b, d, s, dd = x.shape
            # Cross Patch Attention
            patch_in = rearrange(x, 'b d s dd -> (b d) s dd')
            patch_enc = self.patch_attention(patch_in, patch_send, patch_send)
            dim_in = patch_in + self.dropout_attn1(patch_enc)
            dim_in = self.norm1(dim_in)
            dim_in = dim_in + self.dropout_ffn1(self.MLP1(dim_in))
            dim_in = self.norm2(dim_in)
            
            # Cross Dimension Attention
            dim_send = rearrange(dim_in, '(b d) s dd -> (b s) d dd', d = d)
            dim_receive = self.dim_attention(dim_send, dim_send, dim_send)
            dim_enc = dim_send + self.dropout_attn2(dim_receive)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout_ffn2(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)
            
            # output
            out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)
            final_out.append(out)
            if i == 0:
                final_cat = out
            else:
                final_cat = torch.concat((final_cat,out),dim=2)
            i += 1

        return final_out, final_cat
    
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
    

class iformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, patch_nums, 
                d_model=512, d_ff = 1024, n_heads=8, a_layers=3, 
                dropout=0.0, device=torch.device('cuda:0')):
        super(iformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.patch_nums = [in_len // int(patch_num) for patch_num in patch_nums.split(',')]
        self.patchs_num = len(patch_nums)
        self.device = device

        self.attentions = nn.ModuleList()
        for i in range(a_layers):
            self.attentions.append(
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
        x = []
        for i, embedding, pre_norm in zip(range(self.patchs_num),self.enc_value_embeddings,self.pre_norms):

            if (self.in_len_add[i] != 0):
                x_temp = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add[i], -1), x_seq), dim = 1)
            else:
                x_temp = x_seq

            x_temp = embedding(x_temp)
            x_temp += self.enc_pos_embedding[i]
            x_temp = pre_norm(x_temp)
            x.append(x_temp)

        for attention in self.attentions:
            x, y = attention(x)
            
        y = rearrange(y,'b d s dd -> b d (s dd)')

        final_y = self.Predict(y)

        predict = rearrange(final_y,'b d out -> b out d')
        
        return predict


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe
# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding
def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)