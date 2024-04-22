import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from math import ceil

from models.attn import AttentionLayer
from models.embedding import Patch_embedding
from models.iformer import iformer

class Patchformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_lens, 
                d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.2, device=torch.device('cuda:0')):
        super(Patchformer, self).__init__()

        self.seg_lens = [int(seg_len) for seg_len in seg_lens.split(',')]
        self.patch_num = len(seg_lens)
        self.device = device

        self.embeddings = nn.ModuleList()
        self.enc_pos_embedding = []
        self.pre_norms = nn.ModuleList()
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
            self.enc_pos_embedding.append(nn.Parameter(torch.randn(1, data_dim, (pad_in_len // seg_len), d_model)).to(device))
            self.pre_norms.append(nn.LayerNorm(d_model))
        self.predict = nn.Linear(self.total_out_seg_num*d_model, out_len)
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
        self.dec_pos_embedding = iformer(data_dim, in_len, out_len, seg_lens, 
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
            pre_norms=self.pre_norms, 
            out_len_add=self.out_len_add, 
            predicts=self.predict
            )
        
    def forward(self, x):
        
        enc_out = self.encoder(x)

        dec_in = self.dec_pos_embedding(x)

        predict_y = self.decoder(dec_in, enc_out)

        return predict_y
    

class Encoder(nn.Module):
    '''
    The Encoder of Patchformer.
    '''
    def __init__(self, patch_num, embeddings,enc_pos_embedding,pre_norms,in_len_add,
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
            # x_temp += self.enc_pos_embedding[i]
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
    def __init__(self, out_len, d_model, n_heads, d_ff, dropout, predict):
        super(DecoderLayer, self).__init__()
        self.out_len = out_len

        self.self_attention = MultiAttentionLayer(d_model, n_heads, d_ff, dropout)    
        self.patch_attention = self.self_attention.time_attention
        self.cross_attention = self.self_attention.dim_attention
        self.norm1 = self.self_attention.norm1
        self.norm2 = self.self_attention.norm2
        self.norm3 = self.self_attention.norm3
        self.norm4 = self.self_attention.norm4
        self.dropout = self.self_attention.dropout
        self.MLP1 = self.self_attention.MLP1
        self.MLP2 = self.self_attention.MLP2
        self.linear_pred = predict

    def forward(self, x_seq, cross):

        x_list, _ = self.self_attention(x_seq)
        dec_output = []
        i = 0
        for x in x_list:
            b, d, s, dd = x.shape
            # Cross Patch Attention
            patch_in = rearrange(x, 'b d s dd -> (b d) s dd')
            cross_send = rearrange(cross, 'b d s dd -> (b d) s dd')
            patch_enc = self.patch_attention(patch_in, cross_send, cross_send)
            dim_in = patch_in + self.dropout(patch_enc)
            dim_in = self.norm1(dim_in)
            dim_in = dim_in + self.dropout(self.MLP1(dim_in))
            dim_in = self.norm2(dim_in)
            
            # Cross Dimension Attention
            dim_send = rearrange(dim_in, '(b d) s dd -> (b s) d dd', d = d)
            dim_receive = self.cross_attention(dim_send, dim_send, dim_send)
            dim_enc = dim_send + self.dropout(dim_receive)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)
            
            # output
            out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)
            dec_output.append(out)
            if i == 0:
                out_cat = out
            else:
                out_cat = torch.concat((out_cat,out), dim=2)

        out_cat = rearrange(out_cat, 'b d s dd -> (b d) s dd', d = d)
        final_predict = self.linear_pred(out_cat)

        return dec_output, final_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, out_len, patch_num, d_layers, d_model, n_heads, d_ff, dropout, embeddings, pre_norms, out_len_add, predicts
                 ):
        super(Decoder, self).__init__()        
        self.out_len = out_len
        self.patch_num = patch_num
        self.embeddings = embeddings
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
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
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

            # output
            out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)
            x_list.append(out)
            if i == 0:
                x_cat = out
            else:
                x_cat = torch.concat((x_cat,out),dim=2)

        patch_send = rearrange(x_cat, 'b d s dd -> (b d) s dd', d = d)

        final_out = []
        final_cat = 0

        for x in x_list:
            b, d, s, dd = x.shape
            # Cross Patch Attention
            patch_in = rearrange(x, 'b d s dd -> (b d) s dd')
            patch_enc = self.patch_attention(patch_in, patch_send, patch_send)
            dim_in = patch_in + self.dropout(patch_enc)
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
            
            # output
            out = rearrange(dim_enc, '(b s) d dd -> b d s dd', b = b , d = d)
            final_out.append(out)
            if i == 0:
                final_cat = out
            else:
                final_cat = torch.concat((final_cat,out),dim=2)

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