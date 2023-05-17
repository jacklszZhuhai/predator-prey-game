# 3*2，3 ,重要参数 是否act ，act的最大次数，以及循环次数

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import namedtuple
import math, copy, time
from torch.autograd import Variable
import numpy as np
# from utils2 import Channel, clip_gradient

# import torchtext.vocab as vocab

_iscomplex = True  # 暂时还没搞懂此处true和false的差别
# channel = Channel(_iscomplex=_iscomplex)  # 此处感觉没有使用到，等会可以放到训练部分的。


# _snr = 0

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, dense):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.channel_dim = 16  ##偷懒了
        self.num_hidden = 128
        self.from_channel_emb = nn.Sequential(nn.Linear(self.channel_dim, self.num_hidden * 2), nn.ReLU(),
                                              nn.Linear(self.num_hidden * 2, self.num_hidden))
        self.dense = dense

    #     def forward(self, src, tgt, src_mask, tgt_mask,_lens):
    #         batch_size = len(_lens)
    #         "Take in and process masked src and target sequences."
    #         output = self.encode(src, src_mask)
    #         output = channel.agwn(output, _snr=_snr)
    #         output = self.from_channel_emb(output)

    #         return self.decode(output, src_mask,
    #                             tgt, tgt_mask)

    #     def emboutput(self, src, src_mask):  #此处test会用到，感觉修改之后不一定会用的到了
    #         "Take in and process masked src and target sequences."
    #         output = self.encode(src, src_mask)
    #         if useawgn == True:
    #             output = channel.agwn(output, _snr=_snr)
    #         else:
    #             output = channel.fading(output, _snr=_snr)
    #         output = self.from_channel_emb(output)
    #         return output

    def encode(self, src, src_mask):
        return self.encoder.forward(self.src_embed.forward(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder.forward(self.tgt_embed.forward(tgt), memory, src_mask, tgt_mask)


class Denoiser(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, denoise1, denoise2, denoise3, denoise4):
        super(Denoiser, self).__init__()
        self.denoise1 = denoise1
        self.denoise2 = denoise2
        self.denoise3 = denoise3
        self.denoise4 = denoise4

    def denoise11(self, memory, snr):
        return self.denoise1(memory, snr)

    def denoise12(self, memory, snr):
        return self.denoise2(memory, snr)

    def denoise13(self, memory, snr):
        return self.denoise3(memory, snr)

    def denoise14(self, memory, snr):
        return self.denoise4(memory, snr)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # 此处一个（h,h）的全连接层，可以考虑加入

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, hidden_size, act=False):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.norm1 = LayerNorm(16)
        self.act = act
        self.num_layers = N
        self.hidden_size = hidden_size
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                                                  nn.Linear(256, 16))
        self.positionalEncoding = PositionalEncoding(128, 0)
        # 此处偷懒了没使用形参，直接建立了（h，h） 与 （h,c） h 为隐藏层 c为信道维数
        if (self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, x, mask):  # encode内置了信道编码
        "Pass the input (and mask) through each layer in turn."
        if self.act == False:
            for cishu in range(0, 2):
                for layer in self.layers:
                    x = layer(x, mask)

            x = self.to_chanenl_embedding(x)
            return self.norm1(x)
        else:
            x, (remainders, n_updates) = self.act_fn(x, x, mask, None, self.layers, self.num_layers)
            x = self.to_chanenl_embedding(x)
            # 此处的self.dec应该是只走一层的encode,在有dropout的情况下，x和input不一样，否则两者一样，
            return self.norm1(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)




class Decoder2(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, decoder, tgt_embed, generator, N1):
        super(Decoder2, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(N1, 256), nn.ReLU(),
                                                  nn.Linear(256, 128))

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder.forward(self.tgt_embed.forward(tgt), memory, src_mask, tgt_mask)



class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 原先此处为0，感觉不正确 应该为1！又被我改成了0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Dense1(nn.Module):
    "Implement the PE function."

    def __init__(self, n1, L):
        super(Dense1, self).__init__()
        self.layer1 = nn.Linear(n1, 1)
        self.layer2 = nn.Linear(L + 1, 1)
        self.layer2.bias.data.fill_(9)
        self.device = torch.device("cuda:0")

    def forward(self, x, _snr):
        x = self.layer1(x)
        snr = torch.tensor(_snr)
        snr1 = torch.zeros(x.shape[0], 1, 1)
        snr = snr + snr1
        snr = snr.to(self.device)
        x = torch.cat((x, snr), 1)  # B*L+1*1
        x = torch.squeeze(x)
        x = self.layer2(x)  # B*1
        # print(x.shape)
        return x


class Denoise1(nn.Module):
    "Implement the PE function."

    def __init__(self, n1, L):
        super(Denoise1, self).__init__()
        self.layer1 = nn.Linear(n1, 1)
        self.layer2 = nn.Linear(L, L - 1)
        self.device = torch.device("cuda:0")

    def forward(self, x, _snr):
        x = self.layer1(x)
        snr = torch.tensor(_snr).to(self.device)
        snr1 = torch.zeros(x.shape[0], 1, 1).to(self.device)
        snr = snr + snr1
        snr = snr.to(self.device)
        x = torch.cat((x, snr), 1)
        x = torch.squeeze(x)
        x = self.layer2(x)
        x = torch.unsqueeze(x, -1)
        # print(x.shape)
        return x



def make_denoiser(N1=16, N2=32):
    model = Denoiser(Denoise1(N1, N2), Denoise1(N1, N2), Denoise1(N1, N2), Denoise1(N1, N2))
    return model


class Denoiser1(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, hidden_size):
        super(Denoiser1, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.num_layers = N
        self.hidden_size = hidden_size

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, None)
        return self.norm(x)


##################encoder
def make_denoiser1(N=3, N1=32,
                   d_model=256, d_ff=1024, h=8, dropout=0.1, act1=False,
                   act2=False):  # N为有几层，d_model为词向量维度，h表示头的数目，ff是中间的过度，前向传输时候用到
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # position = PositionalEncoding(d_model, dropout)
    model = Denoiser1(EncoderLayer(d_model, c(attn), c(ff), dropout), N, d_model)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model








