import torch.nn as nn
import torch
import math
import numpy as np

# Transformer Parameters
d_model = 128  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
k = 1 # distance of local pattern
#batch_size = 16 # batch size
max_len = 5000


def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return torch.FloatTensor(positional_encoding)


def local_pattern(bs, seq_len):
    lp = torch.zeros(seq_len, seq_len)
    diff = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1))
    k_clamped = k if k < seq_len else seq_len - 1
    lp[diff > k_clamped] = 1
    lp = lp.bool().expand(bs, seq_len, seq_len)
    return lp

def global_pattern(bs, seq_len):
    gp = torch.zeros(seq_len, seq_len)
    gp[0,:] = gp[:,0] = 1
    gp = gp.bool().expand(bs, seq_len, seq_len)
    return ~gp


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q: [batch_size, seq_len]
    # seq_k: [batch_size, seq_len]
    # seq_len could be src_len or it could be tgt_len
    # seq_len in seq_q and seq_len in seq_k maybe not equal
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    zero = torch.zeros(batch_size, len_k, d_model).cuda()
    # eq(zero) is PAD token
    pad_mask = (seq_k == zero).all(dim=2).unsqueeze(1)

    return pad_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        layer_norm = nn.LayerNorm(d_model).cuda()
        return layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        layer_norm = nn.LayerNorm(d_model).cuda()
        return layer_norm(output + residual) # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.pos_emb = get_positional_encoding(max_seq_len=max_len, embed_dim=d_model).cuda() # [max_seq_len, d_model]

    def forward(self, input):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_inputs, ap = input[0], input[1]
        batch_size, len, _ = enc_inputs.size()
        src_len = enc_inputs.size(1)
        pos_emb = self.pos_emb[0 : src_len] # [src_len, d_model]
        enc_outputs = enc_inputs + pos_emb # [batch_size, src_len, d_model]

        lp = local_pattern(batch_size, len)
        gp = global_pattern(batch_size, len)
        fp = ~(~lp + ~gp + ~ap)
        #fp = ~(~gp + ~ap)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        #enc_self_attn_mask = enc_self_attn_mask + fp.cuda() # [batch_size, src_len, src_len]

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask) # enc_outputs: [batch_size, src_len, d_model]
            enc_self_attns.append(enc_self_attn) # enc_self_attn: [batch_size, n_heads, src_len, src_len]
        return enc_outputs, enc_self_attns


class SparseTransformer(nn.Module):
    "Transformer with Sparse Attention."

    def __init__(self):
        super(SparseTransformer, self).__init__()
        self.encoder = Encoder()
    
    def forward(self, x): # x: [[batch_size, src_len, d_model], [batch_size, src_len, src_len]]
        outputs, self_attns = self.encoder(x)

        return outputs, self_attns
