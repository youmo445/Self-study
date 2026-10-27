import torch
from torch import Tensor
import math


def manual_softmax(x:Tensor, dim:int=-1) -> Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    softmax_x = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return softmax_x

def scaled_dot_product_attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor | None = None) -> tuple[Tensor, Tensor]:
    scores = q @ k.transpose(-2, -1) # [B, q_len, k_len]
    scaled_scores = scores / math.sqrt(q.size(-1))
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))
    weights = manual_softmax(scaled_scores)
    context = weights @ v # [B, q_len, dim]
    return context, weights

def make_causal_mask(seq_len:int) -> Tensor:
    mask = torch.ones((seq_len,seq_len)).tril().bool()
    return mask

def make_padding_mask(token_ids:Tensor, pad_id:int) -> Tensor:
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids shape must be [B, seq_len]!")
    valid_tokes = token_ids!=pad_id
    mask = valid_tokes[:,None,None,:] # [B, 1, 1, seq_len]
    return mask
    
def split_heads(x:Tensor, num_heads:int) -> Tensor:
    B, seq_len, dim = x.shape 
    if dim % num_heads != 0:
        raise ValueError(f"dim = {dim} must be divided by num_heads = {num_heads}")
    dim_heads = dim // num_heads
    x = x.reshape(B, seq_len, num_heads, dim_heads)
    x = x.transpose(-3, -2)
    return x

def merge_heads(x:Tensor) -> Tensor:
    B, num_heads, seq_len, dim_heads = x.shape
    x = x.transpose(-3, -2)
    x = x.contiguous()
    x = x.reshape(B, seq_len, num_heads * dim_heads)
    return x

def sinusoidal_position_encoding(max_len:int, d_model:int):
    position = torch.arange(max_len, dtype = torch.float32).unsqueeze(1) # [L, 1]
    even_indices = torch.arange(0, d_model, 2, dtype = torch.float32) # [d'=(d+1)//2]
    frequency_factors = torch.pow(10000.0, -even_indices/d_model) # [d'=(d+1)//2]
    angles = position * frequency_factors # [L, d']
    pos_emb = torch.zeros((max_len, d_model)) # [L, d]
    pos_emb[:, 0::2] = torch.sin(angles)
    num_edd_dimensions = pos_emb[:, 1::2].shape[-1]
    pos_emb[:, 1::2] = torch.cos(angles[:,:num_edd_dimensions])
    pos_emb = pos_emb.unsqueeze(0) # [1, L, d]
    return pos_emb

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_head:int):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        if self.d_model % self.num_head != 0:
            raise ValueError(f"dim = {self.d_model} must be divided by num_heads = {self.num_head}")
        self.d_head = self.d_model // self.num_head
        self.q_proj = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.k_proj = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.v_proj = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.out_proj = torch.nn.Linear(in_features=d_model, out_features=d_model)
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None=None):
        # 同维度线性投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # 拆分多头
        q = split_heads(q, num_heads=self.num_head) #[B, num_heads, q_len, d_heads]
        k = split_heads(k, num_heads=self.num_head) #[B, num_heads, v_len, d_heads]
        v = split_heads(v, num_heads=self.num_head) #[B, num_heads, v_len, d_heads]
        # 算注意力
        context, weights = scaled_dot_product_attention(q, k, v, mask) # [B, num_heads, q_len, d_heads], [B, num_heads, q_len, v_len]
        # 合并多头
        context = merge_heads(context) # [B, q_len, d]
        # 输出投影
        context = self.out_proj(context)
        return context, weights
    def forward_self_kv_cache(self, x_new, last_k, last_v):
        if (last_k is None) != (last_v is None):
            raise ValueError(f"last_k and last v must be either none or not none!!")

        q_new = self.q_proj(x_new) # [B, 1, D]
        k_new = self.k_proj(x_new) # 
        v_new = self.v_proj(x_new)

        q_new = split_heads(q_new, num_heads=self.num_head) # [B, num_heads, 1, d_heads]
        k_new = split_heads(k_new, num_heads=self.num_head) # [B, num_heads, 1, d_heads]
        v_new = split_heads(v_new, num_heads=self.num_head) # [B, num_heads, 1, d_heads]
        if last_v is None:
            k = k_new
            v = v_new
        else:
            k = torch.concat([last_k, k_new], dim=2) # [B, num_heads, n, d_heads]
            v = torch.concat([last_v, v_new], dim=2)
        context, weights = scaled_dot_product_attention(q_new, k, v, mask=None) # [B, num_heads, 1, d_heads]
        context = merge_heads(context) # [B, 1, d]
        context = self.out_proj(context) # [B, 1, d]
        return context, weights, k, v
    def forward_cross_kv_cache(self, x_new, memory, memory_k, memory_v, src_mask):
        if (memory_k is None) != (memory_v is None):
            raise ValueError(f"last_k and last v must be either none or not none!!")
        q_new = self.q_proj(x_new)
        if memory_k is None:
            memory_k = self.k_proj(memory)
            memory_v = self.v_proj(memory)
            memory_k = split_heads(memory_k, num_heads=self.num_head) # [B, num_heads, S, d_heads]
            memory_v = split_heads(memory_v, num_heads=self.num_head) # [B, num_heads, S, d_heads]
        q_new = split_heads(q_new, num_heads=self.num_head) # [B, num_heads, 1, d_heads]
        context, weights = scaled_dot_product_attention(q_new, memory_k, memory_v, mask=src_mask)
        context = merge_heads(context) # [B, 1, d]
        context = self.out_proj(context) # [B, 1, d]
        return context, weights, memory_k, memory_v 

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len:int, d_model:int):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        pos_emb = sinusoidal_position_encoding(self.max_len, self.d_model)
        self.register_buffer("pos_emb", pos_emb)
    def forward(self, x:Tensor, start_pos:int = 0):
        if start_pos<0:
            raise ValueError(f"start_pos={start_pos} must be non negative!")
        if x.ndim != 3:
            raise ValueError(f"input token must have shape [B, L, D]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"check dimension!")
        if x.shape[1] > self.max_len: # [B, L, D]
            raise ValueError(f"seq_len:{x.shape[1]} must not be larger than max_len:{self.max_len}")
        end_pos = start_pos + x.shape[1]
        if end_pos > self.max_len:
            raise ValueError(f"output length={end_pos} larger than max_len={self.max_len}")
        x = x + self.pos_emb[:,start_pos:end_pos]
        return x

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.mlp1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU()
        self.mlp2 = torch.nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_head: int, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

    def forward(self,x:Tensor, src_mask:Tensor|None=None):
        attention_outputs, attention_weights = self.self_attn(x, x, x, src_mask)
        x = self.ln1(x + self.dropout1(attention_outputs))
        x = self.ln2(x + self.dropout2(self.ffn(x)))
        return x, attention_weights

class Encoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_head: int, num_layers: int,dropout:float=0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            EncoderLayer(d_model=d_model, d_ff=d_ff, num_head=num_head, dropout=dropout)
            for _ in range(num_layers)
        )
    def forward(self, x: Tensor, src_mask: Tensor):
        attention_weights_list = []
        for layers in self.layers:
            x, attention_weights = layers(x, src_mask)
            attention_weights_list.append(attention_weights)
        return x, attention_weights_list

class TransformerEncoder(torch.nn.Module):
    def __init__(self, src_vocab_size,max_len, pad_id, d_model, d_ff, num_head, num_layers,dropout:float=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(src_vocab_size, d_model,pad_id)
        self.dropout = torch.nn.Dropout(dropout)
        self.position_encoding = PositionalEncoding(max_len, d_model)
        self.encoder = Encoder(d_model, d_ff, num_head, num_layers, dropout)
    def forward(self, src_token_id, src_mask: Tensor|None=None):
        if src_mask is None:
            src_mask = make_padding_mask(src_token_id, self.pad_id)
        x = self.embedding(src_token_id) # [B, L] -> [B, L, d]
        x = math.sqrt(self.d_model) * x
        x = self.position_encoding(x)
        x = self.dropout(x)
        x, weight_list = self.encoder(x, src_mask)
        return x, weight_list

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_head: int, dropout):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ln3 = torch.nn.LayerNorm(d_model)

    def forward(self,x:Tensor, memory, src_mask:Tensor|None=None, tgt_mask:Tensor|None=None):
        # 第一个maskattn
        attention_outputs_1, attention_weights_1 = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.ln1(x + self.dropout1(attention_outputs_1))
        # 第二个cross_attn
        attention_outputs_2, attention_weights_2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.ln2(x + self.dropout2(attention_outputs_2))
        # 第三个FFN
        x = self.ln3(x + self.dropout3(self.ffn(x)))
        return x, attention_weights_1, attention_weights_2
    def forward_kv_cache(self, x, memory, last_k, last_v, memory_k, memory_v, src_mask):
        # 第一个maskattn
        attention_outputs_1, attention_weights_1, k, v = self.masked_self_attn.forward_self_kv_cache(x, last_k, last_v)
        x = self.ln1(x + self.dropout1(attention_outputs_1))
        # 第二个cross_attn
        attention_outputs_2, attention_weights_2, memory_k, memory_v = self.cross_attn.forward_cross_kv_cache(x, memory, memory_k, memory_v, src_mask)
        x = self.ln2(x + self.dropout2(attention_outputs_2))
        # 第三个FFN
        x = self.ln3(x + self.dropout3(self.ffn(x)))
        return x, attention_weights_1, attention_weights_2, k, v, memory_k, memory_v     

class Decoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_head, num_layers, dropout: float=0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            DecoderLayer(d_model, d_ff, num_head, dropout)
            for _ in range(num_layers)
        )
    def forward(self, x, memory, tgt_mask, src_mask):
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
        return x, self_attn_weights_list, cross_attn_weights_list
    def forward_kv_cache(self, x, memory, layer_caches=None, src_mask=None):
        self_attn_weights_list = []
        cross_attn_weights_list = []
        new_layer_caches = []
        if layer_caches is None:
            layer_caches=[
                {"k": None,
                 "v": None,
                 "memory_k": None,
                 "memory_v": None,
                 }
                 for _ in range(len(self.layers))
            ]
        if len(layer_caches) != len(self.layers):
            raise ValueError(f"number of caches must equal number of decoder layers!")
        for index, layer in enumerate(self.layers):
            cache = layer_caches[index]
            x, self_attn_weights, cross_attn_weights, k, v, memory_k, memory_v = layer.forward_kv_cache(x, memory, cache["k"], cache["v"], cache["memory_k"], cache["memory_v"], src_mask=src_mask)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
            new_layer_caches.append(
                {
                    "k": k,
                    "v": v,
                    "memory_k": memory_k,
                    "memory_v": memory_v,
                }
            )
        return x, self_attn_weights_list, cross_attn_weights_list, new_layer_caches

class TransformerDecoder(torch.nn.Module):
    def __init__(self, tgt_vocab_size,max_len, pad_id, d_model, d_ff, num_head, num_layers,dropout:float=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.position_encoding = PositionalEncoding(max_len, d_model)
        self.decoder = Decoder(d_model, d_ff, num_head, num_layers, dropout)
        self.embedding = torch.nn.Embedding(tgt_vocab_size, d_model, pad_id)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, tgt_token_id, memory, src_mask=None, tgt_mask=None):
        if tgt_mask is None:
            tgt_padding_mask = make_padding_mask(tgt_token_id, self.pad_id) # [B, 1, 1, L]
            casual_mask = make_causal_mask(tgt_token_id.shape[-1]).to(tgt_token_id.device)
            tgt_mask = tgt_padding_mask & casual_mask
        x = self.embedding(tgt_token_id)
        x = math.sqrt(self.d_model) * x
        x = self.position_encoding(x)
        x = self.dropout(x)
        x, self_attn_weight_list, cross_attn_weight_list = self.decoder(x, memory,tgt_mask, src_mask)
        return x, self_attn_weight_list, cross_attn_weight_list
    def forward_kv_cache(self, tgt_token_id, memory, layer_caches=None,src_mask=None):
        if tgt_token_id.ndim != 2 or tgt_token_id.shape[1] != 1:
            raise ValueError(
                "tgt_token_id must have shape [B, 1] during cached decoding"
            )
        if layer_caches is None:
            start_pos = 0
        else:
            start_pos = layer_caches[0]["k"].shape[2]
        x = self.embedding(tgt_token_id)
        x = math.sqrt(self.d_model) * x
        x = self.position_encoding(x, start_pos)
        x = self.dropout(x)
        x, self_attn_weight_list, cross_attn_weight_list, new_layer_caches = self.decoder.forward_kv_cache(x, memory, layer_caches, src_mask)
        return x, self_attn_weight_list, cross_attn_weight_list, new_layer_caches

class Transformer(torch.nn.Module):
    def __init__(self, src_pad_id, tgt_pad_id, src_vocab_size, tgt_vocab_size,max_len, d_model, d_ff, num_head, num_layers,dropout:float=0.1):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.encoder = TransformerEncoder(src_vocab_size,max_len, src_pad_id, d_model, d_ff, num_head, num_layers,dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size,max_len, tgt_pad_id, d_model, d_ff, num_head, num_layers,dropout)
        self.out_proj = torch.nn.Linear(d_model, tgt_vocab_size)
    def forward(self, src_token_id, tgt_token_id, src_mask:Tensor|None=None, tgt_mask:Tensor|None=None):
        if src_mask is None:
            src_mask = make_padding_mask(src_token_id, self.src_pad_id) # [B,1,1,L]
        memory, encoder_weights = self.encoder(src_token_id, src_mask)
        output, decoder_self_attn_weights, decoder_cross_attn_weights = self.decoder(tgt_token_id, memory,src_mask,tgt_mask) # [B, q, d]
        logits = self.out_proj(output)
        return logits, encoder_weights, decoder_self_attn_weights, decoder_cross_attn_weights
    def decode_kv_cache(self,tgt_token_id, memory, layer_caches, src_mask):
        output, decoder_self_attn_weights, decoder_cross_attn_weights, new_layer_caches = self.decoder.forward_kv_cache(tgt_token_id, memory, layer_caches, src_mask) # [B, 1, d]
        logits = self.out_proj(output)
        return logits, decoder_self_attn_weights, decoder_cross_attn_weights, new_layer_caches
        

def shift_targets(tgt_token_ids):
    decoder_input = tgt_token_ids[:,:-1]
    labels = tgt_token_ids[:,1:]
    return decoder_input, labels