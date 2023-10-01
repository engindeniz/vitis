from torch import Tensor, nn
from torch.nn import Linear, MultiheadAttention, Dropout, LayerNorm
from torch.nn import functional as F


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, feed_forward=False,
                 ) -> None:
        super().__init__()
        self.feed_forward = feed_forward

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

        if self.feed_forward:
            self.dropout1 = Dropout(dropout)
            self.dropout2 = Dropout(dropout)
            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)
            self.activation = F.gelu
            self.norm2 = LayerNorm(d_model)

    def forward(self, inputs: Tensor, attn_mask=None, key_padding_mask=None):
        x = inputs
        x = self.norm1(x)
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        x = self.dropout(x)
        x = inputs + x
        if self.feed_forward:
            x = x + self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(self.norm2(x))))))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 feed_forward=False) -> None:
        super().__init__()
        self.feed_forward = feed_forward
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm0 = LayerNorm(d_model)
        self.norm1 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        if self.feed_forward:
            self.dropout1 = Dropout(dropout)
            self.dropout2 = Dropout(dropout)
            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)
            self.activation = F.gelu
            self.norm2 = LayerNorm(d_model)

    def forward(self, q: Tensor, kv: Tensor,
                attn_mask=None, key_padding_mask=None):
        query = q
        key_value = kv

        query = self.norm0(query)
        key_value = self.norm1(key_value)
        x = self.cross_attn(query, key_value, key_value,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        x = self.dropout(x)
        x = x + q
        if self.feed_forward:
            x = x + self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(self.norm2(x))))))
        return x


class Perceiver(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 feed_forward=False, num_layers: int = 0) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.cross_attention_block = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout, feed_forward)
        if self.num_layers > 0:
            self.self_attention_block = nn.Sequential(
                *[SelfAttentionBlock(d_model, nhead, dim_feedforward, dropout, feed_forward) for _ in
                  range(num_layers)])

    def forward(self, q, kv):
        x = self.cross_attention_block(q, kv)
        if self.num_layers > 0:
            x = self.self_attention_block(x)
        return x
