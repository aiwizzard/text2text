""" 
The input consist of 
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, model_dim=512, head=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.mult_attn = MultiHeadAttention(model_dim, head, dropout_rate)

    def forward(self, x, target_mask):
        out = self.layer_norm(x)
        out = self.mult_attn(out, out, out, target_mask)
        out = self.dropout(out)
        return out + x


class SourceTargetAttention(nn.Module):
    def __init__(self, model_dim=512, head=8, dropout_rate=0.1):
        super(SourceTargetAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.mult_attn = MultiHeadAttention(model_dim, head, dropout_rate)

    def forward(self, x, mem, source_mask):
        out = self.layer_norm(x)
        out = self.mult_attn(out, mem, mem, source_mask)
        out = self.dropout(out)
        return out + x


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, head=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        # assert that the model dimension deviced
        # by the number of heads is an integer value
        assert model_dim % head == 0, "invalid model dimension or number of heads"
        self.head = head
        self.key_dim = model_dim // self.head

        # To apply the linear transformation to the incomming data
        self.query_weights = nn.Linear(model_dim, model_dim)
        self.key_weights = nn.Linear(model_dim, model_dim)
        self.value_weights = nn.Linear(model_dim, model_dim)

        self.dense = nn.Linear(model_dim, model_dim)

        self.attention = ScaledDotProductAttention(model_dim, dropout_rate)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query = (
            self.query_weights(query)
            .contiguous()
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        key = (
            self.key_weights(key)
            .contiguous()
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        value = (
            self.value_weights(value)
            .contiguous()
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )

        out, attention = self.attention(query, key, value, mask)
        # Transpose to move the head dimension back and
        # combine the last two dimensions to concatenate all the heads together
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head * self.key_dim)
        )

        return self.dense(out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, model_dim, dropout_rate=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        dk = query.size(-1)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
        attention = self.layer_norm(attention)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        if self.dropout_rate is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, value)

        return output, attention
