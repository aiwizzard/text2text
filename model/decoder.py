import torch.nn as nn

from .ffnet import FFNet
from .attention import SelfAttention, SourceTargetAttention
from .embedding import Embeddings, PositonalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        n_layers=6,
        head=8,
        model_dim=512,
        max_len=512,
        ff_dim=2048,
        dropout_rate=0.1,
    ):
        super(Decoder, self).__init__()
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(head, model_dim, ff_dim, dropout_rate)
                for _ in range(n_layers)
            ]
        )
        self.embedding = Embeddings(vocab_size, model_dim)
        self.postional_encoding = PositonalEncoding(model_dim, max_len, dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, memory, source_mask, target_mask):
        source_mask = source_mask.unsqueeze(-2)
        # apply word embedding
        x = self.embedding(x)
        # apply positional encoding
        x = self.postional_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        x = self.layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, head=8, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        # Self Attention layer, query, key and value come from previous layer
        self.self_attention = SelfAttention(model_dim, head, dropout_rate)
        # Source target attention layer, query come from encoded space.
        # key and value come from previous self attention layer
        self.st_attention = SourceTargetAttention(model_dim, head, dropout_rate)
        self.ffnet = FFNet(model_dim, ff_dim, dropout_rate)

    def forward(self, x, mem, source_mask, target_mask):
        # self attention
        x = self.self_attention(x, target_mask)
        # soure target attention
        x = self.st_attention(x, mem, source_mask)
        # feed forward network
        x = self.ffnet(x)
        return x
