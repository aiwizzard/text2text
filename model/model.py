from numpy import source
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from .bert_encoder import BertEncoder
from .decoder import Decoder
from .generator import Generator


class ChatModel(nn.Module):
    def __init__(self, config) -> None:
        super(ChatModel, self).__init__()
        self.config = config
        self.encoder = BertEncoder.from_pretrained(self.config.bert_model_name).eval()
        self.encoder.freeze()
        self.decoder = Decoder(
            config.vocab_size,
            config.n_layers,
            config.head,
            config.model_dim,
            config.max_len,
            config.ff_dim,
            config.dropout_rate,
        )

        self.generator = Generator(self.config.model_dim, self.config.vocab_size)

        for param in self.decoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
        for param in self.generator.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


    def forward(self,source, source_mask, target, target_mask):
        x = self.encode(source, source_mask)
        x = self.decode(x, source_mask, target, target_mask) # x is the memory
        x = self.generate(x)
        return x
        

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def encode(self, source, attention_mask):
        return self.encoder(source, attention_mask=attention_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(target, memory, source_mask, target_mask)

    def generate(self, x):
        return self.generator(x)

    def load(self, obj):
        self.load_state_dict(obj)
