import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, model_dim, vocab):
        super(Generator, self).__init__()
        self.dense = nn.Linear(model_dim, vocab)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.dense(x)
        x = F.log_softmax(x, dim=-1)
        return x