import torch.nn as nn

class FFNet(nn.Module):
    def __init__(self, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(FFNet, self).__init__()
        self.layer1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.layer2 = nn.Conv1d(ff_dim, model_dim, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.layer1(out.transpose(1, 2))
        out = self.relu(out)
        out = self.layer2(out)
        out = self.dropout(out.transpose(1, 2))
       
        return x + out
