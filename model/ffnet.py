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
        x = self.layer_norm(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x
