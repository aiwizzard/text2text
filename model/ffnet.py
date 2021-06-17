import torch.nn as nn

class FFNet(nn.Module):
    def __init__(self, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(FFNet, self).__init__()
        self.dense1 = nn.Linear(model_dim, ff_dim)
        self.dense2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x
