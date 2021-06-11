import torch

train_data = '.data/train_data.json'
batch_size = 64
model_dim = 512
head = 8
n_layers = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 3

lr = 1e-3
betas = (0.9, 0.98)

warmup = 4000