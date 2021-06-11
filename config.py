import torch

vocab_size = 30000
max_len = 150
batch_size = 64
model_dim = 512
ff_dim = 2048
head = 8
n_layers = 6
dropout_rate = 0.1
epochs = 3

lr = 1e-3
betas = (0.9, 0.98)

warmup = 4000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model_name = 'bert-base-uncased'

train_data = '.data/train_data.json'
data_dir = '.data'
fn = 'trained_model'