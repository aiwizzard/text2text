import json
import torch
from torch._C import device
with open('.data/wordmap.json', 'r') as j:
    word_map = json.load(j)

vocab_size = len(word_map)
max_len = 27
batch_size = 64
model_dim = 768
ff_dim = 2048
head = 8
n_layers = 6
dropout_rate = 0.1
epochs = 3

seed = 116

lr = 1e-5
betas = (0.9, 0.98)
max_grad_norm = 1.0

warmup = 128000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

bert_model_name = 'bert-base-uncased'

train_data = '.data/train_data.json'
data_dir = '.data'
fn = 'trained_model'
