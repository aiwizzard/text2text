import torch
from transformers.models.bert.tokenization_bert import BertTokenizer


class Tokenizer(BertTokenizer):
    def convert(self, x):
        return self.convert_tokens_to_ids(self.tokenize(x[:512]))

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.view(-1).tolist()
        s = ''.join([self.ids_to_tokens[x] for x in token_ids])
        return s
