import json
import torch

import config as config
from model.model import ChatModel
from utils import subsequent_mask


def evaluate(config, query, model, word_map):
    model.eval()
    # tokenizer.encode
    rev_word_map = {k: v for v, k in word_map.items()}
    start_token = word_map['<start>']
    ids = [word_map.get(word, word_map['<unk>']) for word in query.split()]
    print(f"ids: {ids}")
    src = torch.tensor(ids, dtype=torch.long, device=config.device).view(1, -1)
    print(f"src: {src}")
    src_mask = torch.ones(src.size(), dtype=torch.long, device=config.device)
    print(f"src_mask: {src_mask}")
    mem = model.encode(src, src_mask)
    print(f"mem: {mem}")
    words = torch.ones(1, 1).fill_(start_token).long().to(config.device)
    for i in range(config.max_len -1):
        target_mask = subsequent_mask(words.size(1)).to(config.device)
        out = model.decode(words, mem, src_mask, target_mask)
        print(f"out: {out}")
        prob = model.generate(out[:, -1])
        _, candidate = prob.topk(5, dim=1)
        next_word = candidate[0, 0]
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.ones(1, 1).type_as(words).fill_(next_word).long()], dim=1)
        
    words = words.view(-1).detach().cpu().numpy().tolist()[1:]   
        # tokenizer.decode
    sen_index = [w for w in words if w not in {word_map['<start>']}]
    print(len(sen_index))
    sentence = ' '.join([rev_word_map[sen_index[k]] for k in range(len(sen_index))])

    return sentence
def main(config):
    with open('.data/wordmap.json', 'r') as j:
        word_map = json.load(j)

    state_dict = torch.load(f"{config.data_dir}/trained_model.pth", map_location=config.device)

    model = ChatModel(config).to(config.device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    while True:
        query = input('You>')
        if query == 'q':
            break
        text = evaluate(config, query, model, word_map)
        print(f"Arthur> {text}")

if __name__ == '__main__':
    main(config)