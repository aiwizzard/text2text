import torch
import numpy as np

import config as config

def create_masks(source, target, target_y, pad=0):
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        mask = mask.unsqueeze(0)
        return mask

    source_mask = (source != pad).to(config.device)
    source_mask = source_mask.unsqueeze(1)

    target_mask = target != pad
    target_mask = target_mask.unsqueeze(1) 
    target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(target_mask.data)
    target_mask = target_mask.unsqueeze(1)

    target_y_mask = target_y != pad 

    return source_mask, target_mask, target_y_mask
