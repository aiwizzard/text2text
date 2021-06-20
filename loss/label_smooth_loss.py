import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.size = size
        self.confidence = 1.0 - smoothing
        self.smooth = smoothing
        self.padding = 0
        self.true_dist = None

    def forward(self, prediction, target):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        tgt_dist = prediction.clone()
        tgt_dist.fill_(self.smooth / (self.size - 2))
        tgt_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        tgt_dist[:, self.padding] = 0
        mask = target == self.padding
        if mask.dim() > 0:
            tgt_dist.index_fill_(0, mask.squeeze(-1), 0.0)
        self.true_dist = tgt_dist
        return self.criterion(prediction, tgt_dist.clone())
        