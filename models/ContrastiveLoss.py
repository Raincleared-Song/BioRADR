import torch
import torch.nn as nn
from config import ConfigDenoise as Config


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_mrl = Config.loss_func.endswith('mrl')
        self.inner_loss = nn.MarginRankingLoss() if self.use_mrl else nn.SoftMarginLoss()

    def forward(self, scores, labels):
        """
        naive version contrastive loss
        1 - "pos is ahead of neg", -1 - "neg is ahead of pos"
        :param scores: float tensor, shape [sample_number, 2]
        :param labels: long tensor, shape [sample_number], where each value is in [1, -1]
        :return: loss value
        """
        sample_number, candidate_number = scores.shape
        assert labels.shape == torch.Size([sample_number]) and candidate_number == 2
        assert torch.sum(torch.logical_and(labels != -1, labels != 1)) == 0

        if self.use_mrl:
            return self.inner_loss(scores[:, 0], scores[:, 1], labels)
        else:
            return self.inner_loss(scores[:, 0] - scores[:, 1], labels)
