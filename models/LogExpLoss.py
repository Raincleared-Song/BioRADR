import torch
import torch.nn as nn
from config import ConfigDenoise as Config


determine = torch.use_deterministic_algorithms if 'use_deterministic_algorithms' in dir(torch) \
        else torch.set_deterministic


class LogExpLoss(nn.Module):
    def __init__(self, margin_pos=5.0, margin_neg=5.0, gamma=2.0):
        super().__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.gamma = gamma

    def forward(self, scores, labels):
        """
        LogExp Loss Used in CR-CNN paper
        :param scores: float tensor, shape [sample_number, candidate_number of each sample]
        :param labels: long tensor, shape [sample_number], where each value is in range [0, candidate_number)
        :return: loss value
        """
        sample_number, candidate_number = scores.shape
        assert labels.shape == torch.Size([sample_number])

        pos_mask = torch.zeros_like(scores).to(scores)
        determine(False)
        pos_mask[torch.arange(0, sample_number), labels] = 1.0
        determine(True)
        neg_mask = 1.0 - pos_mask

        loss1 = torch.sum(torch.log(1 + torch.exp(self.gamma * (self.margin_pos - scores))) * pos_mask)
        loss2 = torch.sum(torch.log(1 + torch.exp(self.gamma * (self.margin_neg + scores))) * neg_mask)

        return loss1 + Config.negative_lambda * loss2
