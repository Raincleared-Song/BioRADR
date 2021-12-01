import torch
import torch.nn as nn
from config import ConfigDenoise as Config


class ContrastiveLoss(nn.Module):
    def __init__(self):
        self.inner_loss = nn.MarginRankingLoss()
        super().__init__()

    def forward(self, scores, labels):
        """
        naive version contrastive loss
        0 - similar, -1 - dissimilar (pos is ahead), 1 - dissimilar (neg is ahead)
        :param scores: float tensor, shape [sample_number, 2]
        :param labels: long tensor, shape [sample_number], where each value is in [0, 1, -1]
        :return: loss value
        """
        sample_number, candidate_number = scores.shape
        assert labels.shape == torch.Size([sample_number]) and candidate_number == 2
        assert torch.sum(torch.logical_and(torch.logical_and(labels != 0, labels != 1), labels != -1)) == 0
        scores = scores[:, 0] - scores[:, 1]

        # loss0 = torch.sum(torch.square(scores * (labels == 0)))
        # loss1 = torch.sum(labels * torch.square(scores))

        loss0 = torch.sum(torch.abs(scores * (labels == 0)))
        loss1 = torch.sum(labels * scores)

        return loss0 * Config.similar_loss_lambda + loss1
