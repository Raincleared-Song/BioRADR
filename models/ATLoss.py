import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryATLoss(nn.Module):
    def __init__(self, th_logit=0.0, minus_mask=-1e30):
        super().__init__()
        self.th_logit = th_logit
        self.minus_mask = minus_mask

    def forward(self, scores, labels):
        """
        Binary Adaptive Thresholding Loss, where TH class logits is self.th_logit
        :param scores: float tensor, shape [sample_number, candidate_number of each sample]
        :param labels: long tensor, shape [sample_number], where each value is in range [0, candidate_number)
        :return: loss value
        """
        sample_number, candidate_number = scores.shape
        assert labels.shape == torch.Size([sample_number])

        th_scores = torch.zeros_like(scores).to(scores)
        th_scores[:, :] = self.th_logit
        # (sample, candidate, 2), dim(, , 0) is th_score
        scores = torch.stack((th_scores, scores), dim=2)

        p_mask = torch.zeros_like(scores).to(scores)
        p_mask[torch.arange(0, sample_number), labels, 1] = 1.0
        n_mask = 1 - p_mask
        n_mask[:, :, 0] = 0.0

        th_mask = torch.zeros_like(scores).to(scores)
        th_mask[:, :, 0] = 1.0

        # positive loss
        logit1 = scores + n_mask * self.minus_mask
        loss1 = -(F.log_softmax(logit1, dim=2) * p_mask).sum()

        # negative loss
        logit2 = scores + p_mask * self.minus_mask
        loss2 = -(F.log_softmax(logit2, dim=2) * th_mask).sum()

        return loss1 + loss2
