import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ConfigDenoise as Config


determine = torch.use_deterministic_algorithms if 'use_deterministic_algorithms' in dir(torch) \
        else torch.set_deterministic


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
        determine(False)
        p_mask[torch.arange(0, sample_number), labels, 1] = 1.0
        determine(True)
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

        return loss1 + Config.negative_lambda * loss2


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


class BalancedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = multilabel_categorical_crossentropy(labels, logits)
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = torch.zeros_like(logits[..., :1])
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output[:, 1:].sum(1) == 0.).to(logits)

        return output
