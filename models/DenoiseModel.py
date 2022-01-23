import torch
import torch.nn as nn
from .long_input import process_long_input
from config import ConfigDenoise as Config
from transformers import BertModel
from utils import eval_softmax
from .ATLoss import BinaryATLoss
from .ContrastiveLoss import ContrastiveLoss
from .LogExpLoss import LogExpLoss


loss_map = {
    'cross_entropy': nn.CrossEntropyLoss,
    'adaptive_threshold': BinaryATLoss,
    'contrastive_mrl': ContrastiveLoss,
    'contrastive_sml': ContrastiveLoss,
    'log_exp': LogExpLoss,
}


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()

        self.rep_hidden = Config.hidden_size
        self.relation_num = Config.relation_num
        self.bert_hidden = Config.bert_hidden
        self.block_size = Config.block_size

        self.bert = BertModel.from_pretrained(Config.bert_path)
        self.bilinear = nn.Linear(self.bert_hidden * self.block_size, self.rep_hidden)
        self.linear_out = nn.Linear(self.rep_hidden, 1)

        self.loss = loss_map[Config.loss_func]()

    def forward(self, data, mode: str, eval_res: dict = None):
        if mode != 'test':
            return self.forward_train(data, mode, eval_res)
        else:
            return self.forward_test(data, mode, eval_res)

    def forward_test(self, data, mode: str, _):
        assert mode == 'test'

        documents = data['documents']
        attn_mask = data['attn_mask']
        # batch_size * sample_limit * mention_limit
        head_poses = data['head_poses']
        tail_poses = data['tail_poses']

        batch_size, pair_num, mention_lim = head_poses.shape

        # batch_size * sent_pad * 768
        embed_docu = process_long_input(self.bert, documents, attn_mask, Config)
        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, pair_num, mention_lim)
        head_men_rep = embed_docu[indices, head_poses]
        tail_men_rep = embed_docu[indices, tail_poses]
        head_rep = torch.max(head_men_rep, dim=2)[0]
        tail_rep = torch.max(tail_men_rep, dim=2)[0]

        head_rep = head_rep.view(batch_size, pair_num,
                                 self.bert_hidden // self.block_size, self.block_size)
        tail_rep = tail_rep.view(batch_size, pair_num,
                                 self.bert_hidden // self.block_size, self.block_size)
        rel_rep = (head_rep.unsqueeze(4) * tail_rep.unsqueeze(3)).view(batch_size, pair_num,
                                                                       self.bert_hidden * self.block_size)
        rel_rep = self.bilinear(rel_rep)
        score = self.linear_out(rel_rep).squeeze(2)
        return {'score': score, 'loss': 0, 'titles': data['titles']}

    def forward_train(self, data, mode: str, eval_res: dict = None):
        assert mode != 'test'
        if eval_res is None:
            eval_res = {'RD': {'correct_num': 0, 'instance_num': 0}, 'RD_X': {'correct_num': 0, 'instance_num': 0}}

        document1, document2 = data['document1'], data['document2']
        attn_mask1, attn_mask2 = data['attn_mask1'], data['attn_mask2']

        # relation detection
        # intra-document
        rd_head_poses1, rd_tail_poses1 = data['rd_head_poses1'], data['rd_tail_poses1']
        rd_head_poses2, rd_tail_poses2 = data['rd_head_poses2'], data['rd_tail_poses2']
        rd_label1, rd_label2 = data['rd_label1'], data['rd_label2']

        # inter-document
        rd_head_poses1_x, rd_tail_poses1_x = data['rd_head_poses1_x'], data['rd_tail_poses1_x']
        rd_head_poses2_x, rd_tail_poses2_x = data['rd_head_poses2_x'], data['rd_tail_poses2_x']
        rd_label_x = data['rd_label_x']

        doc_hidden1 = process_long_input(self.bert, document1, attn_mask1, Config)
        doc_hidden2 = process_long_input(self.bert, document2, attn_mask2, Config)

        total_loss = 0

        loss1, eval_res = self.forward_rd(doc_hidden1, rd_head_poses1, rd_tail_poses1, rd_label1, eval_res)
        loss2, eval_res = self.forward_rd(doc_hidden2, rd_head_poses2, rd_tail_poses2, rd_label2, eval_res)
        total_loss += loss1 + loss2

        if Config.use_inter:
            loss, eval_res = self.forward_rd_x(doc_hidden1, doc_hidden2, rd_head_poses1_x, rd_tail_poses1_x,
                                               rd_head_poses2_x, rd_tail_poses2_x, rd_label_x, eval_res)
            total_loss += loss

        return {'loss': total_loss, 'eval_res': eval_res}

    def forward_rd(self, embed_doc, head_poses, tail_poses, label, eval_res):
        batch_size, sample_size, negative_size, mention_limit = head_poses.shape

        indices = torch.arange(0, batch_size).view(
            batch_size, 1, 1, 1).repeat(1, sample_size, negative_size, mention_limit)
        head_rep = torch.max(embed_doc[indices, head_poses].view(
            batch_size, sample_size, negative_size, mention_limit, self.bert_hidden), dim=3)[0]
        tail_rep = torch.max(embed_doc[indices, tail_poses].view(
            batch_size, sample_size, negative_size, mention_limit, self.bert_hidden), dim=3)[0]

        head_rep = head_rep.view(batch_size, sample_size, negative_size,
                                 self.bert_hidden // self.block_size, self.block_size)
        tail_rep = tail_rep.view(batch_size, sample_size, negative_size,
                                 self.bert_hidden // self.block_size, self.block_size)
        rel_rep = (head_rep.unsqueeze(5) * tail_rep.unsqueeze(4)).view(batch_size, sample_size, negative_size,
                                                                       self.bert_hidden * self.block_size)
        rel_rep = self.bilinear(rel_rep)

        score = self.linear_out(rel_rep).squeeze(3).view(-1, negative_size)  # (16, 16)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD'] = eval_softmax(score, label, eval_res['RD'])

        return loss, eval_res

    def forward_rd_x(self, embed_doc1, embed_doc2, head1, tail1, head2, tail2, label, eval_res):
        batch_size, sample_size, negative_size, mention_limit = head1.shape

        indices = torch.arange(0, batch_size).view(
            batch_size, 1, 1, 1).repeat(1, sample_size, negative_size, mention_limit)
        head_rep1 = torch.max(embed_doc1[indices, head1].view(
            batch_size, sample_size, negative_size, mention_limit, self.bert_hidden), dim=3)[0]
        tail_rep1 = torch.max(embed_doc1[indices, tail1].view(
            batch_size, sample_size, negative_size, mention_limit, self.bert_hidden), dim=3)[0]
        head_rep2 = torch.max(embed_doc2[indices, head2].view(
            batch_size, sample_size, negative_size, mention_limit, self.bert_hidden), dim=3)[0]
        tail_rep2 = torch.max(embed_doc2[indices, tail2].view(
            batch_size, sample_size, negative_size, mention_limit, self.bert_hidden), dim=3)[0]

        head_rep1 = head_rep1.view(batch_size, sample_size, negative_size,
                                   self.bert_hidden // self.block_size, self.block_size)
        tail_rep1 = tail_rep1.view(batch_size, sample_size, negative_size,
                                   self.bert_hidden // self.block_size, self.block_size)
        rel_rep1 = (head_rep1.unsqueeze(5) * tail_rep1.unsqueeze(4)).view(batch_size, sample_size, negative_size,
                                                                          self.bert_hidden * self.block_size)
        rel_rep1 = self.bilinear(rel_rep1)

        head_rep2 = head_rep2.view(batch_size, sample_size, negative_size,
                                   self.bert_hidden // self.block_size, self.block_size)
        tail_rep2 = tail_rep2.view(batch_size, sample_size, negative_size,
                                   self.bert_hidden // self.block_size, self.block_size)
        rel_rep2 = (head_rep2.unsqueeze(5) * tail_rep2.unsqueeze(4)).view(batch_size, sample_size, negative_size,
                                                                          self.bert_hidden * self.block_size)
        rel_rep2 = self.bilinear(rel_rep2)

        score1 = self.linear_out(rel_rep1).squeeze(3)  # (1, 16, 8)
        score2 = self.linear_out(rel_rep2).squeeze(3)  # (1, 16, 8)

        score = torch.cat((score1, score2), dim=2).view(-1, negative_size << 1)  # (16, 16)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD_X'] = eval_softmax(score, label, eval_res['RD_X'])

        return loss, eval_res
