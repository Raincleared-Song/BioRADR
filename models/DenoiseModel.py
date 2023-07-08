import torch
import torch.nn as nn
from .long_input import process_long_input
from config import ConfigDenoise as Config
from transformers import AutoModel
from opendelta import LoraModel
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

        self.bert = AutoModel.from_pretrained(Config.bert_path)

        if Config.use_group_bilinear:
            self.bilinear = nn.Linear(self.bert_hidden * self.block_size, self.rep_hidden)
        else:
            self.bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.rep_hidden)
        self.linear_out = nn.Linear(self.rep_hidden, 1)

        self.loss = loss_map[Config.loss_func]()

        if Config.model_type == 'llama':
            # use LoRA
            delta_model = LoraModel(
                backbone_model=self, modified_modules=["q_proj", "v_proj"],
            )
            delta_model.freeze_module(exclude=["deltas", "bilinear", "linear_out"], set_state_dict=True)
            delta_model.log()

    def forward(self, data, mode: str, eval_res: dict = None):
        if mode != 'test':
            return self.forward_train(data, mode, eval_res)
        else:
            return self.forward_test(data, mode)

    def forward_test(self, data, mode: str):
        assert mode == 'test'

        documents = data['documents']
        attn_mask = data['attn_mask']
        word_pos = data['word_pos']
        head_ids = data['head_ids']
        tail_ids = data['tail_ids']

        batch_size = documents.shape[0]
        entity_lim = word_pos.shape[1]
        mention_lim = word_pos.shape[2]

        embed_docu = process_long_input(self.bert, documents, attn_mask, Config)
        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim, mention_lim)
        entity_rep = embed_docu[indices, word_pos]
        entity_rep = torch.max(entity_rep, dim=2)[0]

        pair_num = head_ids.shape[1]
        indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, pair_num)

        head_rep = entity_rep[indices, head_ids]
        tail_rep = entity_rep[indices, tail_ids]

        if Config.use_group_bilinear:
            head_rep = head_rep.view(batch_size, pair_num, self.bert_hidden // self.block_size, self.block_size)
            tail_rep = tail_rep.view(batch_size, pair_num, self.bert_hidden // self.block_size, self.block_size)
            rel_rep = (head_rep.unsqueeze(4) * tail_rep.unsqueeze(3)).view(batch_size, pair_num,
                                                                           self.bert_hidden * self.block_size)
            rel_rep = self.bilinear(rel_rep)
        else:
            rel_rep = self.bilinear(head_rep, tail_rep)

        score = self.linear_out(rel_rep).squeeze(2)
        return {'score': score, 'loss': 0, 'titles': data['titles']}

    def forward_train(self, data, mode: str, eval_res: dict = None):
        assert mode != 'test'
        if eval_res is None:
            eval_res = {'RD': {'correct_num': 0, 'instance_num': 0}, 'RD_X': {'correct_num': 0, 'instance_num': 0}}

        document1, document2 = data['document1'], data['document2']
        positions1, positions2 = data['positions1'], data['positions2']
        attn_mask1, attn_mask2 = data['attn_mask1'], data['attn_mask2']

        # relation detection
        # intra-document
        rd_head_ids1, rd_head_ids2 = data['rd_head_ids1'], data['rd_head_ids2']
        rd_tail_ids1, rd_tail_ids2 = data['rd_tail_ids1'], data['rd_tail_ids2']
        rd_label1, rd_label2 = data['rd_label1'], data['rd_label2']

        # inter-document
        rd_head_ids1_x, rd_head_ids2_x = data['rd_head_ids1_x'], data['rd_head_ids2_x']
        rd_tail_ids1_x, rd_tail_ids2_x = data['rd_tail_ids1_x'], data['rd_tail_ids2_x']
        rd_label_x = data['rd_label_x']

        # constants
        batch_size = document1.shape[0]
        entity_lim1, entity_lim2, mention_lim = positions1.shape[1], positions2.shape[1], positions1.shape[2]

        doc_hidden1 = process_long_input(self.bert, document1, attn_mask1, Config)
        doc_hidden2 = process_long_input(self.bert, document2, attn_mask2, Config)

        indices1 = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim1, mention_lim)
        rep_ent1 = doc_hidden1[indices1, positions1].view(batch_size, entity_lim1, mention_lim, self.bert_hidden)
        indices2 = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim2, mention_lim)
        rep_ent2 = doc_hidden2[indices2, positions2].view(batch_size, entity_lim2, mention_lim, self.bert_hidden)

        rep_ent1 = torch.max(rep_ent1, dim=2)[0]
        rep_ent2 = torch.max(rep_ent2, dim=2)[0]

        total_loss = 0

        loss1, eval_res = self.forward_rd(rep_ent1, rd_head_ids1, rd_tail_ids1, rd_label1, eval_res)
        loss2, eval_res = self.forward_rd(rep_ent2, rd_head_ids2, rd_tail_ids2, rd_label2, eval_res)
        total_loss += loss1 + loss2

        if Config.use_inter:
            loss, eval_res = self.forward_rd_x(rep_ent1, rep_ent2, rd_head_ids1_x, rd_head_ids2_x,
                                               rd_tail_ids1_x, rd_tail_ids2_x, rd_label_x, eval_res)
            total_loss += loss

        return {'loss': total_loss, 'eval_res': eval_res}

    def forward_rd(self, ent_rep, head_ids, tail_ids, label, eval_res):
        batch_size, sample_size, negative_size = head_ids.shape  # (1, 16, 16)

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, sample_size, negative_size)
        head_rep = ent_rep[indices, head_ids].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep = ent_rep[indices, tail_ids].view(batch_size, sample_size, negative_size, self.bert_hidden)

        if Config.use_group_bilinear:
            head_rep = head_rep.view(batch_size, sample_size, negative_size,
                                     self.bert_hidden // self.block_size, self.block_size)
            tail_rep = tail_rep.view(batch_size, sample_size, negative_size,
                                     self.bert_hidden // self.block_size, self.block_size)
            rel_rep = (head_rep.unsqueeze(5) * tail_rep.unsqueeze(4)).view(batch_size, sample_size, negative_size,
                                                                           self.bert_hidden * self.block_size)
            rel_rep = self.bilinear(rel_rep)
        else:
            rel_rep = self.bilinear(head_rep, tail_rep)

        score = self.linear_out(rel_rep).squeeze(3).view(-1, negative_size)  # (16, 16)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD'] = eval_softmax(score, label, eval_res['RD'])

        return loss, eval_res

    def forward_rd_x(self, ent_rep1, ent_rep2, head1, head2, tail1, tail2, label, eval_res):
        batch_size, sample_size, negative_size = head1.shape  # (1, 16, 8)

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, sample_size, negative_size)
        head_rep1 = ent_rep1[indices, head1].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep1 = ent_rep1[indices, tail1].view(batch_size, sample_size, negative_size, self.bert_hidden)
        head_rep2 = ent_rep2[indices, head2].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep2 = ent_rep2[indices, tail2].view(batch_size, sample_size, negative_size, self.bert_hidden)

        if Config.use_group_bilinear:
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
        else:
            rel_rep1 = self.bilinear(head_rep1, tail_rep1)
            rel_rep2 = self.bilinear(head_rep2, tail_rep2)

        score1 = self.linear_out(rel_rep1).squeeze(3)  # (1, 16, 8)
        score2 = self.linear_out(rel_rep2).squeeze(3)  # (1, 16, 8)

        score = torch.cat((score1, score2), dim=2).view(-1, negative_size << 1)  # (16, 16)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD_X'] = eval_softmax(score, label, eval_res['RD_X'])

        return loss, eval_res
