import torch
import torch.nn as nn
from config import ConfigPretrain as Config
from transformers import BertModel
from utils import eval_softmax


class PretrainModel(nn.Module):
    def __init__(self):
        super(PretrainModel, self).__init__()

        self.rep_hidden = Config.hidden_size
        self.relation_num = Config.relation_num
        self.bert_hidden = Config.bert_hidden

        self.bert = BertModel.from_pretrained(Config.bert_path)
        self.bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.rep_hidden)
        self.mem_bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, 1)
        self.score_linear = nn.Linear(self.rep_hidden, 1)
        self.rfa_linear = nn.Linear(self.rep_hidden, 1)

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.tasks = Config.pretrain_tasks.split('|')

    def save(self, path: str):
        self.bert.save_pretrained(path)

    def forward_mem(self, query, candidate, label, doc_hidden, eval_res):
        batch_size, sample_size, negative_size = candidate.shape

        indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, sample_size)
        query_rep = doc_hidden[indices, query].view(batch_size, sample_size, self.bert_hidden)

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, sample_size, negative_size)
        candidate_rep = doc_hidden[indices, candidate].view(batch_size, sample_size, negative_size, self.bert_hidden)

        score = self.mem_bilinear(query_rep.unsqueeze(2).repeat(1, 1, negative_size, 1), candidate_rep).squeeze(3).\
            view(batch_size * sample_size, negative_size)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['MEM'] = eval_softmax(score, label, eval_res['MEM'])

        return loss, eval_res

    def forward_mem_x(self, ent_rep1, ent_rep2, query, candidate, label, eval_res):
        batch_size = candidate.shape[0]
        negative_size = candidate.shape[1]

        indices = torch.arange(0, batch_size)
        query_rep = ent_rep1[indices, query].view(batch_size, self.bert_hidden)

        indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, negative_size)
        candidate_rep = ent_rep2[indices, candidate].view(batch_size, negative_size, self.bert_hidden)

        score = self.mem_bilinear(query_rep.unsqueeze(1).repeat(1, negative_size, 1), candidate_rep).squeeze(2).\
            view(batch_size, negative_size)
        loss = self.loss(score, label)
        eval_res['MEM_X'] = eval_softmax(score, label, eval_res['MEM_X'])

        return loss, eval_res

    def forward_rd(self, ent_rep, head_ids, tail_ids, label, eval_res):
        batch_size, sample_size, negative_size = head_ids.shape

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, sample_size, negative_size)
        head_rep = ent_rep[indices, head_ids].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep = ent_rep[indices, tail_ids].view(batch_size, sample_size, negative_size, self.bert_hidden)

        rel_rep = self.bilinear(head_rep, tail_rep)

        score = self.score_linear(rel_rep).squeeze(3).view(-1, negative_size)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD'] = eval_softmax(score, label, eval_res['RD'])

        return loss, eval_res

    def forward_rd_x(self, ent_rep1, ent_rep2, head1, head2, tail1, tail2, label, eval_res):
        batch_size, sample_size, negative_size = head1.shape

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, sample_size, negative_size)
        head_rep1 = ent_rep1[indices, head1].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep1 = ent_rep1[indices, tail1].view(batch_size, sample_size, negative_size, self.bert_hidden)
        head_rep2 = ent_rep2[indices, head2].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep2 = ent_rep2[indices, tail2].view(batch_size, sample_size, negative_size, self.bert_hidden)

        rel_rep1 = self.bilinear(head_rep1, tail_rep1)
        rel_rep2 = self.bilinear(head_rep2, tail_rep2)

        score1 = self.score_linear(rel_rep1).squeeze(3)
        score2 = self.score_linear(rel_rep2).squeeze(3)

        score = torch.cat((score1, score2), dim=2).view(-1, negative_size << 1)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD_X'] = eval_softmax(score, label, eval_res['RD_X'])

        return loss, eval_res

    def forward_rfa(self, ent_rep_q, ent_rep_c, head_q, tail_q, head_c, tail_c, label, eval_res, intra):
        """same for intra and inter task"""
        batch_size, sample_size = head_c.shape

        indices = torch.arange(0, batch_size)
        head_rep_q = ent_rep_q[indices, head_q].view(batch_size, self.bert_hidden)
        tail_rep_q = ent_rep_q[indices, tail_q].view(batch_size, self.bert_hidden)
        rel_rep_q = self.bilinear(head_rep_q, tail_rep_q)

        indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, sample_size)
        head_rep_c = ent_rep_c[indices, head_c].view(batch_size, sample_size, self.bert_hidden)
        tail_rep_c = ent_rep_c[indices, tail_c].view(batch_size, sample_size, self.bert_hidden)
        rel_rep_c = self.bilinear(head_rep_c, tail_rep_c)

        score = self.rfa_linear(torch.pow(rel_rep_q.unsqueeze(1) - rel_rep_c, 2)).squeeze(2)
        loss = self.loss(score, label)
        task = 'RFA' if intra else 'RFA_X'
        eval_res[task] = eval_softmax(score, label, eval_res[task])

        return loss, eval_res

    def forward(self, data, mode: str, eval_res: dict = None):
        assert mode != 'test'

        if eval_res is None:
            eval_res = {key: {'correct_num': 0, 'instance_num': 0} for key in self.tasks}

        document1, document2 = data['document1'], data['document2']
        positions1, positions2 = data['positions1'], data['positions2']
        attn_mask1, attn_mask2 = data['attn_mask1'], data['attn_mask2']

        # mention-entity matching
        # intra-document
        mem_query1, mem_query2 = data['mem_query1'], data['mem_query2']
        mem_candidate1, mem_candidate2 = data['mem_candidate1'], data['mem_candidate2']
        mem_label1, mem_label2 = data['mem_label1'], data['mem_label2']

        # inter-document
        mem_query1_x, mem_query2_x = data['mem_query1_x'], data['mem_query2_x']
        mem_candidate1_x, mem_candidate2_x = data['mem_candidate1_x'], data['mem_candidate2_x']
        mem_label1_x, mem_label2_x = data['mem_label1_x'], data['mem_label2_x']

        # relation detection
        # intra-document
        rd_head_ids1, rd_head_ids2 = data['rd_head_ids1'], data['rd_head_ids2']
        rd_tail_ids1, rd_tail_ids2 = data['rd_tail_ids1'], data['rd_tail_ids2']
        rd_label1, rd_label2 = data['rd_label1'], data['rd_label2']

        # inter-document
        rd_head_ids1_x, rd_head_ids2_x = data['rd_head_ids1_x'], data['rd_head_ids2_x']
        rd_tail_ids1_x, rd_tail_ids2_x = data['rd_tail_ids1_x'], data['rd_tail_ids2_x']
        rd_label_x = data['rd_label_x']

        # relation fact alignment
        # intra-document
        rfa_query_head1, rfa_query_head2 = data['rfa_query_head1'], data['rfa_query_head2']
        rfa_query_tail1, rfa_query_tail2 = data['rfa_query_tail1'], data['rfa_query_tail2']
        rfa_candidate_head1, rfa_candidate_head2 = data['rfa_candidate_head1'], data['rfa_candidate_head2']
        rfa_candidate_tail1, rfa_candidate_tail2 = data['rfa_candidate_tail1'], data['rfa_candidate_tail2']
        rfa_label1, rfa_label2 = data['rfa_label1'], data['rfa_label2']

        # inter-document
        rfa_query_head1_x, rfa_query_head2_x = data['rfa_query_head1_x'], data['rfa_query_head2_x']
        rfa_query_tail1_x, rfa_query_tail2_x = data['rfa_query_tail1_x'], data['rfa_query_tail2_x']
        rfa_candidate_head1_x, rfa_candidate_head2_x = data['rfa_candidate_head1_x'], data['rfa_candidate_head2_x']
        rfa_candidate_tail1_x, rfa_candidate_tail2_x = data['rfa_candidate_tail1_x'], data['rfa_candidate_tail2_x']
        rfa_label12, rfa_label21 = data['rfa_label12'], data['rfa_label21']

        # constants
        batch_size = document1.shape[0]
        entity_lim, mention_lim = positions1.shape[1], positions1.shape[2]

        doc_hidden1 = self.bert(document1, attention_mask=attn_mask1)[0]
        doc_hidden2 = self.bert(document2, attention_mask=attn_mask2)[0]

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim, mention_lim)
        try:
            rep_ent1 = doc_hidden1[indices, positions1].view(batch_size, entity_lim, mention_lim, self.bert_hidden)
            rep_ent2 = doc_hidden2[indices, positions2].view(batch_size, entity_lim, mention_lim, self.bert_hidden)
        except IndexError as err:
            import numpy as np
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(key, np.shape(value))
            print(np.shape(rep_ent1), np.shape(rep_ent2))
            print(np.shape(indices), np.shape(positions1), np.shape(positions2))
            print(batch_size, entity_lim, mention_lim, self.bert_hidden)
            raise err
        rep_ent1 = torch.max(rep_ent1, dim=2)[0]
        rep_ent2 = torch.max(rep_ent2, dim=2)[0]

        total_loss = 0

        if 'MEM' in self.tasks:
            loss1, eval_res = self.forward_mem(mem_query1, mem_candidate1, mem_label1, doc_hidden1, eval_res)
            loss2, eval_res = self.forward_mem(mem_query2, mem_candidate2, mem_label2, doc_hidden2, eval_res)
            total_loss += loss1 + loss2

        if 'MEM_X' in self.tasks:
            loss1, eval_res = self.forward_mem_x(rep_ent1, rep_ent2, mem_query1_x, mem_candidate2_x,
                                                 mem_label2_x, eval_res)
            loss2, eval_res = self.forward_mem_x(rep_ent2, rep_ent1, mem_query2_x, mem_candidate1_x,
                                                 mem_label1_x, eval_res)
            total_loss += loss1 + loss2

        if 'RD' in self.tasks:
            loss1, eval_res = self.forward_rd(rep_ent1, rd_head_ids1, rd_tail_ids1, rd_label1, eval_res)
            loss2, eval_res = self.forward_rd(rep_ent2, rd_head_ids2, rd_tail_ids2, rd_label2, eval_res)
            total_loss += loss1 + loss2

        if 'RD_X' in self.tasks:
            loss, eval_res = self.forward_rd_x(rep_ent1, rep_ent2, rd_head_ids1_x, rd_head_ids2_x,
                                               rd_tail_ids1_x, rd_tail_ids2_x, rd_label_x, eval_res)
            total_loss += loss

        if 'RFA' in self.tasks:
            loss1, eval_res = self.forward_rfa(rep_ent1, rep_ent1, rfa_query_head1, rfa_query_tail1,
                                               rfa_candidate_head1, rfa_candidate_tail1,
                                               rfa_label1, eval_res, intra=True)
            loss2, eval_res = self.forward_rfa(rep_ent2, rep_ent2, rfa_query_head2, rfa_query_tail2,
                                               rfa_candidate_head2, rfa_candidate_tail2,
                                               rfa_label2, eval_res, intra=True)
            total_loss += loss1 + loss2

        if 'RFA_X' in self.tasks:
            loss1, eval_res = self.forward_rfa(rep_ent1, rep_ent2, rfa_query_head1_x, rfa_query_tail1_x,
                                               rfa_candidate_head2_x, rfa_candidate_tail2_x,
                                               rfa_label12, eval_res, intra=False)
            loss2, eval_res = self.forward_rfa(rep_ent2, rep_ent1, rfa_query_head2_x, rfa_query_tail2_x,
                                               rfa_candidate_head1_x, rfa_candidate_tail1_x,
                                               rfa_label21, eval_res, intra=False)
            total_loss += loss1 + loss2

        return {'loss': total_loss, 'eval_res': eval_res}
