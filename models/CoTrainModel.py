import torch
import torch.nn as nn
from .long_input import process_long_input
from config import ConfigDenoise, ConfigFineTune
from transformers import BertModel
from utils import eval_softmax
from .ATLoss import BinaryATLoss
from .ContrastiveLoss import ContrastiveLoss
from .LogExpLoss import LogExpLoss
from utils import eval_multi_label


loss_map = {
    'cross_entropy': nn.CrossEntropyLoss,
    'adaptive_threshold': BinaryATLoss,
    'contrastive_mrl': ContrastiveLoss,
    'contrastive_sml': ContrastiveLoss,
    'log_exp': LogExpLoss,
}


class CoTrainModel(nn.Module):
    def __init__(self):
        super(CoTrainModel, self).__init__()

        self.rep_hidden = ConfigDenoise.hidden_size
        self.relation_num = ConfigDenoise.relation_num
        self.bert_hidden = ConfigDenoise.bert_hidden
        self.block_size = ConfigDenoise.block_size

        self.bert = BertModel.from_pretrained(ConfigDenoise.bert_path)
        if ConfigDenoise.use_group_bilinear:
            self.bilinear = nn.Linear(self.bert_hidden * self.block_size, self.rep_hidden)
        else:
            self.bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.rep_hidden)
        assert self.relation_num == 2
        self.cur_task = 'denoise'
        self.linear_rank = nn.Linear(self.rep_hidden, 1)
        self.linear_re = nn.Linear(self.rep_hidden, self.relation_num)

        self.loss = loss_map[ConfigDenoise.loss_func]()

    def forward(self, data, mode: str, eval_res: dict = None):
        if self.cur_task == 'denoise':
            return self.forward_rank(data, mode, eval_res)
        elif self.cur_task == 'finetune':
            return self.forward_re(data, mode, eval_res)
        else:
            raise NotImplementedError('self.task not in ["denoise", "finetune"]')

    def forward_rank(self, data, mode: str, eval_res: dict = None):
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

        embed_docu = process_long_input(self.bert, documents, attn_mask, ConfigDenoise)
        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim, mention_lim)
        entity_rep = embed_docu[indices, word_pos]
        entity_rep = torch.max(entity_rep, dim=2)[0]

        pair_num = head_ids.shape[1]
        indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, pair_num)

        head_rep = entity_rep[indices, head_ids]
        tail_rep = entity_rep[indices, tail_ids]

        if ConfigDenoise.use_group_bilinear:
            head_rep = head_rep.view(batch_size, pair_num, self.bert_hidden // self.block_size, self.block_size)
            tail_rep = tail_rep.view(batch_size, pair_num, self.bert_hidden // self.block_size, self.block_size)
            rel_rep = (head_rep.unsqueeze(4) * tail_rep.unsqueeze(3)).view(batch_size, pair_num,
                                                                           self.bert_hidden * self.block_size)
            rel_rep = self.bilinear(rel_rep)
        else:
            rel_rep = self.bilinear(head_rep, tail_rep)

        score = self.linear_rank(rel_rep).squeeze(2)
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

        doc_hidden1 = process_long_input(self.bert, document1, attn_mask1, ConfigDenoise)
        doc_hidden2 = process_long_input(self.bert, document2, attn_mask2, ConfigDenoise)

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

        if ConfigDenoise.use_inter:
            loss, eval_res = self.forward_rd_x(rep_ent1, rep_ent2, rd_head_ids1_x, rd_head_ids2_x,
                                               rd_tail_ids1_x, rd_tail_ids2_x, rd_label_x, eval_res)
            total_loss += loss

        return {'loss': total_loss, 'eval_res': eval_res}

    def forward_rd(self, ent_rep, head_ids, tail_ids, label, eval_res):
        batch_size, sample_size, negative_size = head_ids.shape  # (1, 16, 16)

        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, sample_size, negative_size)
        head_rep = ent_rep[indices, head_ids].view(batch_size, sample_size, negative_size, self.bert_hidden)
        tail_rep = ent_rep[indices, tail_ids].view(batch_size, sample_size, negative_size, self.bert_hidden)

        if ConfigDenoise.use_group_bilinear:
            head_rep = head_rep.view(batch_size, sample_size, negative_size,
                                     self.bert_hidden // self.block_size, self.block_size)
            tail_rep = tail_rep.view(batch_size, sample_size, negative_size,
                                     self.bert_hidden // self.block_size, self.block_size)
            rel_rep = (head_rep.unsqueeze(5) * tail_rep.unsqueeze(4)).view(batch_size, sample_size, negative_size,
                                                                           self.bert_hidden * self.block_size)
            rel_rep = self.bilinear(rel_rep)
        else:
            rel_rep = self.bilinear(head_rep, tail_rep)

        score = self.linear_rank(rel_rep).squeeze(3).view(-1, negative_size)  # (16, 16)
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

        if ConfigDenoise.use_group_bilinear:
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

        score1 = self.linear_rank(rel_rep1).squeeze(3)  # (1, 16, 8)
        score2 = self.linear_rank(rel_rep2).squeeze(3)  # (1, 16, 8)

        score = torch.cat((score1, score2), dim=2).view(-1, negative_size << 1)  # (16, 16)
        label = label.view(-1)
        loss = self.loss(score, label)
        eval_res['RD_X'] = eval_softmax(score, label, eval_res['RD_X'])

        return loss, eval_res

    def forward_re(self, data, mode: str, eval_res: dict = None):
        documents = data['documents']
        head_pos = data['head_pos']
        tail_pos = data['tail_pos']
        labels = data['labels']
        label_mask = data['label_mask']
        types = data['types']

        cur_batch_size = documents.shape[0]
        sample_lim = head_pos.shape[1]  # sample_limit
        mention_limit = ConfigFineTune.mention_padding

        if ConfigFineTune.token_padding > 512:
            embed_docu = process_long_input(self.bert, documents, data['attn_mask'], ConfigFineTune)
        else:
            embed_docu = self.bert(documents, attention_mask=data['attn_mask'])[0]

        head_rep = embed_docu[[[[i] * mention_limit] * sample_lim for i in range(cur_batch_size)], head_pos]
        tail_rep = embed_docu[[[[i] * mention_limit] * sample_lim for i in range(cur_batch_size)], tail_pos]

        # max pooling
        if ConfigFineTune.use_logsumexp:
            head_rep = torch.logsumexp(head_rep.view(cur_batch_size, sample_lim,
                                                     mention_limit, self.bert_hidden), dim=2)
            tail_rep = torch.logsumexp(tail_rep.view(cur_batch_size, sample_lim,
                                                     mention_limit, self.bert_hidden), dim=2)
        else:
            head_rep = torch.max(head_rep.view(cur_batch_size, sample_lim, mention_limit, self.bert_hidden), dim=2)[0]
            tail_rep = torch.max(tail_rep.view(cur_batch_size, sample_lim, mention_limit, self.bert_hidden), dim=2)[0]

        if ConfigFineTune.use_group_bilinear:
            head_rep = head_rep.view(cur_batch_size, sample_lim, self.bert_hidden // self.block_size, self.block_size)
            tail_rep = tail_rep.view(cur_batch_size, sample_lim, self.bert_hidden // self.block_size, self.block_size)
            relation_rep = (head_rep.unsqueeze(4) * tail_rep.unsqueeze(3)).view(cur_batch_size, sample_lim,
                                                                                self.bert_hidden * self.block_size)
            relation_rep = self.bilinear(relation_rep)
        else:
            relation_rep = self.bilinear(head_rep, tail_rep)  # (batch, sample_lim, rel_embed)

        if ConfigFineTune.use_entity_type:
            type_tensor = []
            for batch_t in types:
                for sample_t in batch_t:
                    type_tensor.append(self.type_to_embed[sample_t[0]])
                    type_tensor.append(self.type_to_embed[sample_t[1]])
            type_tensor = torch.cat(type_tensor).view((cur_batch_size, sample_lim, self.type_embed_size * 2))
            relation_rep = torch.cat((relation_rep, type_tensor), dim=2)

        predict_out = self.linear_re(relation_rep)

        if mode != 'test':
            labels = labels.view(-1, labels.shape[2])
            label_mask = label_mask.view(-1)

            predict_out = predict_out.view(cur_batch_size * sample_lim, self.relation_num)

            loss = self.loss(predict_out, labels.float())
            loss = torch.sum(loss * label_mask.unsqueeze(1)) / (torch.sum(label_mask) * self.relation_num)

            eval_res = eval_multi_label(predict_out, labels, label_mask, eval_res)
            if mode == 'valid':
                eval_res['instance_num'] = ConfigFineTune.valid_instance_cnt
            return {'loss': loss, 'eval_res': eval_res, 'titles': data['titles']}
        else:
            scores = None
            pos_id, na_id = ConfigFineTune.label2id['Pos'], ConfigFineTune.label2id['NA']
            if ConfigFineTune.output_score_type == 'sigmoid':
                scores = torch.sigmoid(predict_out[:, :, pos_id])
            elif ConfigFineTune.output_score_type == 'softmax':
                scores = torch.softmax(predict_out, dim=2)[:, :, pos_id]
            elif ConfigFineTune.output_score_type == 'pos':
                scores = predict_out[:, :, pos_id]
            elif ConfigFineTune.output_score_type == 'diff':
                scores = predict_out[:, :, pos_id] - predict_out[:, :, na_id]
            elif ConfigFineTune.output_score_type == 'sig_diff':
                scores = torch.sigmoid(predict_out[:, :, pos_id] - predict_out[:, :, na_id])
            predict_out = torch.max(predict_out, dim=2)[1]
            return {'predict': predict_out.cpu().tolist(),  # [b, 90]
                    'pair_ids': data['pair_ids'],  # [b, 90, 2]
                    'titles': data['titles'],
                    'score': scores}  # [b]
