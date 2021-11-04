import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .long_input import process_long_input
from config import ConfigFineTune as Config
from utils import eval_multi_label, time_tag
from torch.autograd import Variable


class FineTuneModel(nn.Module):
    def __init__(self):
        super(FineTuneModel, self).__init__()

        self.rep_hidden = Config.hidden_size
        self.relation_num = Config.relation_num
        self.bert_hidden = Config.bert_hidden
        self.type_embed_size = Config.type_embed_size
        self.block_size = Config.bilinear_block_size

        self.bert = BertModel.from_pretrained(Config.bert_path)

        if Config.use_group_bilinear:
            self.bilinear = nn.Linear(self.bert_hidden * self.block_size, self.rep_hidden)
        else:
            self.bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.rep_hidden)

        if Config.use_loss_weight:
            self.loss_weights = torch.FloatTensor(Config.loss_weight)
            if Config.use_gpu:
                self.loss_weights = Variable(self.loss_weights.to(Config.gpu_device))
            self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.loss_weights)
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')

        if Config.use_entity_type:
            # trainable entity type embeddings
            self.chemical_embed = nn.Parameter(torch.FloatTensor(self.type_embed_size))
            self.gene_embed = nn.Parameter(torch.FloatTensor(self.type_embed_size))
            self.disease_embed = nn.Parameter(torch.FloatTensor(self.type_embed_size))
            self.type_to_embed = {
                'Chemical': self.chemical_embed,
                'Gene': self.gene_embed,
                'Disease': self.disease_embed,
                '': torch.zeros(self.type_embed_size, 
                                dtype=torch.float16 if Config.fp16 else torch.float,
                                device=Config.gpu_device if Config.use_gpu else 'cpu')
            }
            self.linear_out = nn.Linear(self.rep_hidden + 2 * self.type_embed_size, self.relation_num)
        else:
            self.linear_out = nn.Linear(self.rep_hidden, self.relation_num)

    def forward(self, data, mode: str, eval_res: dict = None):
        """
            {'documents': torch.Size([b, 512]),
             'labels': torch.Size([b, 90, 97]),
             'head_pos': torch.Size([b, 90, 3]),
             'tail_pos': torch.Size([b, 90, 3]),
             'label_mask': torch.Size([b, 90]),
             'attn_mask': torch.Size([b, 512]),
             'pair_ids': b, 'titles': b, 'types': b}
        """

        time_tag(2, True)
        documents = data['documents']
        head_pos = data['head_pos']
        tail_pos = data['tail_pos']
        labels = data['labels']
        label_mask = data['label_mask']
        types = data['types']

        cur_batch_size = documents.shape[0]
        sample_lim = head_pos.shape[1]  # sample_limit
        mention_limit = Config.mention_padding

        time_tag(3, True, documents.shape)
        if Config.use_long_input:
            embed_docu = process_long_input(self.bert, documents, data['attn_mask'], Config)
        else:
            embed_docu = self.bert(documents, attention_mask=data['attn_mask'])[0]
        time_tag(4, True, embed_docu.shape)

        head_rep = embed_docu[[[[i] * mention_limit] * sample_lim for i in range(cur_batch_size)], head_pos]
        tail_rep = embed_docu[[[[i] * mention_limit] * sample_lim for i in range(cur_batch_size)], tail_pos]
        time_tag(5, True, head_rep.shape, tail_rep.shape)
        # max pooling
        if Config.use_logsumexp:
            head_rep = torch.logsumexp(head_rep.view(cur_batch_size, sample_lim,
                                                     mention_limit, self.bert_hidden), dim=2)
            tail_rep = torch.logsumexp(tail_rep.view(cur_batch_size, sample_lim,
                                                     mention_limit, self.bert_hidden), dim=2)
        else:
            head_rep = torch.max(head_rep.view(cur_batch_size, sample_lim, mention_limit, self.bert_hidden), dim=2)[0]
            tail_rep = torch.max(tail_rep.view(cur_batch_size, sample_lim, mention_limit, self.bert_hidden), dim=2)[0]

        time_tag(6, True, head_rep.shape, tail_rep.shape)

        if Config.use_group_bilinear:
            head_rep = head_rep.view(cur_batch_size, sample_lim, self.bert_hidden // self.block_size, self.block_size)
            tail_rep = tail_rep.view(cur_batch_size, sample_lim, self.bert_hidden // self.block_size, self.block_size)
            relation_rep = (head_rep.unsqueeze(4) * tail_rep.unsqueeze(3)).view(cur_batch_size, sample_lim,
                                                                                self.bert_hidden * self.block_size)
            relation_rep = self.bilinear(relation_rep)
        else:
            relation_rep = self.bilinear(head_rep, tail_rep)  # (batch, sample_lim, rel_embed)
        time_tag(7, True, relation_rep.shape)

        if Config.use_entity_type:
            type_tensor = []
            for batch_t in types:
                for sample_t in batch_t:
                    type_tensor.append(self.type_to_embed[sample_t[0]])
                    type_tensor.append(self.type_to_embed[sample_t[1]])
            type_tensor = torch.cat(type_tensor).view((cur_batch_size, sample_lim, self.type_embed_size * 2))
            relation_rep = torch.cat((relation_rep, type_tensor), dim=2)

        predict_out = self.linear_out(relation_rep)
        time_tag(8, True, predict_out.shape)

        if mode != 'test':
            labels = labels.view(-1, labels.shape[2])
            label_mask = label_mask.view(-1)

            predict_out = predict_out.view(cur_batch_size * sample_lim, self.relation_num)

            loss = self.loss(predict_out, labels.float())
            loss = torch.sum(loss * label_mask.unsqueeze(1)) / (torch.sum(label_mask) * self.relation_num)
            time_tag(9, True, loss.shape)

            eval_res = eval_multi_label(predict_out, labels, label_mask, eval_res)
            time_tag(10, True)
            if mode == 'valid':
                eval_res['instance_num'] = Config.valid_instance_cnt
            return {'loss': loss, 'eval_res': eval_res, 'titles': data['titles']}
        else:
            scores = None
            if Config.output_score_type == 'sigmoid':
                scores = torch.sigmoid(predict_out[:, :, Config.label2id['Pos']])
            elif Config.output_score_type == 'softmax':
                scores = torch.softmax(predict_out, dim=2)[:, :, Config.label2id['Pos']]
            if scores is not None:
                scores = F.pad(scores, (0, Config.test_sample_limit - scores.shape[-1]))
            predict_out = torch.max(predict_out, dim=2)[1]
            return {'predict': predict_out.cpu().tolist(),  # [b, 90]
                    'pair_ids': data['pair_ids'],  # [b, 90, 2]
                    'titles': data['titles'],
                    'score': scores}  # [b]
