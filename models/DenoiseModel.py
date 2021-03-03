import torch
import torch.nn as nn
from config import ConfigDenoise as Config
from transformers import BertModel
from utils import eval_softmax


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()

        self.rep_hidden = Config.hidden_size
        self.relation_num = Config.relation_num
        self.bert_hidden = Config.bert_hidden

        self.bert = BertModel.from_pretrained(Config.bert_path)
        self.bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.rep_hidden)
        self.linear_out = nn.Linear(self.rep_hidden, 1)

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, data, mode: str, eval_res: dict = None):
        """
            documents torch.Size([4, 512])
            attn_mask torch.Size([4, 512])
            word_pos torch.Size([4, 42, 3])
            head_ids torch.Size([4, 3, 6])
            tail_ids torch.Size([4, 3, 6])
            labels torch.Size([4, 3])

            documents torch.Size([4, 512])
            attn_mask torch.Size([4, 512])
            word_pos torch.Size([4, 42, 3])
            head_ids torch.Size([4, 16, 8])
            tail_ids torch.Size([4, 16, 8])
            labels torch.Size([4, 16])
        """
        documents = data['documents']
        attn_mask = data['attn_mask']
        word_pos = data['word_pos']
        head_ids = data['head_ids']
        tail_ids = data['tail_ids']

        batch_size = documents.shape[0]
        entity_lim = word_pos.shape[1]
        mention_lim = word_pos.shape[2]

        embed_docu = self.bert(documents, attention_mask=attn_mask)[0]
        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim, mention_lim)
        entity_rep = embed_docu[indices, word_pos]
        entity_rep = torch.max(entity_rep, dim=2)[0]

        if mode == 'test':
            pair_num = head_ids.shape[1]
            indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, pair_num)

            head_rep = entity_rep[indices, head_ids]
            tail_rep = entity_rep[indices, tail_ids]
            rel_rep = self.bilinear(head_rep, tail_rep)
            score = self.linear_out(rel_rep).squeeze(2)

            return {'score': score, 'loss': 0, 'titles': data['titles']}

        else:
            if eval_res is None:
                eval_res = {'RD': {'correct_num': 0, 'instance_num': 0}}

            instance_num, candidate_num = head_ids.shape[1], head_ids.shape[2]
            indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, instance_num, candidate_num)

            head_rep = entity_rep[indices, head_ids].view(batch_size, instance_num, candidate_num, self.bert_hidden)
            tail_rep = entity_rep[indices, tail_ids].view(batch_size, instance_num, candidate_num, self.bert_hidden)
            rel_rep = self.bilinear(head_rep, tail_rep)
            score = self.linear_out(rel_rep).squeeze(3).view(-1, candidate_num)

            labels = data['labels'].view(-1)
            loss = self.loss(score, labels)
            eval_res['RD'] = eval_softmax(score, labels, eval_res['RD'])

            return {'loss': loss, 'eval_res': eval_res}
