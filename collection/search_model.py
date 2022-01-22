import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from search_preprocess import TOKENIZER
from transformers import BertModel


MODEL_CONFIG = {
    'hidden_size': 256,
    'bert_hidden': 768,
    'block_size': 64,
    'bert_path': '../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased'
}


class SearchRankModel(nn.Module):
    def __init__(self):
        super(SearchRankModel, self).__init__()

        global MODEL_CONFIG
        self.rep_hidden = MODEL_CONFIG['hidden_size']
        self.bert_hidden = MODEL_CONFIG['bert_hidden']
        self.block_size = MODEL_CONFIG['block_size']

        self.bert = BertModel.from_pretrained(MODEL_CONFIG['bert_path'])
        self.bilinear = nn.Linear(self.bert_hidden * self.block_size, self.rep_hidden)
        self.linear_out = nn.Linear(self.rep_hidden, 1)

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, data, mode: str):
        assert mode == 'test'

        documents = data['documents']
        attn_mask = data['attn_mask']
        word_pos = data['word_pos']
        head_ids = data['head_ids']
        tail_ids = data['tail_ids']

        batch_size = documents.shape[0]
        entity_lim = word_pos.shape[1]
        mention_lim = word_pos.shape[2]

        embed_docu = process_long_input(self.bert, documents, attn_mask)
        indices = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, entity_lim, mention_lim)
        entity_rep = embed_docu[indices, word_pos]
        entity_rep = torch.max(entity_rep, dim=2)[0]

        pair_num = head_ids.shape[1]
        indices = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, pair_num)

        head_rep = entity_rep[indices, head_ids]
        tail_rep = entity_rep[indices, tail_ids]

        head_rep = head_rep.view(batch_size, pair_num,
                                 self.bert_hidden // self.block_size, self.block_size)
        tail_rep = tail_rep.view(batch_size, pair_num,
                                 self.bert_hidden // self.block_size, self.block_size)
        rel_rep = (head_rep.unsqueeze(4) * tail_rep.unsqueeze(3)).view(batch_size, pair_num,
                                                                       self.bert_hidden * self.block_size)
        rel_rep = self.bilinear(rel_rep)

        score = self.linear_out(rel_rep).squeeze(2)

        return {'score': score, 'loss': 0, 'titles': data['titles']}


def process_long_input(model, input_ids, attn_mask):
    """处理较长输入, 句子长度的上限是 1024"""
    n, c = input_ids.size()
    if c <= 512:
        sequence_output = model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )[0]
    else:
        start_tokens = torch.tensor([TOKENIZER.cls_token_id]).to(input_ids)
        end_tokens = torch.tensor([TOKENIZER.sep_token_id]).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)
        new_input_ids, new_attn_mask, num_seg = [], [], []
        seq_len = attn_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attn_mask.append(attn_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attn_mask[i, :512]
                attention_mask2 = attn_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attn_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attn_mask = torch.stack(new_attn_mask, dim=0)
        sequence_output = model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )[0]
        idx = 0
        new_output = []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[idx], (0, 0, 0, c - 512))
                new_output.append(output)
            elif n_s == 2:
                output1 = sequence_output[idx][:512 - len_end]
                mask1 = attn_mask[idx][:512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))

                output2 = sequence_output[idx + 1][len_start:]
                mask2 = attn_mask[idx + 1][len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))

                mask = mask1 + mask2 + 1e-10
                # average pooling
                output = (output1 + output2) / mask.unsqueeze(-1)
                new_output.append(output)
            idx += n_s
        sequence_output = torch.stack(new_output, dim=0)
    return sequence_output
