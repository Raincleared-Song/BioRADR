import torch
import torch.nn as nn
from opt_einsum import contract
from .ATLoss import BalancedLoss
from .UNet import AttentionUNet
from transformers import BertModel
from utils import eval_multi_label
from .long_input import process_long_input
from config import ConfigFineTune as Config


class DocuNetFinetune(nn.Module):
    def __init__(self):
        super(DocuNetFinetune, self).__init__()

        emb_size, hidden_size, unet_in_dim, unet_out_dim, max_height, num_labels, channel_type, down_dim = \
            768, Config.bert_hidden, 3, 256, 35, 1, 'context-based', 256

        self.bert_model = BertModel.from_pretrained(Config.bert_path)
        self.loss_fnt = BalancedLoss()

        self.head_extractor = nn.Linear(1 * hidden_size + unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(1 * hidden_size + unet_out_dim, emb_size)
        self.bilinear = nn.Linear(emb_size * Config.bilinear_block_size, Config.relation_num)

        self.emb_size = emb_size
        self.block_size = Config.bilinear_block_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        self.bertdrop = nn.Dropout(0.6)
        self.unet_in_dim = unet_in_dim
        self.unet_out_dim = unet_out_dim
        self.liner = nn.Linear(hidden_size, unet_in_dim)
        self.min_height = max_height
        self.channel_type = channel_type
        self.segmentation_net = AttentionUNet(input_channels=unet_in_dim,
                                              class_number=unet_out_dim,
                                              down_channel=down_dim)
        assert self.channel_type == 'context-based'

    def encode(self, input_ids, attention_mask):
        sequence_output, attention = process_long_input(self.bert_model, input_ids, attention_mask, Config, True)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        bs, h, _, c = attention.size()

        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_num, e_att = -1, None
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start in e:
                        if start < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start])
                            e_att.append(attention[i, :, start])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start = e[0]
                    if start < c:
                        e_emb = sequence_output[i, start]
                        e_att = attention[i, :, start]
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            assert entity_num >= 0 and e_att is not None
            for _ in range(self.min_height-entity_num-1):
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss, entity_es, entity_as

    @staticmethod
    def get_mask(ents, bs, ne, run_device):
        ent_mask = torch.zeros(bs, ne, device=run_device)
        rel_mask = torch.zeros(bs, ne, ne, device=run_device)
        for _b in range(bs):
            ent_mask[_b, :len(ents[_b])] = 1
            rel_mask[_b, :len(ents[_b]), :len(ents[_b])] = 1
        return ent_mask, rel_mask

    @staticmethod
    def get_ht(rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def get_channel_map(self, sequence_output, entity_as):
        bs, _, d = sequence_output.size()
        ne = self.min_height

        index_pair = []
        for i in range(ne):
            tmp = torch.cat((torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)), dim=-1)
            index_pair.append(tmp)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def forward(self, data, mode: str, eval_res: dict = None):
        input_ids, attention_mask, hts = data['documents'], data['attn_mask'], data['pair_ids']
        labels, label_t, label_m, entity_pos = [], data['labels'], data['label_mask'], data['word_pos']
        label_flat, label_m_flat, sample_counts = [], [], []
        for did, doc_pair_ids in enumerate(hts):
            cur_labels, cur_poses = [], []
            for pid, _ in enumerate(doc_pair_ids):
                lab = label_t[did][pid].cpu()
                assert torch.sum(lab) == 1 and label_m[did][pid] == 1
                cur_labels.append(tuple(lab.tolist()))
                label_flat.append(lab)
                label_m_flat.append(1)
            labels.append(cur_labels)
            sample_counts.append(len(doc_pair_ids))

        loss, prediction, logits = self.forward_main(input_ids, attention_mask, labels, entity_pos, hts)

        if mode != 'test':
            eval_res = eval_multi_label(prediction, label_flat, label_m_flat, eval_res)
            if mode == 'valid':
                eval_res['instance_num'] = Config.valid_instance_cnt

            return {'loss': loss, 'eval_res': eval_res, 'titles': data['titles']}
        else:
            assert prediction.shape == logits.shape == (sum(sample_counts), 2)
            logit_list, pred_list, logit_len, pad_len = [], [], 0, max(sample_counts)
            for cnt in sample_counts:
                logit_list.append(torch.cat(
                    (logits[logit_len:(logit_len+cnt)], torch.zeros(pad_len-cnt, 2).to(logits))).unsqueeze(0))
                pred_list.append(torch.cat(
                    (prediction[logit_len:(logit_len+cnt)], torch.zeros(pad_len-cnt, 2).to(logits))).unsqueeze(0))
                logit_len += cnt
            logits = torch.cat(logit_list, dim=0)
            prediction = torch.cat(pred_list, dim=0)
            scores = None
            pos_id, na_id = Config.label2id['Pos'], Config.label2id['NA']
            if Config.output_score_type == 'sigmoid':
                scores = torch.sigmoid(logits[:, :, pos_id])
            elif Config.output_score_type == 'softmax':
                scores = torch.softmax(logits, dim=2)[:, :, pos_id]
            elif Config.output_score_type == 'pos':
                scores = logits[:, :, pos_id]
            elif Config.output_score_type == 'diff':
                scores = logits[:, :, pos_id] - logits[:, :, na_id]
            elif Config.output_score_type == 'sig_diff':
                scores = torch.sigmoid(logits[:, :, pos_id] - logits[:, :, na_id])
            predict_out = torch.max(prediction, dim=2)[1]
            return {'predict': predict_out.cpu().tolist(),
                    'pair_ids': data['pair_ids'],
                    'titles': data['titles'],
                    'score': scores}

    def forward_main(self, input_ids=None, attention_mask=None, labels=None, entity_pos=None, hts=None):
        """
        input_ids: [4, 315]
        attention_mask: [4, 315]
        labels: [6, 8, 49, 3]
        entity_pos: [5, 9, 14, 4]
        hts: [6, 8, 49, 3]
        """
        # [4, 315, 768], [4, 12, 315, 315]
        sequence_output, attention = self.encode(input_ids, attention_mask)

        # get hs, ts and entity_embs >> entity_rs
        # [66, 768], [66, 768], [5, 9, 14, 4], [35, 35, 35, 35]
        hs, ts, entity_embs, entity_as = self.get_hrt(sequence_output, attention, entity_pos, hts)

        # 获得通道map的两种不同方法
        feature_map = self.get_channel_map(sequence_output, entity_as)
        attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()

        attn_map = self.segmentation_net(attn_input)
        h_t = self.get_ht(attn_map, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        # [66, 2], 0-1 Tensor
        output = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            return loss.to(sequence_output), output, logits
        return output, logits
