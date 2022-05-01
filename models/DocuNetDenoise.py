import torch
import torch.nn as nn
from opt_einsum import contract
from .UNet import AttentionUNet
from transformers import BertModel
from .long_input import process_long_input
from config import ConfigDenoise as Config
from .ATLoss import BinaryATLoss
from .ContrastiveLoss import ContrastiveLoss
from .LogExpLoss import LogExpLoss
from utils import eval_softmax


loss_map = {
    'cross_entropy': nn.CrossEntropyLoss,
    'adaptive_threshold': BinaryATLoss,
    'contrastive_mrl': ContrastiveLoss,
    'contrastive_sml': ContrastiveLoss,
    'log_exp': LogExpLoss,
}


class DocuNetDenoise(nn.Module):
    def __init__(self):
        super(DocuNetDenoise, self).__init__()

        emb_size, hidden_size, unet_in_dim, unet_out_dim, max_height, num_labels, channel_type, down_dim = \
            768, Config.bert_hidden, 3, 256, 35, 1, 'context-based', 256

        self.bert_model = BertModel.from_pretrained(Config.bert_path)
        self.loss_fnt = loss_map[Config.loss_func]()

        self.head_extractor = nn.Linear(1 * hidden_size + unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(1 * hidden_size + unet_out_dim, emb_size)
        # generate a single score
        self.bilinear = nn.Linear(emb_size * Config.block_size, 1)

        self.emb_size = emb_size
        self.block_size = Config.block_size
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
        if mode != 'test':
            return self.forward_train(data, mode, eval_res)
        else:
            return self.forward_test(data, mode)

    def forward_train(self, data, mode: str, eval_res: dict = None):
        assert mode != 'test'
        if eval_res is None:
            eval_res = {'RD': {'correct_num': 0, 'instance_num': 0}, 'RD_X': {'correct_num': 0, 'instance_num': 0}}

        document1, document2 = data['document1'], data['document2']
        positions1, positions2 = data['positions1'], data['positions2']
        attn_mask1, attn_mask2 = data['attn_mask1'], data['attn_mask2']

        # relation detection
        # intra-document
        rd_head_ids1, _ = data['rd_head_ids1'], data['rd_head_ids2']
        # 1 * 16 sample * 16 candidates * 2
        rd_pair_ids1, rd_pair_ids2 = data['rd_pair_ids1'], data['rd_pair_ids2']
        rd_label1, rd_label2 = data['rd_label1'], data['rd_label2']

        # inter-document
        rd_head_ids1_x, _ = data['rd_head_ids1_x'], data['rd_head_ids2_x']
        rd_tail_ids1_x, _ = data['rd_tail_ids1_x'], data['rd_tail_ids2_x']
        rd_pair_ids1_x, rd_pair_ids2_x = data['rd_pair_ids1_x'], data['rd_pair_ids2_x']
        rd_label_x = data['rd_label_x']

        batch_sz, sample_cnt, candidate_cnt = rd_head_ids1.shape
        sequence_output1, attention1 = self.encode(document1, attn_mask1)
        sequence_output2, attention2 = self.encode(document2, attn_mask2)

        hts1, hts2 = [[]], [[]]
        assert len(rd_pair_ids1) == len(rd_pair_ids2) == batch_sz
        for doc_pairs1, doc_pairs2 in zip(rd_pair_ids1, rd_pair_ids2):
            assert len(doc_pairs1) == len(doc_pairs2) == sample_cnt
            for sample1, sample2 in zip(doc_pairs1, doc_pairs2):
                assert len(sample1) == len(sample2) == candidate_cnt
                hts1[-1] += sample1
                hts2[-1] += sample2

        assert rd_head_ids1_x.shape == rd_tail_ids1_x.shape == (batch_sz, sample_cnt, candidate_cnt // 2)
        assert len(rd_pair_ids1_x) == len(rd_pair_ids2_x) == batch_sz
        for doc_pairs1_x, doc_pairs2_x in zip(rd_pair_ids1_x, rd_pair_ids2_x):
            assert len(doc_pairs1_x) == len(doc_pairs2_x) == sample_cnt
            for sample1_x, sample2_x in zip(doc_pairs1_x, doc_pairs2_x):
                assert len(sample1_x) == len(sample2_x) == candidate_cnt // 2
                hts1[-1] += sample1_x
                hts2[-1] += sample2_x

        # 1 * sample_cnt * (candidate_cnt + candidate_cnt // 2)
        logits1 = self.forward_logits(sequence_output1, attention1, positions1, hts1)
        logits2 = self.forward_logits(sequence_output2, attention2, positions2, hts2)
        rd_label1, rd_label2, rd_label_x = rd_label1.view(-1), rd_label2.view(-1), rd_label_x.view(-1)

        intra_score1 = logits1[:, :(sample_cnt*candidate_cnt)].view(sample_cnt, candidate_cnt)
        intra_score2 = logits2[:, :(sample_cnt*candidate_cnt)].view(sample_cnt, candidate_cnt)

        loss1 = self.loss_fnt(intra_score1, rd_label1)
        eval_res['RD'] = eval_softmax(intra_score1, rd_label1, eval_res['RD'])
        loss2 = self.loss_fnt(intra_score2, rd_label2)
        eval_res['RD'] = eval_softmax(intra_score2, rd_label2, eval_res['RD'])
        total_loss = loss1 + loss2

        if Config.use_inter:
            inter_score1 = logits1[:, (sample_cnt*candidate_cnt):].view(sample_cnt, candidate_cnt // 2)
            inter_score2 = logits2[:, (sample_cnt*candidate_cnt):].view(sample_cnt, candidate_cnt // 2)
            inter_score = torch.cat((inter_score1, inter_score2), dim=1).view(sample_cnt, candidate_cnt)
            loss = self.loss_fnt(inter_score, rd_label_x)
            eval_res['RD_X'] = eval_softmax(inter_score, rd_label_x, eval_res['RD_X'])
            total_loss += loss

        return {'loss': total_loss, 'eval_res': eval_res}

    def forward_test(self, data, mode: str):
        assert mode == 'test'
        input_ids, attention_mask, hts, entity_pos = \
            data['documents'], data['attn_mask'], data['pair_ids'], data['word_pos']
        sequence_output, attention = self.encode(input_ids, attention_mask)
        score = self.forward_logits(sequence_output, attention, entity_pos, hts)
        return {'score': score, 'loss': 0, 'titles': data['titles']}

    def forward_logits(self, sequence_output=None, attention=None, entity_pos=None, hts=None):
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

        sample_counts = [len(x) for x in hts]
        logit_list, logit_len, pad_len = [], 0, max(sample_counts)
        for cnt in sample_counts:
            logit_list.append(torch.cat(
                (logits[logit_len:(logit_len + cnt)], torch.zeros(pad_len - cnt, 1).to(logits))).unsqueeze(0))
            logit_len += cnt
        logits = torch.cat(logit_list, dim=0).squeeze(2)

        return logits
