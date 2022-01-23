import random
import torch
from config import ConfigDenoise as Config
from .document_crop import document_crop, mention_choose_by_pair


use_cp: bool = False


def process_denoise(data, mode: str):
    if mode == 'test':
        return process_denoise_test(data, mode)
    else:
        return process_denoise_train(data, mode)


def process_denoise_test(data, mode: str):
    assert mode == 'test'

    global use_cp
    use_cp = 'chemprot' in Config.data_path[mode].lower()
    # head_poses: batch_size * sample_limit * mention_limit
    documents, attn_masks, head_poses, tail_poses, pos_ids, titles = [], [], [], [], [], []
    pmid_key = 'pmid' if len(data) > 0 and 'pmid' in data[0] else 'pmsid'
    for doc in data:
        document, attn_mask, position = process_document(doc, mode)
        documents.append(document)
        attn_masks.append(attn_mask)

        entity_num = len(doc['vertexSet'])
        if use_cp:
            pairs = []
            entities = doc['vertexSet']
            for i in range(entity_num):
                for j in range(entity_num):
                    if entities[i][0]['type'].lower().startswith('chemical') and \
                            entities[j][0]['type'].lower().startswith('gene'):
                        pairs.append((i, j))
        else:
            entities = doc['vertexSet']
            pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if
                     entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease']
        head_pos, tail_pos = [], []
        for hid, tid in pairs:
            head_p, tail_p = mention_choose_by_pair(doc, hid, tid, position,
                                                    Config.mention_padding, Config.crop_mention_option)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
        head_poses.append(head_pos)
        tail_poses.append(tail_pos)
        titles.append(doc[pmid_key])

    # dynamic pad head/tail ids
    test_sample_limit = max(len(item) for item in head_poses)
    sample_counts = []
    for head_pos, tail_pos in zip(head_poses, tail_poses):
        assert len(head_pos) == len(tail_pos)
        sample_counts.append(len(head_pos))
        for _ in range(test_sample_limit - len(head_pos)):
            head_pos.append([0] * Config.mention_padding)
            tail_pos.append([0] * Config.mention_padding)

    return {
        'documents': torch.LongTensor(documents),
        'attn_mask': torch.FloatTensor(attn_masks),
        'head_poses': torch.LongTensor(head_poses),
        'tail_poses': torch.LongTensor(tail_poses),
        'titles': titles,
        'sample_counts': sample_counts
    }


def process_document(data, mode: str):
    if Config.crop_documents:
        document_crop(data)
    sentences = [[Config.tokenizer.tokenize(word) for word in sent] for sent in data['sents']]

    entities = data['vertexSet']
    for i, mentions in enumerate(entities):
        for mention in mentions:
            if Config.entity_marker_type != 't':
                tmp: list = sentences[mention['sent_id']][mention['pos'][0]]
                if Config.entity_marker_type == 'mt':
                    # both mention and type
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [f'[unused{(i << 1) + 1}]', mention['type'], '*'] + tmp
                else:
                    # Config.entity_marker_type == 'm', only mention
                    sentences[mention['sent_id']][mention['pos'][0]] = [f'[unused{(i << 1) + 1}]'] + tmp
            else:
                # Config.entity_marker_type == 't', blank all mention, only type
                for pos in range(mention['pos'][0], mention['pos'][1]):
                    sentences[mention['sent_id']][pos] = []
                sentences[mention['sent_id']][mention['pos'][0]] = \
                    [f'[unused{(i << 1) + 1}]', mention['type'], '*', '[unused0]']
            sentences[mention['sent_id']][mention['pos'][1] - 1].append(f'[unused{(i + 1) << 1}]')

    word_position, document = [], ['[CLS]']
    for sent in sentences:
        word_position.append([])
        for word in sent:
            word_position[-1].append(len(document))
            document += word
    word_position.append([len(document)])

    # pad each document
    if len(document) < Config.token_padding:
        document.append('[SEP]')
        document += ['[PAD]'] * (Config.token_padding - len(document))
        attn_mask = [1] * len(document) + [0] * (Config.token_padding - len(document))
    else:
        document = document[:(Config.token_padding - 1)] + ['[SEP]']
        attn_mask = [1] * Config.token_padding

    positions = []
    for entity in entities:
        cur_entity = []
        for mention in entity:
            if word_position[mention['sent_id']][mention['pos'][0]] < Config.token_padding:
                cur_entity.append((mention['sent_id'], word_position[mention['sent_id']][mention['pos'][0]]))
        positions.append(cur_entity)

    return Config.tokenizer.convert_tokens_to_ids(document), attn_mask, positions


def get_position_matrix(heads, tails, doc, word_pos):
    assert len(heads) == len(tails)
    # instance_num * candidate_num * mention_limit
    head_pos, tail_pos = [], []
    for cans_h, cans_t in zip(heads, tails):
        cur_pos_h, cur_pos_t = [], []
        assert len(cans_h) == len(cans_t)
        for hid, tid in zip(cans_h, cans_t):
            head_p, tail_p = mention_choose_by_pair(doc, hid, tid, word_pos,
                                                    Config.mention_padding, Config.crop_mention_option)
            cur_pos_h.append(head_p)
            cur_pos_t.append(tail_p)
        head_pos.append(cur_pos_h)
        tail_pos.append(cur_pos_t)
    return head_pos, tail_pos


def process_denoise_train(data, mode: str):
    assert mode != 'test'

    documents1, documents2 = [], []
    attn_mask1, attn_mask2 = [], []

    # relation detection
    # intra-document
    rd_head_poses1, rd_head_poses2 = [], []
    rd_tail_poses1, rd_tail_poses2 = [], []
    rd_label1, rd_label2 = [], []

    # inter-document
    rd_head_poses1_x, rd_tail_poses1_x = [], []
    rd_head_poses2_x, rd_tail_poses2_x = [], []
    rd_label_x = []

    for item in data:
        doc1, doc2 = item['doc1'], item['doc2']

        document, mask, pos1 = process_document(doc1, mode)
        documents1.append(document)
        attn_mask1.append(mask)

        document, mask, pos2 = process_document(doc2, mode)
        documents2.append(document)
        attn_mask2.append(mask)

        # intra RD
        head, tail, label = process_intra_rank(doc1)
        head_positions, tail_positions = get_position_matrix(head, tail, doc1, pos1)
        rd_head_poses1.append(head_positions)
        rd_tail_poses1.append(tail_positions)
        rd_label1.append(label)

        head, tail, label = process_intra_rank(doc2)
        head_positions, tail_positions = get_position_matrix(head, tail, doc2, pos2)
        rd_head_poses2.append(head_positions)
        rd_tail_poses2.append(tail_positions)
        rd_label2.append(label)

        # inter RD
        if Config.use_inter:
            head1, tail1, head2, tail2, label = process_inter_rank(doc1, doc2)
            head_positions, tail_positions = get_position_matrix(head1, tail1, doc1, pos1)
            rd_head_poses1_x.append(head_positions)
            rd_tail_poses1_x.append(tail_positions)
            head_positions, tail_positions = get_position_matrix(head2, tail2, doc2, pos2)
            rd_head_poses2_x.append(head_positions)
            rd_tail_poses2_x.append(tail_positions)
            rd_label_x.append(label)

    return {
        'document1': torch.LongTensor(documents1),
        'document2': torch.LongTensor(documents2),
        'attn_mask1': torch.FloatTensor(attn_mask1),
        'attn_mask2': torch.FloatTensor(attn_mask2),

        'rd_head_poses1': torch.LongTensor(rd_head_poses1),
        'rd_tail_poses1': torch.LongTensor(rd_tail_poses1),
        'rd_head_poses2': torch.LongTensor(rd_head_poses2),
        'rd_tail_poses2': torch.LongTensor(rd_tail_poses2),
        'rd_label1': torch.LongTensor(rd_label1),
        'rd_label2': torch.LongTensor(rd_label2),

        'rd_head_poses1_x': torch.LongTensor(rd_head_poses1_x),
        'rd_tail_poses1_x': torch.LongTensor(rd_tail_poses1_x),
        'rd_head_poses2_x': torch.LongTensor(rd_head_poses2_x),
        'rd_tail_poses2_x': torch.LongTensor(rd_tail_poses2_x),
        'rd_label_x': torch.LongTensor(rd_label_x),
    }


def get_pos_neg_pairs(data, ret_dict=False):
    entities = data['vertexSet']
    entity_num = len(entities)
    positive_pairs = {(lab['h'], lab['t']): lab['r'] for lab in data['labels'] if lab['r'] != 'NA'}
    negative_pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if
                      entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease'
                      and (i, j) not in positive_pairs]
    while len(negative_pairs) < Config.negative_num:
        negative_pairs *= 2
    if ret_dict:
        return positive_pairs, negative_pairs
    else:
        return list(positive_pairs.keys()), negative_pairs


def process_intra_rank(data):
    positive_pairs, negative_pairs = get_pos_neg_pairs(data)

    sample_limit = Config.positive_num
    head_ids, tail_ids, label_ids = [], [], []
    if len(positive_pairs) == 0:
        head_ids = tail_ids = [[0] * (Config.negative_num + 1)] * sample_limit
        label_ids = [-100] * sample_limit
        return head_ids, tail_ids, label_ids

    if Config.loss_func.startswith('contrastive'):
        # contrastive learning
        for i in range(sample_limit):
            # dissimilar samples
            pos_pair = random.choice(positive_pairs)
            neg_pair = random.choice(negative_pairs)
            head_ids.append([pos_pair[0], neg_pair[0]])
            tail_ids.append([pos_pair[1], neg_pair[1]])
            # 1 stands for "pos is ahead of neg"
            label_ids.append(1)
        return head_ids, tail_ids, label_ids

    for i in range(sample_limit):
        head_ids.append([])
        tail_ids.append([])

        negative_samples = random.sample(negative_pairs, Config.negative_num)
        pos = random.randint(0, Config.negative_num)
        pairs = negative_samples[:pos] + [random.choice(positive_pairs)] + negative_samples[pos:]
        label_ids.append(pos)
        for pair in pairs:
            head_ids[-1].append(pair[0])
            tail_ids[-1].append(pair[1])
    return head_ids, tail_ids, label_ids


def process_inter_rank(data1, data2):
    positive_pairs1, negative_pairs1 = get_pos_neg_pairs(data1)
    positive_pairs2, negative_pairs2 = get_pos_neg_pairs(data2)

    head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids = [], [], [], [], []
    assert Config.negative_num % 2 == 1
    half_sample_num = (Config.negative_num + 1) // 2
    if len(positive_pairs1) == 0 or len(positive_pairs2) == 0:
        head_ids1 = head_ids2 = tail_ids1 = tail_ids2 = [[0] * half_sample_num] * 3
        label_ids = [-100] * 3
        return head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids

    if Config.loss_func.startswith('contrastive'):
        # contrastive learning
        for i in range(Config.positive_num):
            # dissimilar samples
            if random.random() < 0.5:
                chosen_pair1 = random.choice(positive_pairs1)
                chosen_pair2 = random.choice(negative_pairs2)
                # 1 stands for "pos is ahead of neg"
                label_ids.append(1)
            else:
                chosen_pair1 = random.choice(negative_pairs1)
                chosen_pair2 = random.choice(positive_pairs2)
                # -1 stands for "neg is ahead of pos"
                label_ids.append(-1)
            head_ids1.append([chosen_pair1[0]])
            tail_ids1.append([chosen_pair1[1]])
            head_ids2.append([chosen_pair2[0]])
            tail_ids2.append([chosen_pair2[1]])
        return head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids

    for i in range(Config.positive_num):
        head_ids1.append([])
        tail_ids1.append([])
        head_ids2.append([])
        tail_ids2.append([])
        pairs1 = random.sample(negative_pairs1, half_sample_num)
        pairs2 = random.sample(negative_pairs2, half_sample_num)
        pos = random.randint(0, half_sample_num - 1)
        if random.random() < 0.5:
            label_ids.append(pos)
            pairs1[pos] = random.choice(positive_pairs1)
        else:
            label_ids.append(pos + half_sample_num)
            pairs2[pos] = random.choice(positive_pairs2)
        for pair in pairs1:
            head_ids1[-1].append(pair[0])
            tail_ids1[-1].append(pair[1])
        for pair in pairs2:
            head_ids2[-1].append(pair[0])
            tail_ids2[-1].append(pair[1])

    return head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids
