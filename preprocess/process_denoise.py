import random
import torch
from utils import get_unused_token, type_convert
from config import ConfigDenoise as Config
from .document_crop import document_crop, sentence_mention_crop


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
    documents, attn_masks, word_positions, head_ids, tail_ids, pos_ids, titles, pair_ids = \
        [], [], [], [], [], [], [], []
    pmid_key = 'pmid' if len(data) > 0 and 'pmid' in data[0] else 'pmsid'
    for doc in data:
        document, attn_mask, position = process_document(doc, mode)
        documents.append(document)
        attn_masks.append(attn_mask)
        word_positions.append(position)

        entity_num = len(doc['vertexSet'])
        if use_cp:
            pairs = []
            entities = doc['vertexSet']
            for i in range(entity_num):
                for j in range(entity_num):
                    if 'chemical' in entities[i][0]['type'].lower() and \
                            'gene' in entities[j][0]['type'].lower():
                        pairs.append((i, j))
        else:
            entities = doc['vertexSet']
            pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if
                     'chemical' in entities[i][0]['type'].lower() and 'disease' in entities[j][0]['type'].lower()]
        pair_ids.append(pairs)
        head_ids.append([pair[0] for pair in pairs])
        tail_ids.append([pair[1] for pair in pairs])
        titles.append(str(doc[pmid_key]))

    # dynamic pad positions
    entity_padding = max(len(item) for item in word_positions)
    for item in word_positions:
        for _ in range(entity_padding - len(item)):
            item.append([0] * Config.mention_padding)
    # dynamic pad head/tail ids
    test_sample_limit = max(len(item) for item in head_ids)
    sample_counts = []
    for head_id, tail_id in zip(head_ids, tail_ids):
        assert len(head_id) == len(tail_id)
        sample_counts.append(len(head_id))
        head_id += [0] * (test_sample_limit - len(head_id))
        tail_id += [0] * (test_sample_limit - len(tail_id))
    float_type = torch.FloatType if Config.model_type == 'bert' else torch.HalfTensor

    return {
        'documents': torch.LongTensor(documents),
        'attn_mask': float_type(attn_masks),
        'word_pos': torch.LongTensor(word_positions),
        'head_ids': torch.LongTensor(head_ids),
        'tail_ids': torch.LongTensor(tail_ids),
        'titles': titles,
        'sample_counts': sample_counts,
        'pair_ids': pair_ids,
    }


def process_document(data, mode: str):
    if Config.crop_documents:
        document_crop(data)
    sentence_mention_crop(data, mode, Config.crop_mention_option)
    sentences = [[Config.tokenizer.tokenize(word) for word in sent] for sent in data['sents']]

    entities = data['vertexSet']
    for i, mentions in enumerate(entities):
        for mention in mentions:
            mention['type'] = type_convert(mention['type'])
            if Config.entity_marker_type not in ['t', 't-m']:
                tmp: list = sentences[mention['sent_id']][mention['pos'][0]]
                if Config.entity_marker_type == 'mt':
                    # both mention and type
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [get_unused_token((i << 1) + 1), mention['type'], '*'] + tmp
                elif Config.entity_marker_type == 'm*':
                    sentences[mention['sent_id']][mention['pos'][0]] = ['*'] + tmp
                else:
                    # Config.entity_marker_type == 'm', only mention
                    sentences[mention['sent_id']][mention['pos'][0]] = [get_unused_token((i << 1) + 1)] + tmp
            else:
                # Config.entity_marker_type in ['t', 't-m'], blank all mention, only type
                # t-m: type, *, [MASK]
                for pos in range(mention['pos'][0], mention['pos'][1]):
                    sentences[mention['sent_id']][pos] = []
                if Config.entity_marker_type == 't':
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [get_unused_token((i << 1) + 1), mention['type'], '*', get_unused_token(0)]
                else:
                    assert Config.entity_marker_type == 't-m'
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [mention['type'], '*', get_unused_token(0)]
            if Config.entity_marker_type == 'm*':
                sentences[mention['sent_id']][mention['pos'][1] - 1].append('*')
            elif Config.entity_marker_type != 't-m':
                sentences[mention['sent_id']][mention['pos'][1] - 1].append(get_unused_token((i + 1) << 1))
            else:
                pass

    cls_token, sep_token, pad_token = Config.tokenizer.cls_token, Config.tokenizer.sep_token, Config.tokenizer.pad_token
    word_position, document = [], [cls_token]
    for sent in sentences:
        word_position.append([])
        for word in sent:
            word_position[-1].append(len(document))
            document += word
    word_position.append([len(document)])

    # pad each document
    if len(document) < Config.token_padding:
        document.append(sep_token)
        attn_mask = [1] * len(document) + [0] * (Config.token_padding - len(document))
        document += [pad_token] * (Config.token_padding - len(document))
    else:
        document = document[:(Config.token_padding - 1)] + [sep_token]
        attn_mask = [1] * Config.token_padding

    positions = []
    for entity in entities:
        cur_entity = []
        for mention in entity:
            if word_position[mention['sent_id']][mention['pos'][0]] < Config.token_padding:
                cur_entity.append(word_position[mention['sent_id']][mention['pos'][0]])
            if len(cur_entity) == Config.mention_padding:
                break
        positions.append(cur_entity)
    # padding length of mention number to 3
    for i in range(len(positions)):
        if len(positions[i]) == 0:
            positions[i] = [0] * Config.mention_padding
        positions[i] += [positions[i][0]] * (Config.mention_padding - len(positions[i]))

    return Config.tokenizer.convert_tokens_to_ids(document), attn_mask, positions


def process_denoise_train(data, mode: str):
    assert mode != 'test'

    documents1, documents2 = [], []
    positions1, positions2 = [], []
    attn_mask1, attn_mask2 = [], []

    # relation detection
    # intra-document
    rd_head_ids1, rd_head_ids2 = [], []
    rd_tail_ids1, rd_tail_ids2 = [], []
    rd_pair_ids1, rd_pair_ids2 = [], []
    rd_label1, rd_label2 = [], []

    # inter-document
    rd_head_ids1_x, rd_head_ids2_x = [], []
    rd_tail_ids1_x, rd_tail_ids2_x = [], []
    rd_pair_ids1_x, rd_pair_ids2_x = [], []
    rd_label_x = []

    for item in data:
        doc1, doc2 = item['doc1'], item['doc2']

        document, mask, pos = process_document(doc1, mode)
        documents1.append(document)
        attn_mask1.append(mask)
        positions1.append(pos)

        document, mask, pos = process_document(doc2, mode)
        documents2.append(document)
        attn_mask2.append(mask)
        positions2.append(pos)

        # intra RD
        head, tail, pair_ids, label = process_intra_rank(doc1)
        rd_head_ids1.append(head)
        rd_tail_ids1.append(tail)
        rd_pair_ids1.append(pair_ids)
        rd_label1.append(label)

        head, tail, pair_ids, label = process_intra_rank(doc2)
        rd_head_ids2.append(head)
        rd_tail_ids2.append(tail)
        rd_pair_ids2.append(pair_ids)
        rd_label2.append(label)

        # inter RD
        if Config.use_inter:
            head1, tail1, head2, tail2, pair_ids1, pair_ids2, label = process_inter_rank(doc1, doc2)
            rd_head_ids1_x.append(head1)
            rd_head_ids2_x.append(head2)
            rd_tail_ids1_x.append(tail1)
            rd_tail_ids2_x.append(tail2)
            rd_pair_ids1_x.append(pair_ids1)
            rd_pair_ids2_x.append(pair_ids2)
            rd_label_x.append(label)

    # dynamically padding positions
    entity_pad1, entity_pad2 = max(len(p) for p in positions1), max(len(p) for p in positions2)
    for idx, cur_p in enumerate(positions1):
        pad_len = entity_pad1 - len(cur_p)
        for _ in range(pad_len):
            cur_p.append([0] * Config.mention_padding)
    for idx, cur_p in enumerate(positions2):
        pad_len = entity_pad2 - len(cur_p)
        for _ in range(pad_len):
            cur_p.append([0] * Config.mention_padding)
    float_type = torch.FloatType if Config.model_type == 'bert' else torch.HalfTensor

    return {
        'document1': torch.LongTensor(documents1),
        'document2': torch.LongTensor(documents2),
        'positions1': torch.LongTensor(positions1),
        'positions2': torch.LongTensor(positions2),
        'attn_mask1': float_type(attn_mask1),
        'attn_mask2': float_type(attn_mask2),

        'rd_head_ids1': torch.LongTensor(rd_head_ids1),
        'rd_head_ids2': torch.LongTensor(rd_head_ids2),
        'rd_tail_ids1': torch.LongTensor(rd_tail_ids1),
        'rd_tail_ids2': torch.LongTensor(rd_tail_ids2),
        'rd_pair_ids1': rd_pair_ids1,
        'rd_pair_ids2': rd_pair_ids2,
        'rd_label1': torch.LongTensor(rd_label1),
        'rd_label2': torch.LongTensor(rd_label2),

        'rd_head_ids1_x': torch.LongTensor(rd_head_ids1_x),
        'rd_head_ids2_x': torch.LongTensor(rd_head_ids2_x),
        'rd_tail_ids1_x': torch.LongTensor(rd_tail_ids1_x),
        'rd_tail_ids2_x': torch.LongTensor(rd_tail_ids2_x),
        'rd_pair_ids1_x': rd_pair_ids1_x,
        'rd_pair_ids2_x': rd_pair_ids2_x,
        'rd_label_x': torch.LongTensor(rd_label_x),
    }


def get_pos_neg_pairs(data, ret_dict=False):
    entities = data['vertexSet']
    entity_num = len(entities)
    positive_pairs = {(lab['h'], lab['t']): lab['r'] for lab in data['labels'] if lab['r'] != 'NA'}
    negative_pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if
                      'chemical' in entities[i][0]['type'].lower() and 'disease' in entities[j][0]['type'].lower()
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
    head_ids, tail_ids, label_ids, pair_ids = [], [], [], []
    if len(positive_pairs) == 0:
        head_ids = tail_ids = [[0] * (Config.negative_num + 1)] * sample_limit
        pair_ids = [[(0, 0)] * (Config.negative_num + 1)] * sample_limit
        label_ids = [-100] * sample_limit
        return head_ids, tail_ids, pair_ids, label_ids

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
        pair_ids.append([])

        negative_samples = random.sample(negative_pairs, Config.negative_num)
        pos = random.randint(0, Config.negative_num)
        pairs = negative_samples[:pos] + [random.choice(positive_pairs)] + negative_samples[pos:]
        label_ids.append(pos)
        for pair in pairs:
            head_ids[-1].append(pair[0])
            tail_ids[-1].append(pair[1])
            pair_ids[-1].append(pair)
    return head_ids, tail_ids, pair_ids, label_ids


def process_inter_rank(data1, data2):
    positive_pairs1, negative_pairs1 = get_pos_neg_pairs(data1)
    positive_pairs2, negative_pairs2 = get_pos_neg_pairs(data2)

    head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids = [], [], [], [], []
    pair_ids1, pair_ids2 = [], []
    assert Config.negative_num % 2 == 1
    half_sample_num = (Config.negative_num + 1) // 2
    if len(positive_pairs1) == 0 or len(positive_pairs2) == 0:
        head_ids1 = head_ids2 = tail_ids1 = tail_ids2 = [[0] * half_sample_num] * 3
        pair_ids1 = pair_ids2 = [[(0, 0)] * half_sample_num] * 3
        label_ids = [-100] * 3
        return head_ids1, tail_ids1, head_ids2, tail_ids2, pair_ids1, pair_ids2, label_ids

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
            pair_ids1.append([chosen_pair1])
            head_ids2.append([chosen_pair2[0]])
            tail_ids2.append([chosen_pair2[1]])
            pair_ids2.append([chosen_pair2])
        return head_ids1, tail_ids1, head_ids2, tail_ids2, pair_ids1, pair_ids2, label_ids

    for i in range(Config.positive_num):
        head_ids1.append([])
        tail_ids1.append([])
        pair_ids1.append([])
        head_ids2.append([])
        tail_ids2.append([])
        pair_ids2.append([])
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
            pair_ids1[-1].append(pair)
        for pair in pairs2:
            head_ids2[-1].append(pair[0])
            tail_ids2[-1].append(pair[1])
            pair_ids2[-1].append(pair)

    return head_ids1, tail_ids1, head_ids2, tail_ids2, pair_ids1, pair_ids2, label_ids
