import torch
import random
from config import ConfigPretrain as Config


def process_pretrain(data, mode: str):
    assert mode != 'test'

    documents1, documents2 = [], []
    positions1, positions2 = [], []
    attn_mask1, attn_mask2 = [], []

    # mention-entity matching
    # intra-document
    mem_query1, mem_query2 = [], []
    mem_candidate1, mem_candidate2 = [], []
    mem_label1, mem_label2 = [], []

    # inter-document
    mem_query1_x, mem_query2_x = [], []
    mem_candidate1_x, mem_candidate2_x = [], []
    mem_label1_x, mem_label2_x = [], []

    # relation detection
    # intra-document
    rd_head_ids1, rd_head_ids2 = [], []
    rd_tail_ids1, rd_tail_ids2 = [], []
    rd_label1, rd_label2 = [], []

    # inter-document
    rd_head_ids1_x, rd_head_ids2_x = [], []
    rd_tail_ids1_x, rd_tail_ids2_x = [], []
    rd_label_x = []

    # relation fact alignment
    # intra-document
    rfa_query_head1, rfa_query_head2 = [], []
    rfa_query_tail1, rfa_query_tail2 = [], []
    rfa_candidate_head1, rfa_candidate_head2 = [], []
    rfa_candidate_tail1, rfa_candidate_tail2 = [], []
    rfa_label1, rfa_label2 = [], []

    # inter-document
    rfa_query_head1_x, rfa_query_head2_x = [], []
    rfa_query_tail1_x, rfa_query_tail2_x = [], []
    rfa_candidate_head1_x, rfa_candidate_head2_x = [], []
    rfa_candidate_tail1_x, rfa_candidate_tail2_x = [], []
    rfa_label12, rfa_label21 = [], []

    for item in data:
        doc1, doc2 = item['doc1'], item['doc2']
        pair1, pair2 = item['pair1'], item['pair2']

        # encode document and intra MEM
        mention_mask1 = process_intra_mention(doc1)
        mention_mask2 = process_intra_mention(doc2)

        document, mask, pos, men_pos, men_candidate, men_lab = process_document(doc1, mode, mention_mask1)
        documents1.append(document)
        attn_mask1.append(mask)
        positions1.append(pos)
        mem_query1.append(men_pos)
        mem_candidate1.append(men_candidate)
        mem_label1.append(men_lab)

        document, mask, pos, men_pos, men_candidate, men_lab = process_document(doc2, mode, mention_mask2)
        documents2.append(document)
        attn_mask2.append(mask)
        positions2.append(pos)
        mem_query2.append(men_pos)
        mem_candidate2.append(men_candidate)
        mem_label2.append(men_lab)

        # inter MEM
        query1, query2, candidate1, candidate2, label1, label2 = process_inter_mention(doc1, doc2)
        mem_query1_x.append(query1)
        mem_query2_x.append(query2)
        mem_candidate1_x.append(candidate1)
        mem_candidate2_x.append(candidate2)
        mem_label1_x.append(label1)
        mem_label2_x.append(label2)

        # intra RD
        head, tail, label = process_intra_rank(doc1)
        rd_head_ids1.append(head)
        rd_tail_ids1.append(tail)
        rd_label1.append(label)

        head, tail, label = process_intra_rank(doc2)
        rd_head_ids2.append(head)
        rd_tail_ids2.append(tail)
        rd_label2.append(label)

        # inter RD
        head1, tail1, head2, tail2, label = process_inter_rank(doc1, doc2)
        rd_head_ids1_x.append(head1)
        rd_head_ids2_x.append(head2)
        rd_tail_ids1_x.append(tail1)
        rd_tail_ids2_x.append(tail2)
        rd_label_x.append(label)

        # intra RFA
        query_h, query_t, head, tail, label = process_intra_rel(doc1)
        rfa_query_head1.append(query_h)
        rfa_query_tail1.append(query_t)
        rfa_candidate_head1.append(head)
        rfa_candidate_tail1.append(tail)
        rfa_label1.append(label)

        query_h, query_t, head, tail, label = process_intra_rel(doc2)
        rfa_query_head2.append(query_h)
        rfa_query_tail2.append(query_t)
        rfa_candidate_head2.append(head)
        rfa_candidate_tail2.append(tail)
        rfa_label2.append(label)

        # inter RFA
        rfa_query_head1_x.append(pair1[0])
        rfa_query_tail1_x.append(pair1[1])
        rfa_query_head2_x.append(pair2[0])
        rfa_query_tail2_x.append(pair2[1])

        head, tail, label = process_inter_rel(doc1, pair1)
        rfa_candidate_head1_x.append(head)
        rfa_candidate_tail1_x.append(tail)
        rfa_label21.append(label)

        head, tail, label = process_inter_rel(doc2, pair2)
        rfa_candidate_head2_x.append(head)
        rfa_candidate_tail2_x.append(tail)
        rfa_label12.append(label)

        return {
            'document1': torch.LongTensor(documents1),
            'document2': torch.LongTensor(documents2),
            'positions1': torch.LongTensor(positions1),
            'positions2': torch.LongTensor(positions2),
            'attn_mask1': torch.FloatTensor(attn_mask1),
            'attn_mask2': torch.FloatTensor(attn_mask2),

            'mem_query1': torch.LongTensor(mem_query1),
            'mem_query2': torch.LongTensor(mem_query2),
            'mem_candidate1': torch.LongTensor(mem_candidate1),
            'mem_candidate2': torch.LongTensor(mem_candidate2),
            'mem_label1': torch.LongTensor(mem_label1),
            'mem_label2': torch.LongTensor(mem_label2),

            'mem_query1_x': torch.LongTensor(mem_query1_x),
            'mem_query2_x': torch.LongTensor(mem_query2_x),
            'mem_candidate1_x': torch.LongTensor(mem_candidate1_x),
            'mem_candidate2_x': torch.LongTensor(mem_candidate2_x),
            'mem_label1_x': torch.LongTensor(mem_label1_x),
            'mem_label2_x': torch.LongTensor(mem_label2_x),

            'rd_head_ids1': torch.LongTensor(rd_head_ids1),
            'rd_head_ids2': torch.LongTensor(rd_head_ids2),
            'rd_tail_ids1': torch.LongTensor(rd_tail_ids1),
            'rd_tail_ids2': torch.LongTensor(rd_tail_ids2),
            'rd_label1': torch.LongTensor(rd_label1),
            'rd_label2': torch.LongTensor(rd_label2),

            'rd_head_ids1_x': torch.LongTensor(rd_head_ids1_x),
            'rd_head_ids2_x': torch.LongTensor(rd_head_ids2_x),
            'rd_tail_ids1_x': torch.LongTensor(rd_tail_ids1_x),
            'rd_tail_ids2_x': torch.LongTensor(rd_tail_ids2_x),
            'rd_label_x': torch.LongTensor(rd_label_x),

            'rfa_query_head1': torch.LongTensor(rfa_query_head1),
            'rfa_query_head2': torch.LongTensor(rfa_query_head2),
            'rfa_query_tail1': torch.LongTensor(rfa_query_tail1),
            'rfa_query_tail2': torch.LongTensor(rfa_query_tail2),
            'rfa_candidate_head1': torch.LongTensor(rfa_candidate_head1),
            'rfa_candidate_head2': torch.LongTensor(rfa_candidate_head2),
            'rfa_candidate_tail1': torch.LongTensor(rfa_candidate_tail1),
            'rfa_candidate_tail2': torch.LongTensor(rfa_candidate_tail2),
            'rfa_label1': torch.LongTensor(rfa_label1),
            'rfa_label2': torch.LongTensor(rfa_label2),

            'rfa_query_head1_x': torch.LongTensor(rfa_query_head1_x),
            'rfa_query_head2_x': torch.LongTensor(rfa_query_head2_x),
            'rfa_query_tail1_x': torch.LongTensor(rfa_query_tail1_x),
            'rfa_query_tail2_x': torch.LongTensor(rfa_query_tail2_x),
            'rfa_candidate_head1_x': torch.LongTensor(rfa_candidate_head1_x),
            'rfa_candidate_head2_x': torch.LongTensor(rfa_candidate_head2_x),
            'rfa_candidate_tail1_x': torch.LongTensor(rfa_candidate_tail1_x),
            'rfa_candidate_tail2_x': torch.LongTensor(rfa_candidate_tail2_x),
            'rfa_label12': torch.LongTensor(rfa_label12),
            'rfa_label21': torch.LongTensor(rfa_label21)
        }


def process_document(data, mode: str, mention_mask: list = None):
    entities = data['vertexSet']
    entity_num = len(entities)
    sentences = [[Config.tokenizer.tokenize(word) for word in sent] for sent in data['sents']]

    # mask some mentions for mention-entity linking
    if mention_mask is not None:
        for mention, eid in mention_mask:
            for pos in range(mention['pos'][0], mention['pos'][1]):
                sentences[mention['sent_id']][pos] = []
            sentences[mention['sent_id']][mention['pos'][0]].insert(0, '[unused200]')

    for eid, mentions in enumerate(entities):
        # blank mention by probability
        if random.random() > Config.blank_ratio:
            for mention in mentions:
                for pos in range(mention['pos'][0], mention['pos'][1]):
                    sentences[mention['sent_id']][pos] = []
                sentences[mention['sent_id']][mention['pos'][0]] = ['[unused0]']
        for mention in mentions:
            sentences[mention['sent_id']][mention['pos'][0]].insert(0, f'[unused{(eid << 1) + 1}]')
            sentences[mention['sent_id']][mention['pos'][1] - 1].append(f'[unused{(eid + 1) << 1}]')

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
        attn_mask = [1] * len(document) + [0] * (Config.token_padding - len(document))
        document += ['[PAD]'] * (Config.token_padding - len(document))
    else:
        document = document[:(Config.token_padding - 1)] + ['[SEP]']
        attn_mask = [1] * Config.token_padding

    positions = []
    for entity in entities:
        cur_entity = []
        for mention in entity:
            if word_position[mention['sent_id']][mention['pos'][0]] < 512:
                cur_entity.append(word_position[mention['sent_id']][mention['pos'][0]])
            if len(cur_entity) == Config.mention_padding:
                break
        positions.append(cur_entity)
    # padding length of mention number to 3
    for i in range(len(positions)):
        if len(positions[i]) == 0:
            positions[i] = [0] * Config.mention_padding
        positions[i] += [positions[i][0]] * (Config.mention_padding - len(positions[i]))
    while len(positions) < Config.entity_padding[mode]:
        positions.append([0] * Config.mention_padding)

    # for mention-entity linking
    mention_pos, mention_candidate, mention_label = [], [], []
    mention_limit, candidate_limit = Config.mention_sample_num, Config.mention_candidate_num
    if len(mention_mask) == 0:
        mention_pos = [0] * mention_limit
        mention_label = [-100] * mention_limit  # invalid label
        mention_candidate = [[0] * candidate_limit] * mention_limit
    for mention, eid in mention_mask:
        # entity number is never smaller than 5
        candidates = random.sample(range(entity_num), candidate_limit)
        if eid in candidates:
            mention_label.append(candidates.index(eid))
        else:
            rep_pos = random.randint(0, candidate_limit - 1)
            candidates[rep_pos] = eid
            mention_label.append(rep_pos)
        mention_candidate.append(candidates)

        if word_position[mention['sent_id']][mention['pos'][0]] < Config.token_padding:
            mention_pos.append(word_position[mention['sent_id']][mention['pos'][0]])
        else:
            mention_pos.append(0)
            mention_label[-1] = -100  # invalid

    return Config.tokenizer.convert_tokens_to_ids(document), \
        attn_mask, positions, mention_pos, mention_candidate, mention_label


def process_intra_mention(data):
    mention_limit = Config.mention_sample_num
    entities = data['vertexSet']
    # only consider those entities with more than 1 mention
    entity_ids = {eid for eid, mentions in enumerate(entities) if len(mentions) > 1}
    entity_ids = random.sample(entity_ids, min(mention_limit, len(entity_ids)))
    mention_masks = []  # [Tuple(mention, eid)]
    for eid in entity_ids:
        mid = random.randint(0, len(entities[eid]) - 1)
        mention_masks.append((entities[eid][mid], eid))
        entities[eid].pop(mid)
    if 0 < len(mention_masks) < mention_limit:
        mention_masks += [mention_masks[0]] * (mention_limit - len(mention_masks))
    return mention_masks


def process_inter_mention(data1, data2):
    name_to_eid = {}
    for eid, mentions in enumerate(data1['vertexSet']):
        for mention in mentions:
            name_to_eid[mention['name']] = eid
    entity_set = set()
    for eid, mentions in enumerate(data2['vertexSet']):
        for mention in mentions:
            if mention['name'] in name_to_eid:
                entity_set.add((name_to_eid[mention['name']], eid))
    entity_set = list(entity_set)

    if len(entity_set) == 0:  # no same entity
        label1 = label2 = -100  # invalid
        entity1 = entity2 = 0
        candidate1 = candidate2 = [0] * Config.negative_num
    else:
        entity1, entity2 = random.choice(entity_set)
        mid1 = list(range(len(data1['vertexSet'])))
        mid1.remove(entity1)
        mid2 = list(range(len(data2['vertexSet'])))
        mid2.remove(entity2)
        while len(mid1) < Config.negative_num:
            mid1 *= 2
        while len(mid2) < Config.negative_num:
            mid2 *= 2
        candidate1 = random.sample(mid1, Config.negative_num - 1)
        candidate2 = random.sample(mid2, Config.negative_num - 1)
        label1 = random.randint(0, Config.negative_num - 1)
        label2 = random.randint(0, Config.negative_num - 1)
        candidate1 = candidate1[:label1] + [entity1] + candidate1[label1:]
        candidate2 = candidate2[:label2] + [entity2] + candidate2[label2:]

    return entity1, entity2, candidate1, candidate2, label1, label2


def get_pos_neg_pairs(data, ret_dict=False):
    entities = data['vertexSet']
    entity_num = len(entities)
    positive_pairs = {(lab['h'], lab['t']): lab['r'] for lab in data['labels'] if lab['exist']}
    negative_pairs = [(i, j) for i in range(entity_num) for j in range(entity_num)
                      if i != j and (i, j) not in positive_pairs]
    while len(negative_pairs) < Config.negative_num:
        negative_pairs *= 2
    if ret_dict:
        return positive_pairs, negative_pairs
    else:
        return list(positive_pairs.keys()), negative_pairs


def process_intra_rank(data):
    positive_pairs, negative_pairs = get_pos_neg_pairs(data)

    sample_limit = Config.rd_sample_num
    head_ids, tail_ids, label_ids = [], [], []
    if len(positive_pairs) == 0:
        head_ids = tail_ids = [[0] * Config.negative_num] * sample_limit
        label_ids = [-100] * sample_limit
        return head_ids, tail_ids, label_ids

    for i in range(sample_limit):
        head_ids.append([])
        tail_ids.append([])

        negative_samples = random.sample(negative_pairs, Config.negative_num - 1)
        pos = random.randint(0, Config.negative_num - 1)
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
    if len(positive_pairs1) == 0 or len(positive_pairs2) == 0:
        head_ids1 = head_ids2 = tail_ids1 = tail_ids2 = [[0] * (Config.negative_num // 2)] * 3
        label_ids = [-100] * 3
        return head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids

    for i in range(Config.rd_sample_num):
        head_ids1.append([])
        tail_ids1.append([])
        head_ids2.append([])
        tail_ids2.append([])
        pairs1 = random.sample(negative_pairs1, Config.negative_num // 2)
        pairs2 = random.sample(negative_pairs2, Config.negative_num // 2)
        pos = random.randint(0, (Config.negative_num // 2) - 1)
        if random.random() < 0.5:
            label_ids.append(pos)
            pairs1[pos] = random.choice(positive_pairs1)
        else:
            label_ids.append(pos + Config.negative_num // 2)
            pairs2[pos] = random.choice(positive_pairs2)
        for pair in pairs1:
            head_ids1[-1].append(pair[0])
            tail_ids1[-1].append(pair[1])
        for pair in pairs2:
            head_ids2[-1].append(pair[0])
            tail_ids2[-1].append(pair[1])

    return head_ids1, tail_ids1, head_ids2, tail_ids2, label_ids


def process_inter_rel(data, candidate_pair):
    if candidate_pair == (0, 0):  # invalid
        return [0] * (Config.positive_num + Config.negative_num + 1), \
               [0] * (Config.positive_num + Config.negative_num + 1), -100

    positive_pairs, negative_pairs = get_pos_neg_pairs(data, ret_dict=True)

    positive_pairs = [pair for pair in positive_pairs.keys()
                      if positive_pairs[pair] != positive_pairs[candidate_pair]]
    if len(positive_pairs) == 0:
        positive_samples = random.sample(negative_pairs, Config.positive_num)
    else:
        while len(positive_pairs) < Config.positive_num:
            positive_pairs *= 2
        positive_samples = random.sample(positive_pairs, Config.positive_num)
    negative_samples = random.sample(negative_pairs, Config.negative_num)

    head_ids, tail_ids = [], []

    samples = [pair for pair in positive_samples] + [pair for pair in negative_samples]
    random.shuffle(samples)
    label = random.randint(0, len(samples))
    samples.insert(label, candidate_pair)
    for pair in samples:
        head_ids.append(pair[0])
        tail_ids.append(pair[1])
    return head_ids, tail_ids, label


def process_intra_rel(data):
    rel_to_labels = {}
    for lab in data['labels']:
        if not lab['exist']:
            continue
        if lab['r'] not in rel_to_labels:
            rel_to_labels[lab['r']] = []
        rel_to_labels[lab['r']].append(lab)
    for rel in [key for key in rel_to_labels.keys() if len(rel_to_labels[key]) < 2]:
        del rel_to_labels[rel]

    if len(rel_to_labels) == 0:
        label = -100  # invalid
        head_ids = tail_ids = [0] * (Config.positive_num + Config.negative_num + 1)
        query = (0, 0)
    else:
        rel = random.choice(list(rel_to_labels.keys()))
        target_pairs = random.sample(rel_to_labels[rel], 2)
        query = (target_pairs[0]['h'], target_pairs[0]['t'])

        positive_pairs, negative_pairs = get_pos_neg_pairs(data, ret_dict=True)

        positive_pairs = [pair for pair in positive_pairs.keys() if positive_pairs[pair] != rel]
        if len(positive_pairs) == 0:
            positive_samples = random.sample(negative_pairs, Config.positive_num)
        else:
            while len(positive_pairs) < Config.positive_num:
                positive_pairs *= 2
            positive_samples = random.sample(positive_pairs, Config.positive_num)
        negative_samples = random.sample(negative_pairs, Config.negative_num)

        samples = positive_samples + negative_samples
        random.shuffle(samples)
        label = random.randint(0, len(samples))
        samples.insert(label, (target_pairs[1]['h'], target_pairs[1]['t']))

        head_ids = [pair[0] for pair in samples]
        tail_ids = [pair[1] for pair in samples]

    return query[0], query[1], head_ids, tail_ids, label
