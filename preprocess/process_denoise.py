import random
import torch
from config import ConfigDenoise as Config


def process_denoise(data, mode: str):
    documents, attn_masks, word_positions, head_ids, tail_ids, pos_ids, titles = [], [], [], [], [], [], []
    for doc in data:
        document, attn_mask, position = process_document(doc, mode)
        documents.append(document)
        attn_masks.append(attn_mask)
        word_positions.append(position)

        if mode == 'test':
            entity_pad = Config.entity_padding[mode]
            entity_num = len(doc['vertexSet'])
            pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if i != j]
            pairs += [(0, 0)] * (entity_pad * (entity_pad - 1) - len(pairs))
            head_ids.append([pair[0] for pair in pairs])
            tail_ids.append([pair[1] for pair in pairs])
            titles.append(doc['title'])
        else:
            head_id, tail_id, pos_id = process_rank(doc)
            head_ids.append(head_id)
            tail_ids.append(tail_id)
            pos_ids.append(pos_id)

    if mode == 'test':
        return {
            'documents': torch.LongTensor(documents),
            'attn_mask': torch.FloatTensor(attn_masks),
            'word_pos': torch.LongTensor(word_positions),
            'head_ids': torch.LongTensor(head_ids),
            'tail_ids': torch.LongTensor(tail_ids),
            'titles': titles
        }
    else:
        return {
            'documents': torch.LongTensor(documents),
            'attn_mask': torch.FloatTensor(attn_masks),
            'word_pos': torch.LongTensor(word_positions),
            'head_ids': torch.LongTensor(head_ids),
            'tail_ids': torch.LongTensor(tail_ids),
            'labels': torch.LongTensor(pos_ids)
        }


def process_document(data, mode: str):
    sentences = [[Config.tokenizer.tokenize(word) for word in sent] for sent in data['sents']]

    entities = data['vertexSet']
    for i, mentions in enumerate(entities):
        for mention in mentions:
            sentences[mention['sent_id']][mention['pos'][0]].insert(0, f'[unused{(i << 1) + 1}]')
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

    # padding entity number to 42
    while len(positions) < Config.entity_padding[mode]:
        positions.append([0] * Config.mention_padding)

    return Config.tokenizer.convert_tokens_to_ids(document), attn_mask, positions


def process_rank(data):
    entity_num = len(data['vertexSet'])
    positive_pairs = set([(lab['h'], lab['t']) for lab in data['labels']])
    negative_pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if i != j
                      and (i, j) not in positive_pairs]
    while len(negative_pairs) < Config.negative_num:
        negative_pairs *= 2
    positive_pairs = list(positive_pairs)

    head_ids, tail_ids, pos_ids = [], [], []
    positive_samples = []
    while len(positive_samples) + len(positive_pairs) <= Config.positive_num:
        positive_samples += positive_pairs
        random.shuffle(positive_pairs)
    positive_samples += random.sample(positive_pairs, Config.positive_num - len(positive_samples))

    for i in range(Config.positive_num):
        head_ids.append([])
        tail_ids.append([])

        positive_sample = positive_samples[i]
        negative_samples = random.sample(negative_pairs, Config.negative_num)
        pos_p = random.randint(0, Config.negative_num)
        pos_ids.append(pos_p)
        for pair in (negative_samples[:pos_p] + [positive_sample] + negative_samples[pos_p:]):
            head_ids[-1].append(pair[0])
            tail_ids[-1].append(pair[1])

    return head_ids, tail_ids, pos_ids
