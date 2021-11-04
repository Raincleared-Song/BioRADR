from config import ConfigFineTune as Config
import numpy as np
import torch


def process_cdr(data, mode: str):
    # batch = [process_single(doc, mode) for doc in data]
    # max_len = max(len(doc['input_ids']) for doc in batch)
    # # batch padding only for text
    # input_ids = [doc['input_ids'] + [0] * (max_len - len(doc['input_ids'])) for doc in batch]
    # input_mask = [[1.0] * len(doc['input_ids']) + [0.0] * (max_len - len(doc['input_ids'])) for doc in batch]
    # labels = [doc['labels'] for doc in batch]
    # positions = [doc['positions'] for doc in batch]
    # pair_ids = [doc['pair_ids'] for doc in batch]
    #
    # return {
    #     'input_ids': torch.LongTensor(input_ids),
    #     'attn_mask': torch.FloatTensor(input_mask),
    #     'labels': labels,
    #     'entity_pos': positions,
    #     'pair_ids': pair_ids
    # }
    batch = []
    for doc in data:
        batch.append(process_single(doc, mode))
    return collate_fn(batch)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    titles = [f['title'] for f in batch]
    output = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'labels': labels,
        'entity_pos': entity_pos,
        'hts': hts,
        'titles': titles
    }
    return output


def process_single(data, mode: str):
    Config.token_padding *= 2
    entities = data['vertexSet']
    sentences = [[Config.tokenizer.tokenize(word) for word in sent] for sent in data['sents']]

    # TODO entity type marker
    for i, mentions in enumerate(entities):
        for mention in mentions:
            sentences[mention['sent_id']][mention['pos'][0]].insert(0, f'[unused{(i << 1) + 1}]')
            sentences[mention['sent_id']][mention['pos'][1] - 1].append(f'[unused{(i + 1) << 1}]')
            # sentences[mention['sent_id']][mention['pos'][0]].insert(0, '*')
            # sentences[mention['sent_id']][mention['pos'][1] - 1].append('*')

    word_position, document = [], ['[CLS]']
    for sent in sentences:
        word_position.append([])
        for word in sent:
            word_position[-1].append(len(document))
            document += word
    word_position.append([len(document)])

    # TODO long documents 1024, no padding here
    document.append('[SEP]')
    assert len(document) < Config.token_padding

    positions = []
    for entity in entities:
        cur_entity = []
        for mention in entity:
            assert word_position[mention['sent_id']][mention['pos'][0]] < Config.token_padding
            cur_entity.append((word_position[mention['sent_id']][mention['pos'][0]],
                               word_position[mention['sent_id']][mention['pos'][1]]))
        positions.append(cur_entity)

    label_num = len(data['labels'])
    # label_num * 2 for cdr
    label_mat = np.zeros((label_num, Config.relation_num))
    label_mat[:, Config.label2id['NA']] = 1
    positive_pairs = []  # hts
    if 'labels' in data:
        for lid, lab in enumerate(data['labels']):
            positive_pairs.append((lab['h'], lab['t']))
            if lab['r'] != 'NA':
                label_mat[lid, Config.label2id[lab['r']]] = 1
                label_mat[lid, Config.label2id['NA']] = 0
            assert entities[lab['h']][0]['type'] == 'Chemical' and entities[lab['t']][0]['type'] == 'Disease'

    return {
        'input_ids': Config.tokenizer.convert_tokens_to_ids(document),
        'entity_pos': positions,
        'labels': label_mat,
        'hts': positive_pairs,
        'title': data['pmid'],
    }
