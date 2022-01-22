import os
import torch
from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained('../huggingface/scibert_scivocab_cased' if os.path.exists(
    '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased')
CONFIG = {
    'use_type_marker': True,
    'token_padding': 1024,
    'mention_padding': 3,
    'entity_padding': 2
}


def process_denoise(data, mode: str):
    assert mode == 'test'

    documents, attn_masks, word_positions, head_ids, tail_ids, pos_ids, titles = [], [], [], [], [], [], []
    for doc in data:
        document, attn_mask, position = process_document(doc)
        documents.append(document)
        attn_masks.append(attn_mask)
        word_positions.append(position)

        entity_num = len(doc['vertexSet'])
        entities = doc['vertexSet']
        pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if
                 entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease']
        # 只有一个 pair
        assert len(pairs) == 1
        head_ids.append([pair[0] for pair in pairs])
        tail_ids.append([pair[1] for pair in pairs])
        titles.append(doc['pmid'])

    return {
        'documents': torch.LongTensor(documents),
        'attn_mask': torch.FloatTensor(attn_masks),
        'word_pos': torch.LongTensor(word_positions),
        'head_ids': torch.LongTensor(head_ids),
        'tail_ids': torch.LongTensor(tail_ids),
        'titles': titles
    }


def process_document(data):
    global CONFIG, TOKENIZER
    entity_padding, use_type_marker, token_padding, mention_padding = \
        CONFIG['entity_padding'], CONFIG['use_type_marker'], CONFIG['token_padding'], CONFIG['mention_padding']
    sentences = [[TOKENIZER.tokenize(word) for word in sent] for sent in data['sents']]

    entities = data['vertexSet']
    # must be only two entities
    assert len(entities) == entity_padding
    for i, mentions in enumerate(entities):
        for mention in mentions:
            tmp: list = sentences[mention['sent_id']][mention['pos'][0]]
            if use_type_marker:
                sentences[mention['sent_id']][mention['pos'][0]] = \
                    [f'[unused{(i << 1) + 1}]', mention['type'], '*'] + tmp
            else:
                sentences[mention['sent_id']][mention['pos'][0]] = [f'[unused{(i << 1) + 1}]'] + tmp
            sentences[mention['sent_id']][mention['pos'][1] - 1].append(f'[unused{(i + 1) << 1}]')

    word_position, document = [], ['[CLS]']
    for sent in sentences:
        word_position.append([])
        for word in sent:
            word_position[-1].append(len(document))
            document += word
    word_position.append([len(document)])

    # pad each document
    if len(document) < token_padding:
        document.append('[SEP]')
        document += ['[PAD]'] * (token_padding - len(document))
        attn_mask = [1] * len(document) + [0] * (token_padding - len(document))
    else:
        document = document[:(token_padding - 1)] + ['[SEP]']
        attn_mask = [1] * token_padding

    positions = []
    for entity in entities:
        cur_entity = []
        for mention in entity:
            if word_position[mention['sent_id']][mention['pos'][0]] < token_padding:
                cur_entity.append(word_position[mention['sent_id']][mention['pos'][0]])
            if len(cur_entity) == mention_padding:
                break
        positions.append(cur_entity)
    # padding length of mention number to 3
    for i in range(len(positions)):
        if len(positions[i]) == 0:
            positions[i] = [0] * mention_padding
        positions[i] += [positions[i][0]] * (mention_padding - len(positions[i]))

    return TOKENIZER.convert_tokens_to_ids(document), attn_mask, positions
